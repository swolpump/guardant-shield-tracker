#!/usr/bin/env python3
"""
Automated data updater for Guardant Shield Tracker.

Fetches live data from available APIs and updates data.json.
Run manually or via GitHub Actions on a schedule.

Data sources updated:
  1. Google Trends ("guardant shield") via pytrends - FREE, no API key
  2. SEC EDGAR XBRL total revenue - FREE, no API key
  3. Latest SEC filing metadata - FREE, no API key

Data sources NOT updated (no free API):
  - SimilarWeb traffic (paid API $499+/mo)
  - Shield page views / ordering clicks (internal analytics)
  - Shield test volumes / Shield revenue / ASP (quarterly earnings only)
  - Guidance / milestones (editorial)

Usage:
  pip install pytrends requests
  python update_data.py                    # Update data.json in parent dir
  python update_data.py --output ../data.json  # Custom output path
"""

import json
import time
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    sys.exit(1)


def fetch_google_trends(keyword="guardant shield", months=14):
    """Fetch Google Trends data using pytrends library."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("WARNING: pytrends not installed. Skipping Google Trends.")
        print("  Install with: pip install pytrends")
        return None

    print(f"Fetching Google Trends for '{keyword}'...")
    try:
        pytrends = TrendReq(hl='en-US', tz=300)
        # Build payload for the keyword
        pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='US')
        # Get interest over time
        df = pytrends.interest_over_time()

        if df.empty:
            print("WARNING: No Google Trends data returned.")
            return None

        # Extract monthly values
        monthly = df[keyword].resample('M').mean().round().astype(int)
        labels = [d.strftime("%b '%y") for d in monthly.index]
        values = monthly.tolist()

        # Trim to requested months
        labels = labels[-months:]
        values = values[-months:]

        print(f"  Got {len(values)} months of data. Latest: {values[-1]}/100")
        return {"months": labels, "values": values}

    except Exception as e:
        print(f"WARNING: Google Trends fetch failed: {e}")
        return None


def fetch_edgar_xbrl_revenue(cik="0001576280"):
    """Fetch total Guardant Health revenue from SEC EDGAR XBRL API."""
    print(f"Fetching EDGAR XBRL revenue for CIK {cik}...")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {
        "User-Agent": "ShieldTracker/1.0 (automated data updater)",
        "Accept": "application/json"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        facts = data.get("facts", {}).get("us-gaap", {})
        rev_concept = (
            facts.get("RevenueFromContractWithCustomerExcludingAssessedTax") or
            facts.get("Revenues") or
            facts.get("Revenue")
        )

        if not rev_concept:
            print("WARNING: No revenue concept found in XBRL data.")
            return None

        entries = rev_concept["units"]["USD"]

        # Filter for quarterly entries (~90 day periods)
        quarterly = []
        for e in entries:
            if not e.get("start") or not e.get("end"):
                continue
            start = datetime.strptime(e["start"], "%Y-%m-%d")
            end = datetime.strptime(e["end"], "%Y-%m-%d")
            days = (end - start).days
            if 80 <= days <= 100:
                quarterly.append(e)

        # Deduplicate: keep latest filing per period
        deduped = {}
        for e in quarterly:
            key = f"{e['start']}_{e['end']}"
            if key not in deduped or e["filed"] > deduped[key]["filed"]:
                deduped[key] = e

        quarterly = sorted(deduped.values(), key=lambda x: x["end"])

        result = []
        for e in quarterly[-12:]:  # Last 12 quarters
            end = datetime.strptime(e["end"], "%Y-%m-%d")
            q = (end.month - 1) // 3 + 1
            label = f"Q{q} '{str(end.year)[2:]}"
            result.append({
                "label": label,
                "value_millions": round(e["val"] / 1e6, 1),
                "filed": e["filed"],
                "end": e["end"]
            })

        print(f"  Got {len(result)} quarters. Latest: {result[-1]['label']} = ${result[-1]['value_millions']}M")
        return result

    except Exception as e:
        print(f"WARNING: EDGAR XBRL fetch failed: {e}")
        return None


def fetch_latest_sec_filing(cik="0001576280"):
    """Fetch latest SEC filing info."""
    print(f"Fetching latest SEC filing for CIK {cik}...")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": "ShieldTracker/1.0 (automated data updater)",
        "Accept": "application/json"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        recent = data["filings"]["recent"]
        target_forms = {"10-Q", "10-K", "8-K", "10-K/A", "10-Q/A"}

        for i in range(len(recent["form"])):
            if recent["form"][i] in target_forms:
                return {
                    "form": recent["form"][i],
                    "date": recent["filingDate"][i],
                    "accession": recent["accessionNumber"][i],
                    "primaryDoc": recent["primaryDocument"][i]
                }

        return None
    except Exception as e:
        print(f"WARNING: SEC filing fetch failed: {e}")
        return None


def update_data_json(data_path, trends_data=None, xbrl_data=None, filing_data=None):
    """Update data.json with new live data."""
    print(f"\nUpdating {data_path}...")

    with open(data_path, "r") as f:
        data = json.load(f)

    updated = False

    # Update Google Trends
    if trends_data:
        # Map trends months to the existing months array if possible
        existing_months = data["webMetrics"]["months"]
        existing_trends = data["webMetrics"]["googleTrends"]

        # Build a lookup from trends data
        trends_lookup = dict(zip(trends_data["months"], trends_data["values"]))

        # Update existing months where we have new data
        for i, month in enumerate(existing_months):
            if month in trends_lookup:
                existing_trends[i] = trends_lookup[month]

        # Add any new months not in existing data
        for month, value in zip(trends_data["months"], trends_data["values"]):
            if month not in existing_months:
                existing_months.append(month)
                existing_trends.append(value)
                # Also extend other web metrics arrays with None/0 placeholders
                data["webMetrics"]["siteVisits"].append(0)
                data["webMetrics"]["shieldPageViews"].append(0)
                data["webMetrics"]["orderingClicks"].append(0)

        # Update search interest KPI
        latest_trend = trends_data["values"][-1]
        prev_trend = trends_data["values"][-2] if len(trends_data["values"]) > 1 else latest_trend
        mom_change = round(((latest_trend - prev_trend) / prev_trend) * 100) if prev_trend else 0
        data["kpis"]["searchInterest"]["value"] = f"{latest_trend}/100"
        data["kpis"]["searchInterest"]["delta"] = f"+{mom_change}% MoM" if mom_change >= 0 else f"{mom_change}% MoM"

        updated = True
        print("  Updated Google Trends data")

    # Update XBRL revenue (store as metadata for cross-reference)
    if xbrl_data:
        data["_xbrlTotalRevenue"] = xbrl_data
        updated = True
        print("  Added XBRL total revenue cross-reference")

    # Update filing info
    if filing_data:
        data["_latestFiling"] = filing_data
        updated = True
        print("  Updated latest SEC filing info")

    # Update metadata
    if updated:
        data["meta"]["lastUpdated"] = datetime.now().strftime("%Y-%m-%d")
        data["meta"]["lastAutoUpdate"] = datetime.now().isoformat()

        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\ndata.json updated successfully at {data['meta']['lastUpdated']}")
    else:
        print("\nNo updates were made.")

    return updated


def main():
    parser = argparse.ArgumentParser(description="Update Guardant Shield Tracker data")
    parser.add_argument("--output", "-o", default=None, help="Path to data.json (default: ../data.json)")
    parser.add_argument("--skip-trends", action="store_true", help="Skip Google Trends fetch")
    parser.add_argument("--skip-xbrl", action="store_true", help="Skip EDGAR XBRL fetch")
    parser.add_argument("--keyword", default="guardant shield", help="Google Trends keyword")
    parser.add_argument("--cik", default="0001576280", help="SEC CIK number")
    args = parser.parse_args()

    # Determine data.json path
    if args.output:
        data_path = Path(args.output)
    else:
        data_path = Path(__file__).parent.parent / "data.json"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found.")
        sys.exit(1)

    print("=" * 50)
    print("Guardant Shield Tracker - Data Updater")
    print("=" * 50)
    print(f"Target: {data_path}")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Fetch all data
    trends_data = None if args.skip_trends else fetch_google_trends(args.keyword)
    time.sleep(1)  # Be polite to APIs

    xbrl_data = None if args.skip_xbrl else fetch_edgar_xbrl_revenue(args.cik)
    time.sleep(1)

    filing_data = fetch_latest_sec_filing(args.cik)

    # Update data.json
    updated = update_data_json(data_path, trends_data, xbrl_data, filing_data)

    if updated:
        print("\nDone! Commit and push data.json to update the live dashboard.")
    else:
        print("\nNo changes made.")


if __name__ == "__main__":
    main()
