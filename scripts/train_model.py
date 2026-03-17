#!/usr/bin/env python3
"""
ML Prediction Pipeline for Guardant Shield Tracker.

Trains a model on historical Shield data + public signals to predict
next-quarter test volumes and revenue with confidence intervals.

Architecture:
  - BayesianRidge regression (handles small datasets, gives uncertainty)
  - Leave-one-out cross-validation (maximize use of limited data)
  - Feature engineering from multiple public data sources
  - Ensemble of time-series + signal-based predictions

Data sources (all free, no API keys needed):
  1. Google Trends via pytrends (monthly search interest)
  2. SEC EDGAR XBRL (total company revenue by quarter)
  3. Historical data.json (Shield tests, revenue, ASP, web metrics)

Usage:
  pip install -r requirements.txt
  python train_model.py                        # Train and update data.json
  python train_model.py --dry-run              # Preview without saving
  python train_model.py --output ../data.json  # Custom path
"""

import json
import sys
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from math import sqrt

import numpy as np

try:
    from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests required. Install with: pip install requests")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================
# DATA FETCHING
# ============================================

def fetch_google_trends(keyword="guardant shield", months=18):
    """Fetch monthly Google Trends data via pytrends."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("  WARNING: pytrends not installed, using data.json values")
        return None

    print(f"  Fetching Google Trends for '{keyword}'...")
    try:
        pytrends = TrendReq(hl='en-US', tz=300)
        pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='US')
        df = pytrends.interest_over_time()
        if df.empty:
            return None

        monthly = df[keyword].resample('M').mean().round().astype(int)
        return {
            "dates": [d.strftime("%Y-%m") for d in monthly.index],
            "values": monthly.tolist()
        }
    except Exception as e:
        print(f"  WARNING: Google Trends failed: {e}")
        return None


def fetch_edgar_xbrl(cik="0001576280"):
    """Fetch quarterly total revenue from EDGAR XBRL."""
    print(f"  Fetching EDGAR XBRL revenue (CIK {cik})...")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": "ShieldTracker/1.0", "Accept": "application/json"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        facts = data.get("facts", {}).get("us-gaap", {})

        rev_concept = (
            facts.get("RevenueFromContractWithCustomerExcludingAssessedTax") or
            facts.get("Revenues")
        )
        if not rev_concept:
            return None

        entries = rev_concept["units"]["USD"]
        quarterly = []
        for e in entries:
            if not e.get("start") or not e.get("end"):
                continue
            from datetime import datetime as dt
            days = (dt.strptime(e["end"], "%Y-%m-%d") - dt.strptime(e["start"], "%Y-%m-%d")).days
            if 80 <= days <= 100:
                quarterly.append(e)

        # Deduplicate
        deduped = {}
        for e in quarterly:
            key = f"{e['start']}_{e['end']}"
            if key not in deduped or e["filed"] > deduped[key]["filed"]:
                deduped[key] = e

        result = []
        for e in sorted(deduped.values(), key=lambda x: x["end"]):
            end = datetime.strptime(e["end"], "%Y-%m-%d")
            q = (end.month - 1) // 3 + 1
            fy = end.year
            result.append({
                "quarter": f"Q{q}",
                "year": fy,
                "label": f"Q{q} '{str(fy)[2:]}",
                "value_millions": round(e["val"] / 1e6, 1)
            })

        return result[-16:]  # Last 16 quarters
    except Exception as e:
        print(f"  WARNING: EDGAR XBRL failed: {e}")
        return None


# ============================================
# FEATURE ENGINEERING
# ============================================

def build_training_data(data, trends_data=None, xbrl_data=None):
    """
    Build feature matrix and targets from historical data.

    Features per quarter:
      1. quarter_index     - Time trend (0, 1, 2, ...)
      2. quarter_index_sq  - Quadratic time trend (exponential growth capture)
      3. google_trends_avg - Average Google Trends for the quarter
      4. trends_momentum   - Change in Google Trends vs prior quarter
      5. prior_tests       - Previous quarter's test volume
      6. prior_tests_growth- Growth rate from 2 quarters ago to prior quarter
      7. web_clicks_avg    - Average ordering clicks for the quarter months
      8. clicks_momentum   - Change in clicks vs prior quarter
      9. is_q4             - Seasonality flag (Q4 tends to be strongest)
     10. total_rev_ratio   - Shield rev as % of total GH rev (if XBRL available)
    """

    quarters = data["quarters"]
    tests = data["shieldTests"]
    revenue = data["shieldRevenue"]
    asp = data["asp"]
    web = data["webMetrics"]
    n_q = len(quarters)

    # Map monthly web data to quarters
    # Assume months align: each quarter = 3 months
    # web data starts Jan '25 (Q1 '25 starts at index 4 in quarters)
    monthly_clicks = web["orderingClicks"]
    monthly_visits = web["siteVisits"]
    monthly_trends = web["googleTrends"]
    n_months = len(monthly_clicks)

    # Override Google Trends with live data if available
    if trends_data:
        # Align by date
        live_trends = dict(zip(trends_data["dates"], trends_data["values"]))
        print(f"  Live Google Trends: {len(live_trends)} months")

    # Build quarterly aggregates for web metrics
    # We know web data starts Jan '25, which is Q1 '25 (index 4 in our quarters array)
    # So quarters 0-3 (Q1'24-Q4'24) have no web data
    web_start_quarter = 4  # Q1 '25

    def quarterly_avg(monthly_arr, q_offset):
        """Get average of 3 months for a given quarter offset from web start."""
        start_month = q_offset * 3
        end_month = start_month + 3
        if end_month > len(monthly_arr):
            end_month = len(monthly_arr)
        if start_month >= len(monthly_arr):
            return None
        vals = monthly_arr[start_month:end_month]
        return np.mean(vals) if vals else None

    # Build XBRL revenue lookup
    xbrl_lookup = {}
    if xbrl_data:
        for entry in xbrl_data:
            xbrl_lookup[entry["label"]] = entry["value_millions"]

    # Feature matrix
    X = []
    y_tests = []
    y_revenue = []
    feature_names = [
        "quarter_index", "quarter_index_sq", "google_trends_avg",
        "trends_momentum", "prior_tests", "prior_tests_growth",
        "web_clicks_avg", "clicks_momentum", "is_q4", "total_rev_ratio"
    ]

    for i in range(1, n_q):  # Start at 1 to have prior quarter
        q_label = quarters[i]

        # 1. Time trend
        q_idx = i
        q_idx_sq = i * i

        # 2. Google Trends (quarterly average)
        web_q_offset = i - web_start_quarter
        if web_q_offset >= 0:
            gt_avg = quarterly_avg(monthly_trends, web_q_offset)
            gt_prev = quarterly_avg(monthly_trends, web_q_offset - 1) if web_q_offset > 0 else None
        else:
            gt_avg = None
            gt_prev = None

        # Use imputation for missing Google Trends (early quarters)
        if gt_avg is None:
            # Estimate from time trend: assume linear growth
            gt_avg = 30 + i * 5  # Rough estimate
            gt_prev = 30 + (i-1) * 5

        # 3. Trends momentum
        trends_mom = (gt_avg - gt_prev) / gt_prev * 100 if gt_prev and gt_prev > 0 else 0

        # 4. Prior quarter tests
        prior_tests = tests[i - 1]
        prior_growth = 0
        if i >= 2 and tests[i-2] > 0:
            prior_growth = (tests[i-1] - tests[i-2]) / tests[i-2] * 100

        # 5. Web clicks
        if web_q_offset >= 0:
            clicks_avg = quarterly_avg(monthly_clicks, web_q_offset) or 0
            clicks_prev = quarterly_avg(monthly_clicks, web_q_offset - 1) if web_q_offset > 0 else None
        else:
            clicks_avg = 0
            clicks_prev = None
        clicks_mom = (clicks_avg - clicks_prev) / clicks_prev * 100 if clicks_prev and clicks_prev > 0 else 0

        # 6. Seasonality
        is_q4 = 1 if "'24" in q_label and "Q4" in q_label or "'25" in q_label and "Q4" in q_label else 0
        # More robust Q4 detection
        if q_label.startswith("Q4"):
            is_q4 = 1

        # 7. Total revenue ratio
        total_rev = xbrl_lookup.get(q_label, 0)
        shield_rev = revenue[i]
        rev_ratio = shield_rev / total_rev * 100 if total_rev > 0 else 0

        features = [
            q_idx, q_idx_sq, gt_avg, trends_mom, prior_tests,
            prior_growth, clicks_avg, clicks_mom, is_q4, rev_ratio
        ]

        X.append(features)
        y_tests.append(tests[i])
        y_revenue.append(revenue[i])

    X = np.array(X, dtype=float)
    y_tests = np.array(y_tests, dtype=float)
    y_revenue = np.array(y_revenue, dtype=float)

    return X, y_tests, y_revenue, feature_names


def build_prediction_features(data, trends_data=None, xbrl_data=None):
    """Build feature vector for the next quarter (Q1 '26) prediction."""

    quarters = data["quarters"]
    tests = data["shieldTests"]
    revenue = data["shieldRevenue"]
    web = data["webMetrics"]
    n_q = len(quarters)

    # Next quarter index
    q_idx = n_q  # 8
    q_idx_sq = q_idx * q_idx

    # Google Trends: use latest 3 months
    monthly_trends = web["googleTrends"]
    if trends_data and len(trends_data["values"]) >= 3:
        gt_avg = np.mean(trends_data["values"][-3:])
        gt_prev = np.mean(trends_data["values"][-6:-3]) if len(trends_data["values"]) >= 6 else np.mean(monthly_trends[-3:])
    else:
        gt_avg = np.mean(monthly_trends[-3:]) if len(monthly_trends) >= 3 else monthly_trends[-1]
        gt_prev = np.mean(monthly_trends[-6:-3]) if len(monthly_trends) >= 6 else gt_avg * 0.9

    trends_mom = (gt_avg - gt_prev) / gt_prev * 100 if gt_prev > 0 else 0

    # Prior quarter (Q4 '25)
    prior_tests = tests[-1]
    prior_growth = (tests[-1] - tests[-2]) / tests[-2] * 100 if tests[-2] > 0 else 0

    # Web clicks: latest 3 months
    monthly_clicks = web["orderingClicks"]
    clicks_avg = np.mean(monthly_clicks[-3:]) if len(monthly_clicks) >= 3 else monthly_clicks[-1]
    clicks_prev = np.mean(monthly_clicks[-6:-3]) if len(monthly_clicks) >= 6 else clicks_avg * 0.8
    clicks_mom = (clicks_avg - clicks_prev) / clicks_prev * 100 if clicks_prev > 0 else 0

    # Q1 is not Q4
    is_q4 = 0

    # Total revenue ratio: use latest
    xbrl_lookup = {}
    if xbrl_data:
        for entry in xbrl_data:
            xbrl_lookup[entry["label"]] = entry["value_millions"]
    last_total = xbrl_lookup.get(quarters[-1], 0)
    last_shield = revenue[-1]
    rev_ratio = last_shield / last_total * 100 if last_total > 0 else 0

    features = [
        q_idx, q_idx_sq, gt_avg, trends_mom, prior_tests,
        prior_growth, clicks_avg, clicks_mom, is_q4, rev_ratio
    ]

    return np.array([features], dtype=float)


# ============================================
# MODEL TRAINING
# ============================================

def train_and_predict(X, y, X_pred, feature_names, target_name="tests"):
    """
    Train BayesianRidge + Ridge ensemble with LOO-CV.

    Returns prediction with confidence intervals and diagnostics.
    """
    n_samples = len(y)
    print(f"\n  Training {target_name} model on {n_samples} samples, {X.shape[1]} features...")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_pred_scaled = scaler.transform(X_pred)

    # ---- Model 1: BayesianRidge (gives uncertainty estimates) ----
    bayesian = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True
    )
    bayesian.fit(X_scaled, y)
    bay_pred, bay_std = bayesian.predict(X_pred_scaled, return_std=True)

    # ---- Model 2: Ridge (more stable with small data) ----
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    ridge_pred = ridge.predict(X_pred_scaled)

    # ---- Model 3: ElasticNet (feature selection) ----
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
    enet.fit(X_scaled, y)
    enet_pred = enet.predict(X_pred_scaled)

    # ---- Ensemble: weighted average ----
    # BayesianRidge gets highest weight (best for uncertainty)
    weights = [0.5, 0.3, 0.2]  # Bayesian, Ridge, ElasticNet
    ensemble_pred = (
        weights[0] * bay_pred[0] +
        weights[1] * ridge_pred[0] +
        weights[2] * enet_pred[0]
    )

    # ---- Leave-One-Out Cross-Validation ----
    loo = LeaveOneOut()
    loo_preds = []
    loo_actuals = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        bay_cv = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        bay_cv.fit(X_train, y_train)
        pred = bay_cv.predict(X_test)[0]

        loo_preds.append(pred)
        loo_actuals.append(y_test[0])

    loo_preds = np.array(loo_preds)
    loo_actuals = np.array(loo_actuals)

    # CV metrics
    mae = mean_absolute_error(loo_actuals, loo_preds)
    rmse = sqrt(mean_squared_error(loo_actuals, loo_preds))
    r2 = r2_score(loo_actuals, loo_preds)
    mape = np.mean(np.abs((loo_actuals - loo_preds) / loo_actuals)) * 100

    # Residual standard deviation for confidence intervals
    residuals = loo_actuals - loo_preds
    residual_std = np.std(residuals)

    # Combined uncertainty: model uncertainty + residual uncertainty
    total_std = sqrt(bay_std[0]**2 + residual_std**2)

    # Confidence intervals
    ci_80 = 1.282 * total_std  # 80% CI
    ci_95 = 1.960 * total_std  # 95% CI

    # Feature importance (from BayesianRidge coefficients)
    coefs = bayesian.coef_
    importance = np.abs(coefs)
    importance_pct = importance / importance.sum() * 100

    feature_importance = sorted(
        zip(feature_names, coefs.tolist(), importance_pct.tolist()),
        key=lambda x: abs(x[1]), reverse=True
    )

    print(f"  LOO-CV Results:")
    print(f"    RÂ²:   {r2:.3f}")
    print(f"    MAE:  {mae:,.0f}")
    print(f"    RMSE: {rmse:,.0f}")
    print(f"    MAPE: {mape:.1f}%")
    print(f"  Ensemble prediction: {ensemble_pred:,.0f}")
    print(f"  80% CI: [{ensemble_pred - ci_80:,.0f}, {ensemble_pred + ci_80:,.0f}]")
    print(f"  95% CI: [{ensemble_pred - ci_95:,.0f}, {ensemble_pred + ci_95:,.0f}]")
    print(f"  Individual models: Bayesian={bay_pred[0]:,.0f}, Ridge={ridge_pred[0]:,.0f}, ElasticNet={enet_pred[0]:,.0f}")

    return {
        "pointEstimate": round(float(ensemble_pred)),
        "ci80": {"low": round(float(ensemble_pred - ci_80)), "high": round(float(ensemble_pred + ci_80))},
        "ci95": {"low": round(float(ensemble_pred - ci_95)), "high": round(float(ensemble_pred + ci_95))},
        "individualModels": {
            "bayesianRidge": round(float(bay_pred[0])),
            "ridge": round(float(ridge_pred[0])),
            "elasticNet": round(float(enet_pred[0]))
        },
        "uncertainty": {
            "modelStd": round(float(bay_std[0])),
            "residualStd": round(float(residual_std)),
            "totalStd": round(float(total_std))
        },
        "crossValidation": {
            "method": "Leave-One-Out",
            "nSamples": int(n_samples),
            "r2": round(float(r2), 3),
            "mae": round(float(mae)),
            "rmse": round(float(rmse)),
            "mape": round(float(mape), 1),
            "predictions": [
                {"quarter": f"Sample {i+1}", "actual": round(float(a)), "predicted": round(float(p))}
                for i, (a, p) in enumerate(zip(loo_actuals, loo_preds))
            ]
        },
        "featureImportance": [
            {"feature": name, "coefficient": round(float(coef), 2), "importancePct": round(float(imp), 1)}
            for name, coef, imp in feature_importance
        ]
    }


# ============================================
# ASP & REVENUE DERIVATION
# ============================================

def predict_asp(data):
    """
    Predict ASP for next quarter using trend analysis.
    Uses exponential smoothing on historical ASP values.
    """
    asp_history = data["asp"]

    # Exponential weighted average (more weight on recent)
    weights = np.array([0.5 ** (len(asp_history) - 1 - i) for i in range(len(asp_history))])
    weights = weights / weights.sum()
    trend_asp = np.average(asp_history, weights=weights)

    # Also factor in management guidance for Medicare mix shift (~5% decline)
    latest_asp = asp_history[-1]
    guided_asp = latest_asp * 0.95

    # Blend: 60% guided, 40% trend
    predicted_asp = 0.6 * guided_asp + 0.4 * trend_asp

    return {
        "predicted": round(float(predicted_asp)),
        "trendBased": round(float(trend_asp)),
        "guidanceBased": round(float(guided_asp)),
        "latestActual": asp_history[-1],
        "method": "Exponential smoothing blended with management guidance (Medicare mix shift)"
    }


# ============================================
# MAIN PIPELINE
# ============================================

def run_pipeline(data_path, dry_run=False):
    """Execute the full ML prediction pipeline."""

    print("=" * 60)
    print("GUARDANT SHIELD TRACKER - ML PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Data: {data_path}")
    print()

    # Load data.json
    with open(data_path) as f:
        data = json.load(f)

    print("1. FETCHING LIVE DATA")
    print("-" * 40)

    # Fetch Google Trends
    keyword = data.get("config", {}).get("googleTrendsKeyword", "guardant shield")
    trends_data = fetch_google_trends(keyword)

    # Fetch EDGAR XBRL
    cik = data.get("config", {}).get("cik", "0001576280")
    xbrl_data = fetch_edgar_xbrl(cik)

    print()
    print("2. BUILDING FEATURES")
    print("-" * 40)

    # Build training data
    X, y_tests, y_revenue, feature_names = build_training_data(data, trends_data, xbrl_data)
    print(f"  Training samples: {len(y_tests)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")
    print(f"  Test volume range: {y_tests.min():,.0f} - {y_tests.max():,.0f}")
    print(f"  Revenue range: ${y_revenue.min():.1f}M - ${y_revenue.max():.1f}M")

    # Build prediction features for Q1 '26
    X_pred = build_prediction_features(data, trends_data, xbrl_data)
    print(f"  Prediction features (Q1 '26): {X_pred[0].tolist()}")

    print()
    print("3. TRAINING MODELS")
    print("-" * 40)

    # Train test volume model
    tests_result = train_and_predict(X, y_tests, X_pred, feature_names, "Shield Tests")

    # Train revenue model
    rev_result = train_and_predict(X, y_revenue, X_pred, feature_names, "Shield Revenue ($M)")

    # Predict ASP
    asp_result = predict_asp(data)

    # Derive revenue from tests Ã ASP as cross-check
    derived_revenue = tests_result["pointEstimate"] * asp_result["predicted"] / 1e6
    print(f"\n  ASP Prediction: ${asp_result['predicted']}")
    print(f"  Derived Revenue (tests Ã ASP): ${derived_revenue:.1f}M")
    print(f"  Direct Revenue Model: ${rev_result['pointEstimate'] / 1e6:.1f}M" if rev_result['pointEstimate'] > 1000 else f"  Direct Revenue Model: ${rev_result['pointEstimate']:.1f}M")

    # Revenue is in $M in training, so result should be in $M
    print()
    print("4. PREDICTION SUMMARY")
    print("-" * 40)

    # Determine target quarter
    target_quarter = "Q1 '26"
    guidance_low = data["guidance2026"]["quarterlyTestsLow"][0]
    guidance_high = data["guidance2026"]["quarterlyTestsHigh"][0]
    guidance_mid = (guidance_low + guidance_high) / 2

    verdict = "ABOVE" if tests_result["pointEstimate"] > guidance_mid else "BELOW" if tests_result["pointEstimate"] < guidance_low else "IN LINE WITH"

    print(f"  Target: {target_quarter}")
    print(f"  ML Test Prediction:  {tests_result['pointEstimate']:,} tests")
    print(f"  80% CI:              [{tests_result['ci80']['low']:,}, {tests_result['ci80']['high']:,}]")
    print(f"  95% CI:              [{tests_result['ci95']['low']:,}, {tests_result['ci95']['high']:,}]")
    print(f"  Company Guidance:    [{guidance_low:,}, {guidance_high:,}]")
    print(f"  Verdict:             {verdict} GUIDANCE")
    print(f"  Revenue Prediction:  ${rev_result['pointEstimate']:.1f}M")
    print(f"  ASP Prediction:      ${asp_result['predicted']}")

    # Assemble model prediction object
    model_prediction = {
        "targetQuarter": target_quarter,
        "generatedAt": datetime.now().isoformat(),
        "dataPoints": int(len(y_tests)),
        "verdict": verdict,
        "tests": tests_result,
        "revenue": rev_result,
        "asp": asp_result,
        "derivedRevenue": round(float(derived_revenue), 1),
        "guidance": {
            "low": guidance_low,
            "high": guidance_high,
            "mid": round(guidance_mid)
        },
        "dataSources": {
            "googleTrends": "live" if trends_data else "data.json",
            "edgarXBRL": "live" if xbrl_data else "unavailable",
            "webMetrics": "data.json",
            "historicalData": "data.json"
        },
        "modelArchitecture": {
            "type": "Ensemble (BayesianRidge + Ridge + ElasticNet)",
            "weights": {"bayesianRidge": 0.5, "ridge": 0.3, "elasticNet": 0.2},
            "crossValidation": "Leave-One-Out",
            "scaler": "StandardScaler",
            "features": feature_names
        }
    }

    if dry_run:
        print("\n  DRY RUN - not saving to data.json")
        print(json.dumps(model_prediction, indent=2))
    else:
        # Save to data.json
        data["modelPrediction"] = model_prediction
        data["meta"]["lastModelRun"] = datetime.now().isoformat()
        data["meta"]["lastUpdated"] = datetime.now().strftime("%Y-%m-%d")

        # Also update Google Trends in webMetrics if we got live data
        if trends_data:
            # Update the googleTrends array in webMetrics
            existing_months = data["webMetrics"]["months"]
            for date_str, value in zip(trends_data["dates"], trends_data["values"]):
                # Convert 2025-01 to "Jan '25"
                dt = datetime.strptime(date_str, "%Y-%m")
                label = dt.strftime("%b '%y")
                if label in existing_months:
                    idx = existing_months.index(label)
                    data["webMetrics"]["googleTrends"][idx] = value

        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved model predictions to {data_path}")

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return model_prediction


def main():
    parser = argparse.ArgumentParser(description="Shield Tracker ML Prediction Pipeline")
    parser.add_argument("--output", "-o", default=None, help="Path to data.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--keyword", default=None, help="Google Trends keyword override")
    args = parser.parse_args()

    if args.output:
        data_path = Path(args.output)
    else:
        data_path = Path(__file__).parent.parent / "data.json"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        sys.exit(1)

    # Override keyword if specified
    if args.keyword:
        with open(data_path) as f:
            d = json.load(f)
        d.setdefault("config", {})["googleTrendsKeyword"] = args.keyword
        with open(data_path, "w") as f:
            json.dump(d, f, indent=2)

    run_pipeline(data_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
