from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COLORS = ["#1B4332", "#2D6A4F", "#40916C", "#74C69D", "#D8F3DC", "#F4A261"]
RANDOM_STATE = 42


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    figures = output_dir / "figures"
    results = output_dir / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return figures, results


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a_arr, b_arr, out=np.zeros_like(a_arr), where=np.abs(b_arr) > 1e-12)


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "transaction": "transaction_data.csv",
        "product": "product.csv",
        "campaign_desc": "campaign_desc.csv",
        "campaign_table": "campaign_table.csv",
        "coupon_redempt": "coupon_redempt.csv",
        "hh_demographic": "hh_demographic.csv",
    }
    data = {}
    for key, name in files.items():
        frame = pd.read_csv(data_dir / name, low_memory=False)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        data[key] = frame
    return data


def enrich_transactions(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trans = data["transaction"].copy()
    numeric = ["household_key", "basket_id", "day", "product_id", "quantity", "sales_value", "week_no", "retail_disc", "coupon_disc", "coupon_match_disc"]
    for col in numeric:
        trans[col] = pd.to_numeric(trans[col], errors="coerce")
    product = data["product"][["product_id", "department", "brand", "commodity_desc"]].copy()
    product["product_id"] = pd.to_numeric(product["product_id"], errors="coerce")
    trans = trans.merge(product, on="product_id", how="left", validate="many_to_one")
    trans["retail_discount_abs"] = trans["retail_disc"].fillna(0).abs()
    trans["coupon_discount_abs"] = trans[["coupon_disc", "coupon_match_disc"]].fillna(0).abs().sum(axis=1)
    trans["coupon_line"] = (trans["coupon_discount_abs"] > 0).astype(int)
    trans["private_label"] = trans["brand"].astype(str).str.strip().str.lower().eq("private").astype(int)
    return trans


def build_household_features(trans: pd.DataFrame) -> pd.DataFrame:
    valid = trans[(trans["quantity"] > 0) & (trans["sales_value"] > 0)].copy()
    basket = valid.groupby(["household_key", "basket_id"], as_index=False).agg(
        basket_sales=("sales_value", "sum"), basket_qty=("quantity", "sum")
    )
    basket_stats = basket.groupby("household_key", as_index=False).agg(
        avg_basket_value=("basket_sales", "mean"), avg_basket_qty=("basket_qty", "mean")
    )
    hh = valid.groupby("household_key", as_index=False).agg(
        total_sales=("sales_value", "sum"),
        total_qty=("quantity", "sum"),
        baskets=("basket_id", "nunique"),
        active_days=("day", "nunique"),
        active_weeks=("week_no", "nunique"),
        last_day=("day", "max"),
        unique_products=("product_id", "nunique"),
        unique_departments=("department", "nunique"),
        retail_discount=("retail_discount_abs", "sum"),
        coupon_discount=("coupon_discount_abs", "sum"),
        coupon_lines=("coupon_line", "sum"),
        private_label_share=("private_label", "mean"),
    ).merge(basket_stats, on="household_key", how="left")
    hh["recency_days"] = valid["day"].max() - hh["last_day"]
    hh["retail_discount_ratio"] = safe_div(hh["retail_discount"], hh["total_sales"])
    hh["coupon_discount_ratio"] = safe_div(hh["coupon_discount"], hh["total_sales"])
    hh["baskets_per_active_week"] = safe_div(hh["baskets"], hh["active_weeks"])
    return hh.replace([np.inf, -np.inf], np.nan).fillna(0)


def segment_households(hh: pd.DataFrame, results: Path, figures: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = [
        "recency_days", "total_sales", "baskets", "active_weeks", "unique_products",
        "avg_basket_value", "retail_discount_ratio", "coupon_discount_ratio",
        "private_label_share", "baskets_per_active_week",
    ]
    X = hh[features].copy()
    for col in ["recency_days", "total_sales", "baskets", "active_weeks", "unique_products", "avg_basket_value"]:
        X[col] = np.log1p(X[col].clip(lower=0))
    X_scaled = StandardScaler().fit_transform(X)
    score_rows = []
    for k in range(2, 7):
        candidate = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = candidate.fit_predict(X_scaled)
        score_rows.append({
            "k": k,
            "silhouette": float(silhouette_score(X_scaled, labels)),
            "inertia": float(candidate.inertia_),
        })
    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(results / "segment_silhouette_scores.csv", index=False)

    chosen_k = 4
    model = KMeans(n_clusters=chosen_k, n_init=30, random_state=RANDOM_STATE)
    segmented = hh.copy()
    segmented["cluster"] = model.fit_predict(X_scaled)

    seed_aris = []
    for seed in range(10):
        seed_labels = KMeans(n_clusters=chosen_k, n_init=30, random_state=seed).fit_predict(X_scaled)
        seed_aris.append(float(adjusted_rand_score(segmented["cluster"], seed_labels)))

    winsorized = hh[features].copy()
    for col in features:
        lower, upper = winsorized[col].quantile([0.01, 0.99])
        winsorized[col] = winsorized[col].clip(lower, upper)
    for col in ["recency_days", "total_sales", "baskets", "active_weeks", "unique_products", "avg_basket_value"]:
        winsorized[col] = np.log1p(winsorized[col].clip(lower=0))
    winsorized_scaled = StandardScaler().fit_transform(winsorized)
    winsorized_labels = KMeans(n_clusters=chosen_k, n_init=30, random_state=RANDOM_STATE).fit_predict(winsorized_scaled)
    stability = {
        "chosen_k": chosen_k,
        "selection_basis": "k=4 retained for distinct business profiles; k=2 has the strongest silhouette and is disclosed",
        "feature_scaling": "log1p on skewed activity/value variables followed by StandardScaler on all model features",
        "seed_stability_min_ari": float(min(seed_aris)),
        "seed_stability_mean_ari": float(np.mean(seed_aris)),
        "winsorized_1_99pct_ari_vs_primary": float(adjusted_rand_score(segmented["cluster"], winsorized_labels)),
        "outlier_note": "log transforms limit leverage; a 1st/99th percentile sensitivity run is reported but not used to overwrite observed values",
    }
    (results / "segment_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    profile = segmented.groupby("cluster", as_index=False).agg(
        households=("household_key", "count"),
        recency_days=("recency_days", "mean"),
        total_sales=("total_sales", "mean"),
        baskets=("baskets", "mean"),
        avg_basket_value=("avg_basket_value", "mean"),
        coupon_discount_ratio=("coupon_discount_ratio", "mean"),
        retail_discount_ratio=("retail_discount_ratio", "mean"),
        private_label_share=("private_label_share", "mean"),
    )
    high = int(profile.loc[profile["total_sales"].idxmax(), "cluster"])
    dormant = int(profile.loc[profile["recency_days"].idxmax(), "cluster"])
    remaining = [int(c) for c in profile["cluster"] if c not in {high, dormant}]
    promo = int(profile.set_index("cluster").loc[remaining, ["coupon_discount_ratio", "retail_discount_ratio"]].sum(axis=1).idxmax())
    mainstream = next(c for c in remaining if c != promo)
    name_map = {
        high: "High-value loyal shoppers",
        dormant: "Dormant / lapsed households",
        promo: "Promotion-driven routine shoppers",
        mainstream: "High-basket-value established shoppers",
    }
    segmented["segment_name"] = segmented["cluster"].map(name_map)
    profile["segment_name"] = profile["cluster"].map(name_map)
    profile.to_csv(results / "segment_profiles.csv", index=False)

    heat_cols = ["recency_days", "total_sales", "baskets", "avg_basket_value", "coupon_discount_ratio", "retail_discount_ratio"]
    heat = profile.set_index("segment_name")[heat_cols]
    heat_z = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(heat_z, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(heat_cols)), [c.replace("_", " ").title() for c in heat_cols], rotation=35, ha="right")
    ax.set_yticks(range(len(heat_z)), heat_z.index)
    ax.set_title("Four-Segment Customer Profile (Standardized Means)")
    fig.colorbar(im, ax=ax, label="Standard deviations from segment mean")
    savefig(figures / "01_customer_segment_profiles.png")

    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    ax2 = ax1.twinx()
    ax1.plot(score_df["k"], score_df["silhouette"], marker="o", linewidth=2, color=COLORS[1], label="Silhouette")
    ax2.plot(score_df["k"], score_df["inertia"], marker="s", linewidth=1.8, color=COLORS[5], label="Inertia")
    ax1.axvline(chosen_k, color="#111111", linestyle="--", linewidth=1.2, label="Chosen k=4")
    ax1.set(title="Customer Segment Selection Diagnostics", xlabel="Number of clusters (k)", ylabel="Silhouette score")
    ax2.set_ylabel("K-means inertia")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
    savefig(figures / "04_segment_selection.png")
    return segmented, score_df, stability


def build_campaign_features(data: dict[str, pd.DataFrame], trans: pd.DataFrame) -> pd.DataFrame:
    desc = data["campaign_desc"].copy()
    table = data["campaign_table"].copy()
    for frame in (desc, table):
        frame["campaign"] = pd.to_numeric(frame["campaign"], errors="coerce")
        frame["household_key"] = pd.to_numeric(frame.get("household_key"), errors="coerce") if "household_key" in frame else np.nan
    exposure = table.merge(
        desc[["campaign", "description", "start_day", "end_day"]],
        on=["campaign", "description"], how="left", validate="many_to_one",
    )
    redempt = data["coupon_redempt"].copy()
    redempt["household_key"] = pd.to_numeric(redempt["household_key"], errors="coerce")
    redempt["campaign"] = pd.to_numeric(redempt["campaign"], errors="coerce")
    redeemed_keys = set(zip(redempt["household_key"], redempt["campaign"]))
    exposure["redeemed"] = [int((hh, camp) in redeemed_keys) for hh, camp in zip(exposure["household_key"], exposure["campaign"])]

    trans_small = trans[(trans["quantity"] > 0) & (trans["sales_value"] > 0)][
        ["household_key", "day", "basket_id", "product_id", "quantity", "sales_value", "retail_discount_abs", "coupon_discount_abs", "coupon_line", "private_label"]
    ].sort_values(["household_key", "day"])
    groups = {int(hh): grp for hh, grp in trans_small.groupby("household_key", sort=False)}
    rows = []
    for row in exposure.itertuples(index=False):
        hh = int(row.household_key)
        start = float(row.start_day)
        history = groups.get(hh)
        if history is None:
            pre = history
        else:
            pre = history[history["day"] < start]
        if pre is None or len(pre) == 0:
            values = {
                "pre_total_sales": 0.0, "pre_total_qty": 0.0, "pre_baskets": 0,
                "pre_active_days": 0, "pre_unique_products": 0, "pre_avg_basket_value": 0.0,
                "pre_retail_discount_ratio": 0.0, "pre_coupon_discount_ratio": 0.0,
                "pre_coupon_line_share": 0.0, "pre_private_label_share": 0.0, "pre_recency_days": 999.0,
            }
        else:
            total_sales = float(pre["sales_value"].sum())
            basket_values = pre.groupby("basket_id")["sales_value"].sum()
            values = {
                "pre_total_sales": total_sales,
                "pre_total_qty": float(pre["quantity"].sum()),
                "pre_baskets": int(pre["basket_id"].nunique()),
                "pre_active_days": int(pre["day"].nunique()),
                "pre_unique_products": int(pre["product_id"].nunique()),
                "pre_avg_basket_value": float(basket_values.mean()),
                "pre_retail_discount_ratio": float(pre["retail_discount_abs"].sum() / total_sales) if total_sales else 0.0,
                "pre_coupon_discount_ratio": float(pre["coupon_discount_abs"].sum() / total_sales) if total_sales else 0.0,
                "pre_coupon_line_share": float(pre["coupon_line"].mean()),
                "pre_private_label_share": float(pre["private_label"].mean()),
                "pre_recency_days": float(start - pre["day"].max()),
            }
        values.update({
            "household_key": hh,
            "campaign": int(row.campaign),
            "description": str(row.description),
            "start_day": start,
            "redeemed": int(row.redeemed),
        })
        rows.append(values)
    return pd.DataFrame(rows)


def run_campaign_model(model_df: pd.DataFrame, results: Path, figures: Path) -> tuple[pd.DataFrame, dict]:
    target = "redeemed"
    numeric = [c for c in model_df.columns if c.startswith("pre_")] + ["start_day"]
    categorical = ["description"]
    X = model_df[numeric + categorical]
    y = model_df[target]
    groups = model_df["household_key"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    }
    metric_rows, fitted, test_probs, curves = [], {}, {}, {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("prep", prep), ("model", estimator)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metric_rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "pr_auc": float(average_precision_score(y_test, prob)),
            "brier": float(brier_score_loss(y_test, prob)),
        })
        fitted[name], test_probs[name], curves[name] = pipe, prob, roc_curve(y_test, prob)
    metrics = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics.to_csv(results / "campaign_model_metrics.csv", index=False)
    best_name = str(metrics.loc[0, "model"])
    best_prob = test_probs[best_name]

    ranked = pd.DataFrame({"actual": y_test.to_numpy(), "probability": best_prob}).sort_values("probability", ascending=False).reset_index(drop=True)
    ranked["decile"] = pd.qcut(ranked.index, 10, labels=np.arange(1, 11))
    deciles = ranked.groupby("decile", observed=True).agg(cases=("actual", "size"), responders=("actual", "sum"), response_rate=("actual", "mean")).reset_index()
    baseline = float(ranked["actual"].mean())
    deciles["lift"] = deciles["response_rate"] / baseline
    deciles.to_csv(results / "campaign_targeting_deciles.csv", index=False)
    top_20 = ranked.head(max(1, int(np.ceil(len(ranked) * 0.20))))
    targeting = {
        "test_observations": int(len(ranked)),
        "test_households": int(model_df.iloc[test_idx]["household_key"].nunique()),
        "baseline_response_rate": baseline,
        "top_decile_response_rate": float(deciles.loc[deciles["decile"] == 1, "response_rate"].iloc[0]),
        "top_decile_lift": float(deciles.loc[deciles["decile"] == 1, "lift"].iloc[0]),
        "top_20pct_responder_capture": float(top_20["actual"].sum() / ranked["actual"].sum()),
        "best_model": best_name,
    }
    (results / "campaign_targeting_summary.json").write_text(json.dumps(targeting, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for color, (name, (fpr, tpr, _)) in zip(COLORS, curves.items()):
        auc_value = metrics.loc[metrics["model"] == name, "roc_auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="#888888")
    ax.set(title="Campaign Redemption Model", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(frameon=False)
    savefig(figures / "02_campaign_model_roc.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(deciles["decile"].astype(str), deciles["response_rate"] * 100, color=COLORS[2])
    ax.axhline(baseline * 100, color="#111111", linestyle="--", label=f"Baseline {baseline:.1%}")
    ax.set(title="Observed Response by Propensity Decile", xlabel="Predicted propensity decile (1 = highest)", ylabel="Response rate (%)")
    ax.legend(frameon=False)
    savefig(figures / "03_campaign_decile_lift.png")

    rates = model_df.groupby("description")["redeemed"].agg(["mean", "count"]).reset_index()
    rates.to_csv(results / "campaign_response_by_type.csv", index=False)
    return metrics, targeting


def pre_campaign_value_response(model_df: pd.DataFrame, results: Path, figures: Path) -> tuple[pd.DataFrame, dict]:
    """Profile observed response by value tiers built only from pre-campaign behavior.

    The tier is exposure-specific and uses rate-normalized pre-period features so a
    later campaign does not automatically imply a higher cumulative value. All
    confidence intervals resample households, preserving repeated exposures.
    """
    frame = model_df.copy()
    exposure_days = frame["start_day"].clip(lower=30)
    frame["pre_sales_per_30d"] = frame["pre_total_sales"] / exposure_days * 30
    frame["pre_baskets_per_30d"] = frame["pre_baskets"] / exposure_days * 30
    score_inputs = pd.DataFrame({
        "sales_rate": np.log1p(frame["pre_sales_per_30d"].clip(lower=0)),
        "basket_rate": np.log1p(frame["pre_baskets_per_30d"].clip(lower=0)),
        "recency": -np.log1p(frame["pre_recency_days"].clip(lower=0)),
    })
    scaled = (score_inputs - score_inputs.mean()) / score_inputs.std(ddof=0).replace(0, 1)
    frame["pre_value_score"] = scaled.mean(axis=1)
    frame["pre_value_tier"] = pd.qcut(
        frame["pre_value_score"].rank(method="first"),
        q=3,
        labels=["Lower pre-campaign value", "Middle pre-campaign value", "Higher pre-campaign value"],
    )

    order = ["Lower pre-campaign value", "Middle pre-campaign value", "Higher pre-campaign value"]
    grouped = frame.groupby("pre_value_tier", observed=True).agg(
        exposures=("redeemed", "size"),
        households=("household_key", "nunique"),
        observed_responses=("redeemed", "sum"),
        observed_response_rate=("redeemed", "mean"),
        mean_pre_sales_per_30d=("pre_sales_per_30d", "mean"),
        mean_pre_baskets_per_30d=("pre_baskets_per_30d", "mean"),
        mean_pre_recency_days=("pre_recency_days", "mean"),
    ).reindex(order).reset_index()
    baseline = float(frame["redeemed"].mean())
    grouped["response_index_vs_all"] = grouped["observed_response_rate"] / baseline

    rng = np.random.default_rng(RANDOM_STATE)
    households = frame["household_key"].drop_duplicates().to_numpy()
    boot = {tier: [] for tier in order}
    household_groups = {hh: grp for hh, grp in frame.groupby("household_key", observed=True)}
    for _ in range(500):
        sampled = rng.choice(households, size=len(households), replace=True)
        sample = pd.concat([household_groups[hh] for hh in sampled], ignore_index=True)
        rates = sample.groupby("pre_value_tier", observed=True)["redeemed"].mean()
        for tier in order:
            if tier in rates:
                boot[tier].append(float(rates.loc[tier]))
    grouped["cluster_bootstrap_ci_low"] = [float(np.quantile(boot[t], 0.025)) for t in order]
    grouped["cluster_bootstrap_ci_high"] = [float(np.quantile(boot[t], 0.975)) for t in order]
    grouped.to_csv(results / "pre_campaign_value_response.csv", index=False)

    by_type = frame.groupby(["description", "pre_value_tier"], observed=True).agg(
        exposures=("redeemed", "size"),
        households=("household_key", "nunique"),
        observed_response_rate=("redeemed", "mean"),
    ).reset_index()
    by_type.to_csv(results / "pre_campaign_value_response_by_type.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    rates = grouped["observed_response_rate"].to_numpy()
    yerr = np.vstack([
        rates - grouped["cluster_bootstrap_ci_low"].to_numpy(),
        grouped["cluster_bootstrap_ci_high"].to_numpy() - rates,
    ])
    ax.bar(order, rates * 100, color=[COLORS[4], COLORS[2], COLORS[0]])
    ax.errorbar(order, rates * 100, yerr=yerr * 100, fmt="none", ecolor="#111111", capsize=5)
    ax.axhline(baseline * 100, linestyle="--", color="#111111", label=f"All exposures {baseline:.1%}")
    ax.set(title="Observed Campaign Response by Pre-Campaign Customer Value", xlabel="", ylabel="Observed redemption rate (%)")
    ax.tick_params(axis="x", rotation=12)
    ax.legend(frameon=False)
    savefig(figures / "08_pre_campaign_value_response.png")

    summary = {
        "definition": "Exposure-specific terciles of the mean standardized log sales rate, log basket rate, and inverse recency, all measured before campaign start",
        "baseline_observed_response_rate": baseline,
        "tiers": grouped.to_dict(orient="records"),
        "confidence_interval": "500-replicate percentile bootstrap resampling households with all repeated exposures",
        "interpretation": "descriptive observed response profiling; not randomized uplift or a causal promotion effect",
    }
    (results / "pre_campaign_value_response_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return grouped, summary


def segment_category_analysis(trans: pd.DataFrame, segmented: pd.DataFrame, results: Path, figures: Path) -> tuple[pd.DataFrame, dict]:
    """Describe category mix and basket penetration within retrospective segments."""
    valid = trans[(trans["quantity"] > 0) & (trans["sales_value"] > 0)].copy()
    valid = valid.merge(
        segmented[["household_key", "segment_name"]],
        on="household_key", how="inner", validate="many_to_one",
    )
    valid["department"] = valid["department"].astype(str).str.strip()
    valid = valid[~valid["department"].isin(["", "nan", "None"])]

    segment_totals = valid.groupby("segment_name").agg(
        segment_sales=("sales_value", "sum"),
        segment_baskets=("basket_id", "nunique"),
        segment_households=("household_key", "nunique"),
    )
    overall_share = valid.groupby("department")["sales_value"].sum()
    overall_share = overall_share / overall_share.sum()
    category = valid.groupby(["segment_name", "department"], as_index=False).agg(
        sales_value=("sales_value", "sum"),
        baskets=("basket_id", "nunique"),
        households=("household_key", "nunique"),
    )
    category = category.join(segment_totals, on="segment_name")
    category["sales_share_pct"] = category["sales_value"] / category["segment_sales"] * 100
    category["basket_penetration_pct"] = category["baskets"] / category["segment_baskets"] * 100
    category["household_penetration_pct"] = category["households"] / category["segment_households"] * 100
    category["overall_sales_share_pct"] = category["department"].map(overall_share) * 100
    category["sales_share_index_vs_all"] = category["sales_share_pct"] / category["overall_sales_share_pct"]
    category = category.sort_values(["segment_name", "sales_share_index_vs_all"], ascending=[True, False])
    category.to_csv(results / "segment_category_profiles.csv", index=False)

    eligible = category[(category["baskets"] >= 1_000) & (category["overall_sales_share_pct"] >= 1.0)]
    top_index = eligible.sort_values("sales_share_index_vs_all", ascending=False).groupby("segment_name", as_index=False).first()
    top_index.to_csv(results / "segment_category_top_overindex.csv", index=False)

    top_departments = valid.groupby("department")["sales_value"].sum().nlargest(8).index
    heat = category[category["department"].isin(top_departments)].pivot(
        index="segment_name", columns="department", values="sales_share_index_vs_all"
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    im = ax.imshow(heat.clip(upper=2.0), cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.5)
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Customer Segment Category Mix (Index vs All Households)")
    fig.colorbar(im, ax=ax, label="Sales-share index (1.0 = portfolio average)")
    savefig(figures / "09_segment_category_index.png")

    summary = {
        "method": "retrospective descriptive segment-by-department sales share and basket penetration",
        "eligible_top_overindex_rule": "at least 1,000 segment baskets and at least 1% overall sales share",
        "top_overindexed_department_by_segment": top_index[[
            "segment_name", "department", "sales_share_index_vs_all", "sales_share_pct", "basket_penetration_pct"
        ]].to_dict(orient="records"),
        "interpretation": "descriptive product-mix differences; not causal response or incremental sales",
    }
    (results / "segment_category_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return category, summary


def association_rules(trans: pd.DataFrame, results: Path, figures: Path) -> pd.DataFrame:
    valid = trans[(trans["quantity"] > 0) & (trans["sales_value"] > 0)].copy()
    valid["department"] = valid["department"].astype(str).str.strip()
    basket_dept = valid.loc[~valid["department"].isin(["", "nan", "None"]), ["basket_id", "department"]].drop_duplicates()
    basket_lists = basket_dept.groupby("basket_id")["department"].agg(list)
    n_baskets = len(basket_lists)
    single = Counter()
    pairs = Counter()
    for items in basket_lists:
        unique = sorted(set(items))
        single.update(unique)
        pairs.update(combinations(unique, 2))
    rows = []
    for (a, b), count in pairs.items():
        support = count / n_baskets
        if support < 0.01:
            continue
        for antecedent, consequent in ((a, b), (b, a)):
            confidence = count / single[antecedent]
            consequent_support = single[consequent] / n_baskets
            rows.append({
                "antecedent": antecedent,
                "consequent": consequent,
                "support": support,
                "confidence": confidence,
                "lift": confidence / consequent_support if consequent_support else np.nan,
                "basket_count": count,
            })
    rules = pd.DataFrame(rows).sort_values(["lift", "support"], ascending=False)
    rules.to_csv(results / "department_association_rules.csv", index=False)
    top = rules.head(12).sort_values("lift")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = top["antecedent"] + " -> " + top["consequent"]
    ax.barh(labels, top["lift"], color=COLORS[3])
    ax.axvline(1, color="#111111", linestyle="--")
    ax.set(title="Top Department Basket Affinities", xlabel="Lift", ylabel="")
    savefig(figures / "05_department_association_rules.png")
    return rules


def make_lag_features(values: list[float], week: int) -> list[float]:
    def lag(n: int) -> float:
        return values[-n] if len(values) >= n else values[0]
    return [
        lag(1), lag(2), lag(4), lag(8), lag(12),
        float(np.mean(values[-4:])), float(np.mean(values[-8:])),
        np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52),
    ]


def forecast_weekly_sales(trans: pd.DataFrame, results: Path, figures: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = trans[(trans["quantity"] > 0) & (trans["sales_value"] > 0) & trans["commodity_desc"].notna()].copy()
    top_commodities = valid.groupby("commodity_desc")["sales_value"].sum().nlargest(5).index.tolist()
    weekly = valid[valid["commodity_desc"].isin(top_commodities)].groupby(["commodity_desc", "week_no"], as_index=False)["sales_value"].sum()
    min_week, max_week = int(valid["week_no"].min()), int(valid["week_no"].max())
    test_weeks = 8
    metrics, predictions = [], []
    for commodity in top_commodities:
        series = weekly[weekly["commodity_desc"] == commodity].set_index("week_no")["sales_value"].reindex(range(min_week, max_week + 1), fill_value=0.0)
        values = series.to_numpy(float)
        train, test = values[:-test_weeks], values[-test_weeks:]
        test_week_numbers = series.index.to_numpy()[-test_weeks:]
        seasonal = values[-test_weeks - 52 : -52] if len(values) >= test_weeks + 52 else np.repeat(train[-1], test_weeks)
        moving = np.repeat(np.mean(train[-4:]), test_weeks)

        train_X, train_y = [], []
        history = train.tolist()
        for idx in range(12, len(train)):
            hist = train[:idx].tolist()
            train_X.append(make_lag_features(hist, int(series.index[idx])))
            train_y.append(train[idx])
        rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(np.asarray(train_X), np.asarray(train_y))
        recursive = history.copy()
        rf_pred = []
        for week in test_week_numbers:
            pred = max(0.0, float(rf.predict([make_lag_features(recursive, int(week))])[0]))
            rf_pred.append(pred)
            recursive.append(pred)

        model_preds = {"Seasonal Naive": np.asarray(seasonal), "Moving Average (4 weeks)": moving, "Random Forest": np.asarray(rf_pred)}
        for name, pred in model_preds.items():
            abs_error = np.abs(test - pred)
            metrics.append({
                "commodity": commodity,
                "model": name,
                "mae": float(mean_absolute_error(test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(test, pred))),
                "wape_pct": float(abs_error.sum() / np.abs(test).sum() * 100) if np.abs(test).sum() else np.nan,
                "abs_error_sum": float(abs_error.sum()),
                "actual_sum": float(np.abs(test).sum()),
            })
            for week, actual, predicted in zip(test_week_numbers, test, pred):
                predictions.append({"commodity": commodity, "week_no": int(week), "model": name, "actual_sales": actual, "predicted_sales": float(predicted)})

    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.DataFrame(predictions)
    metrics_df.to_csv(results / "forecast_metrics_by_commodity.csv", index=False)
    pred_df.to_csv(results / "forecast_predictions.csv", index=False)
    overall = metrics_df.groupby("model", as_index=False).agg(abs_error_sum=("abs_error_sum", "sum"), actual_sum=("actual_sum", "sum"), avg_rmse=("rmse", "mean"))
    overall["overall_wape_pct"] = overall["abs_error_sum"] / overall["actual_sum"] * 100
    overall = overall.sort_values("overall_wape_pct")
    overall.to_csv(results / "forecast_model_summary.csv", index=False)

    pivot = metrics_df.pivot(index="commodity", columns="model", values="wape_pct")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot, cmap="YlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Weekly Sales Forecast WAPE by Commodity")
    fig.colorbar(im, ax=ax, label="WAPE (%)")
    savefig(figures / "06_forecast_wape_heatmap.png")

    winners = metrics_df.sort_values("wape_pct").groupby("commodity", as_index=False).first()
    best_commodity = str(winners.sort_values("wape_pct").iloc[0]["commodity"])
    plot_data = pred_df[pred_df["commodity"] == best_commodity]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    actual = plot_data.drop_duplicates("week_no").sort_values("week_no")
    ax.plot(actual["week_no"], actual["actual_sales"], color="#111111", marker="o", linewidth=2.5, label="Actual")
    for color, (name, grp) in zip(COLORS, plot_data.groupby("model")):
        grp = grp.sort_values("week_no")
        ax.plot(grp["week_no"], grp["predicted_sales"], marker="o", label=name, color=color)
    ax.set(title=f"8-Week Holdout Forecast: {best_commodity}", xlabel="Week", ylabel="Weekly sales value")
    ax.legend(frameon=False)
    savefig(figures / "07_forecast_vs_actual.png")
    return metrics_df, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Retail customer, campaign and demand analytics")
    parser.add_argument("--data-dir", required=True, help="Directory containing dunnhumby CSV files")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    figures, results = ensure_dirs(output_dir)
    plt.style.use("seaborn-v0_8-whitegrid")

    data = load_data(data_dir)
    trans = enrich_transactions(data)
    data_quality = {
        "transaction_rows": int(len(trans)),
        "households": int(trans["household_key"].nunique()),
        "baskets": int(trans["basket_id"].nunique()),
        "products_in_transactions": int(trans["product_id"].nunique()),
        "weeks": int(trans["week_no"].nunique()),
        "negative_or_zero_quantity_rows": int((trans["quantity"] <= 0).sum()),
        "negative_or_zero_sales_rows": int((trans["sales_value"] <= 0).sum()),
        "exact_duplicate_transaction_rows": int(trans.duplicated().sum()),
        "missing_department_rows": int(trans["department"].isna().sum()),
        "campaign_exposures": int(len(data["campaign_table"])),
        "coupon_redemptions": int(len(data["coupon_redempt"])),
    }
    (results / "data_quality.json").write_text(json.dumps(data_quality, indent=2), encoding="utf-8")

    household = build_household_features(trans)
    segmented, silhouette, segment_stability = segment_households(household, results, figures)
    campaign_df = build_campaign_features(data, trans)
    campaign_metrics, targeting = run_campaign_model(campaign_df, results, figures)
    value_response, value_response_summary = pre_campaign_value_response(campaign_df, results, figures)
    segment_categories, segment_category_summary = segment_category_analysis(trans, segmented, results, figures)
    rules = association_rules(trans, results, figures)
    forecast_metrics, forecast_summary = forecast_weekly_sales(trans, results, figures)

    best_forecast = forecast_summary.iloc[0].to_dict()
    headline = {
        "dataset": data_quality,
        "segmentation": {
            "segments": 4,
            "k4_silhouette": float(silhouette.loc[silhouette["k"] == 4, "silhouette"].iloc[0]),
            "best_tested_k": int(silhouette.sort_values("silhouette", ascending=False).iloc[0]["k"]),
            "best_tested_silhouette": float(silhouette["silhouette"].max()),
            "seed_stability_min_ari": segment_stability["seed_stability_min_ari"],
            "winsorized_1_99pct_ari_vs_primary": segment_stability["winsorized_1_99pct_ari_vs_primary"],
            "note": "k=4 retained for four distinct business profiles; k=2 has the strongest silhouette and both inertia/silhouette diagnostics are disclosed.",
        },
        "campaign_model": {
            "best_model": targeting["best_model"],
            "best_metrics": campaign_metrics.iloc[0].to_dict(),
            **{k: v for k, v in targeting.items() if k != "best_model"},
            "split": "household-level group holdout to avoid the same household appearing in train and test",
        },
        "pre_campaign_value_response": value_response_summary,
        "segment_category_analysis": segment_category_summary,
        "basket_analysis": {
            "rules_exported": int(len(rules)),
            "minimum_support": 0.01,
            "top_rule": rules.iloc[0].to_dict() if len(rules) else None,
        },
        "forecast": {
            "commodities": int(forecast_metrics["commodity"].nunique()),
            "models": int(forecast_metrics["model"].nunique()),
            "holdout_weeks": 8,
            "best_overall_model": best_forecast["model"],
            "best_overall_wape_pct": float(best_forecast["overall_wape_pct"]),
        },
        "limitations": [
            "Campaign redemption is an observed response label, not causal incremental uplift.",
            "Pre-campaign value tiers and segment-category differences are descriptive and do not identify treatment effects.",
            "The four-cluster solution favors actionability over the highest silhouette score.",
            "Forecasting covers five high-sales commodities and one 8-week holdout window.",
            "Raw course-provided data are not redistributed in this portfolio version.",
        ],
    }
    (results / "headline_metrics.json").write_text(json.dumps(headline, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(headline, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
