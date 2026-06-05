from flask import Flask, render_template, request, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import pandas as pd
import os
import pickle
import uuid
from datetime import datetime

from utils import preprocess_data
from model import run_full_pipeline

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = "uploads"
STATE_FOLDER = os.path.join(UPLOAD_FOLDER, "session_state")
REPORT_PREVIEW_ROWS = 200
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATE_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def default_state():
    return {
        "df": None,
        "result": None,
        "result_days": None,
        "days": 7,
    }


def get_session_id():
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return sid


def get_state_path():
    return os.path.join(STATE_FOLDER, f"{get_session_id()}.pkl")


def load_state():
    filepath = get_state_path()
    if not os.path.exists(filepath):
        return default_state()

    try:
        with open(filepath, "rb") as file_obj:
            state = pickle.load(file_obj)
    except Exception:
        return default_state()

    merged = default_state()
    merged.update(state if isinstance(state, dict) else {})
    return merged


def save_state(state):
    filepath = get_state_path()
    with open(filepath, "wb") as file_obj:
        pickle.dump(state, file_obj)


def format_indian_number(value):

    try:
        num = int(round(float(value)))
    except (TypeError, ValueError):
        num = 0

    sign = "-" if num < 0 else ""
    s = str(abs(num))

    if len(s) <= 3:
        return f"{sign}{s}"

    last_three = s[-3:]
    remaining = s[:-3]
    parts = []

    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]

    if remaining:
        parts.insert(0, remaining)

    return f"{sign}{','.join(parts + [last_three])}"


def format_revenue_display(value):
    return f"Rs {format_indian_number(value)}"
 
 
def format_display_date(value):
    if not value:
        return "-"

    try:
        return pd.to_datetime(value).strftime("%d %b %Y")
    except Exception:
        return str(value)


def build_dashboard_rows(df):
    base = df.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")

    group_cols = ["date", "product_name", "month"]
    if "category" in base.columns:
        group_cols.append("category")

    summary = (
        base.groupby(group_cols, as_index=False)
        .agg(
            quantity_sold=("quantity_sold", "sum"),
            predicted_sales=("predicted_sales", "sum"),
            revenue=("revenue", "sum")
        )
    )

    return summary.to_dict(orient="records")


def build_category_maps(df):
    cat_products = (
        df.groupby("category")["product_name"]
        .unique()
        .apply(sorted)
        .to_dict()
    )
    cat_products = {str(k): list(v) for k, v in cat_products.items()}

    cat_top_bottom = {}
    for cat, grp in df.groupby("category"):
        sales = grp.groupby("product_name")["quantity_sold"].sum()
        top = sales.sort_values(ascending=False).head(5)
        bottom = sales.sort_values(ascending=True).head(5)
        cat_top_bottom[str(cat)] = {
            "top": {str(k): int(v) for k, v in top.items()},
            "bottom": {str(k): int(v) for k, v in bottom.items()},
        }

    return cat_products, cat_top_bottom


def build_product_forecasts(df, days):

    base = df.copy()
    base["date"] = pd.to_datetime(base["date"])

    last_date = base["date"].max()
    forecasts = {}

    for product_name, group in base.groupby("product_name"):
        grp = group.sort_values("date")
        recent = grp["predicted_sales"].tail(min(7, len(grp))).astype(float)
        recent_actual = grp["quantity_sold"].tail(min(30, len(grp))).astype(float)

        base_level = float(recent.mean()) if len(recent) else 0.0
        slope = 0.0
        if len(recent) > 1:
            slope = float((recent.iloc[-1] - recent.iloc[0]) / (len(recent) - 1))

        weekday_factor = {i: 1.0 for i in range(7)}
        if len(recent_actual):
            tmp = grp.tail(min(90, len(grp))).copy()
            tmp["weekday"] = tmp["date"].dt.dayofweek
            weekday_avg = tmp.groupby("weekday")["quantity_sold"].mean()
            overall_avg = float(tmp["quantity_sold"].mean()) if float(tmp["quantity_sold"].mean()) > 0 else 1.0
            for wd, val in weekday_avg.items():
                weekday_factor[int(wd)] = max(float(val) / overall_avg, 0.6)

        items = []
        for i in range(1, days + 1):
            forecast_date = last_date + pd.Timedelta(days=i)
            wd = int(forecast_date.dayofweek)
            predicted = max((base_level + slope * i) * weekday_factor.get(wd, 1.0), 0)
            items.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "predicted_sales": round(predicted)
            })

        forecasts[str(product_name)] = items

    return forecasts


def build_alerts_summary(df, forecast_days=7, lead_time_days=3):
 
    alerts_df = df.copy()
    alerts_df["date"] = pd.to_datetime(alerts_df["date"])
    alerts_df["predicted_sales"] = alerts_df["predicted_sales"].clip(lower=0)
    analysis_date = alerts_df["date"].max()
    product_forecasts = build_product_forecasts(alerts_df, days=forecast_days)
 
    summary = (
        alerts_df.sort_values("date")
        .groupby("product_name", as_index=False)
        .agg(
            current_stock=("stock_available", "last"),
            avg_daily_sales=("quantity_sold", "mean"),
            predicted_daily_sales=("predicted_sales", "mean"),
            latest_price=("price", "last"),
            sales_std=("quantity_sold", "std")
        )
    )
 
    summary["current_stock"]          = summary["current_stock"].round().astype(int)
    summary["avg_daily_sales"]         = summary["avg_daily_sales"].round().astype(int)
    summary["predicted_daily_sales"]   = summary["predicted_daily_sales"].round().astype(int)
    summary["daily_demand"]            = summary[["avg_daily_sales", "predicted_daily_sales"]].max(axis=1).clip(lower=0.1)
 
    summary["days_of_stock_left"] = (
        summary["current_stock"] / summary["daily_demand"]
    ).round().astype(int)
 
    summary["predicted_sales_next_Ndays"] = summary["product_name"].map(
        lambda name: int(round(sum(item["predicted_sales"] for item in product_forecasts.get(str(name), []))))
    )
 
    summary["shortfall_units"] = (
        summary["predicted_sales_next_Ndays"] - summary["current_stock"]
    ).clip(lower=0)
 
    summary["volatility_value"] = (
        summary["sales_std"].fillna(0.0) /
        summary["daily_demand"].replace(0, 1.0)
    ).fillna(0.0).round(2)
 
    vol_low  = float(summary["volatility_value"].quantile(0.33)) if len(summary) else 0.0
    vol_high = float(summary["volatility_value"].quantile(0.66)) if len(summary) else 0.0
 
    statuses        = []
    recommendations = []
    reorder_dates   = []
    volatility_scores = []
 
    for _, row in summary.iterrows():
        dsl       = row["days_of_stock_left"]
        shortfall = row["shortfall_units"]
 
        if dsl < lead_time_days:
            statuses.append("Stockout Risk")
            recommendations.append("Place reorder immediately — stock critically low")
        elif dsl <= forecast_days and shortfall > 0:
            statuses.append("Stockout Risk")
            recommendations.append(f"Stock will fall short in {forecast_days}-day window — reorder soon")
        elif dsl > 30:
            statuses.append("Overstock")
            recommendations.append("Excess stock — consider discounting or pausing orders")
        else:
            statuses.append("Balanced")
            recommendations.append("Stock healthy — monitor and reorder on schedule")
 
        if row["volatility_value"] >= vol_high:
            volatility_scores.append("High")
        elif row["volatility_value"] >= vol_low:
            volatility_scores.append("Medium")
        else:
            volatility_scores.append("Low")
 
        reorder_offset = int(max(dsl - lead_time_days, 0))
        reorder_dates.append(
            (analysis_date + pd.Timedelta(days=reorder_offset)).strftime("%Y-%m-%d")
        )
 
    summary["stock_status"]     = statuses
    summary["recommendation"]   = recommendations
    summary["reorder_date"]     = reorder_dates
    summary["volatility_score"] = volatility_scores
    summary["lead_time_days"]   = int(lead_time_days)
    summary["analysis_date"]    = analysis_date.strftime("%Y-%m-%d")
 
    stockout_mask = summary["stock_status"] == "Stockout Risk"
 
    summary["revenue_at_risk"] = 0.0
    summary.loc[stockout_mask, "revenue_at_risk"] = (
        summary.loc[stockout_mask, "shortfall_units"] * summary.loc[stockout_mask, "latest_price"]
    ).round(2)
    summary.loc[~stockout_mask, "shortfall_units"] = 0.0
 
    return summary.sort_values(
        ["revenue_at_risk", "days_of_stock_left", "product_name"],
        ascending=[False, True, True]
    ).reset_index(drop=True)


@app.route("/", methods=["GET", "POST"])
def home():
    state = load_state()

    if request.method == "POST":
        file = request.files.get("file")

        try:
            days = int(request.form.get("days", 7))
        except (TypeError, ValueError):
            days = 7

        days = max(1, min(days, 60))

        if file is None or not file.filename:
            return render_template("index.html", error="Please upload a CSV file.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            processed_df, mapping, missing = preprocess_data(df)

            if missing:
                state["df"] = None
                state["result"] = None
                state["result_days"] = None
                state["days"] = days
                save_state(state)
                return render_template(
                    "index.html",
                    mapping=mapping,
                    missing=missing,
                    error="Required columns are missing. Please add them before analysis.",
                    days=days
                )

            state["df"] = processed_df
            state["result"] = None
            state["result_days"] = None
            state["days"] = days
            save_state(state)

            return redirect(url_for("analyze"))

        except Exception as e:
            state["df"] = None
            state["result"] = None
            state["result_days"] = None
            state["days"] = days
            save_state(state)
            return render_template("index.html", error=str(e), days=days)

    return render_template("index.html", days=7)


@app.route("/analyze")
def analyze():
    state = load_state()
    df = state.get("df")
    days = state.get("days", 7)

    if df is None:
        return redirect(url_for("home"))

    if state.get("result") is not None and state.get("result_days") == days:
        return redirect(url_for("dashboard"))

    try:
        result = run_full_pipeline(df, days=days)
    except Exception as e:
        return render_template("index.html", error=f"Analysis failed: {e}", days=days)

    state["result"] = result
    state["result_days"] = days
    save_state(state)

    return redirect(url_for("dashboard"))


@app.route("/reanalyze", methods=["POST"])
def reanalyze():
    state = load_state()

    try:
        days = int(request.form.get("days", 7))
    except (TypeError, ValueError):
        days = 7

    days = max(1, min(days, 60))
    state["days"] = days
    state["result"] = None
    state["result_days"] = None
    save_state(state)

    return redirect(url_for("analyze"))


@app.route("/dashboard")
def dashboard():
    state = load_state()
    result = state.get("result")

    if result is None:
        return redirect(url_for("home"))

    df = result["data"]
    future = result["future"]
    product_rows = build_dashboard_rows(df)
    total_product_rows = len(product_rows)
    product_forecasts = build_product_forecasts(df, days=state["days"])
    product_latest_price = (
        df.sort_values("date")
        .groupby("product_name", as_index=False)["price"]
        .last()
    )
    price_map = {str(r["product_name"]): float(r["price"]) for _, r in product_latest_price.iterrows()}
    top_products_map = result["top_products"].to_dict()
    bottom_products_map = result.get("bottom_products", pd.Series(dtype=float)).to_dict()

    future_display = future.copy()
    future_display["date"] = pd.to_datetime(future_display["date"]).dt.strftime("%Y-%m-%d")
    avg_price_all = float(df["price"].mean()) if len(df) else 0.0
    predicted_revenue_total = float(future_display["predicted_sales"].sum()) * avg_price_all
    avg_sales_day = float(df.groupby("date")["quantity_sold"].sum().mean()) if len(df) else 0.0

    cat_products, cat_top_bottom = {}, {}
    if "category" in df.columns:
        cat_products, cat_top_bottom = build_category_maps(df)

    # Train/test split info
    has_test_split  = result.get("has_test_split", False)
    train_cutoff    = result.get("train_cutoff", "")
    test_metrics    = result.get("test_metrics") or {}
    test_accuracy   = round(float(test_metrics.get("accuracy_percent", 0.0)), 2)
    test_mae        = round(float(test_metrics.get("mae", 0.0)), 2)

    return render_template(
        "dashboard.html",
        total_sales=int(df["quantity_sold"].sum()),
        total_revenue=int(df["revenue"].sum()),
        total_revenue_display=format_revenue_display(df["revenue"].sum()),
        predicted_revenue_display=format_revenue_display(predicted_revenue_total),
        avg_sales_day=int(round(avg_sales_day)),
        accuracy_percent=round(float(result["accuracy"].get("accuracy_percent", 0.0)), 2),
        forecast_days=state["days"],
        future_data=future_display.to_dict(orient="records"),
        future_by_product=product_forecasts,
        product_price_map=price_map,
        product_rows=product_rows,
        total_product_rows=total_product_rows,
        top_products=top_products_map,
        bottom_products=bottom_products_map,
        seasonal=result["seasonal"].to_dict(),
        model_name=result.get("best_model_name", "Unknown"),
        fit_diagnostics=result.get("fit_diagnostics", {}),
        category_products=cat_products,
        category_top_bottom=cat_top_bottom,
        has_test_split=has_test_split,
        train_cutoff=train_cutoff,
        test_accuracy=test_accuracy,
        test_mae=test_mae,
    )


@app.route("/alerts")
def alerts():
    state = load_state()
    result = state.get("result")

    if result is None:
        return redirect(url_for("home"))

    df = result["data"]
    forecast_days = state.get("days", 7)
    alerts_summary = build_alerts_summary(df, forecast_days=forecast_days)
    alerts_data = alerts_summary.to_dict(orient="records")
 
    understock_items = [x for x in alerts_data if x.get("stock_status") == "Stockout Risk"]
    overstock_items = [x for x in alerts_data if x.get("stock_status") == "Overstock"]
    balanced_items = [x for x in alerts_data if x.get("stock_status") == "Balanced"]
    priority_items = sorted(
        [x for x in alerts_data if float(x.get("revenue_at_risk", 0) or 0) > 0],
        key=lambda item: float(item.get("revenue_at_risk", 0) or 0),
        reverse=True
    )[:5]
 
    stockout_count = len(understock_items)
    overstock_count = len(overstock_items)
    balanced_count = len(balanced_items)
    total_revenue_at_risk = sum(float(x.get("revenue_at_risk", 0) or 0) for x in alerts_data)
    analysis_date = alerts_summary["analysis_date"].iloc[0] if not alerts_summary.empty else ""
    lead_time_days = int(alerts_summary["lead_time_days"].iloc[0]) if not alerts_summary.empty else 3
 
    return render_template(
        "alerts.html",
        alerts=alerts_data,
        understock_items=understock_items,
        overstock_items=overstock_items,
        balanced_items=balanced_items,
        priority_items=priority_items,
        stockout_count=stockout_count,
        overstock_count=overstock_count,
        balanced_count=balanced_count,
        total_revenue_at_risk_display=format_revenue_display(total_revenue_at_risk),
        analysis_date_display=format_display_date(analysis_date),
        lead_time_days=lead_time_days,
        forecast_days=forecast_days
    )


@app.route("/reports")
def reports():
    state = load_state()
    result = state.get("result")

    if result is None:
        return redirect(url_for("home"))

    df = result["data"]
    cleaned_df = state.get("df")
    customers = result.get("customers")
    customer_items = []

    if customers is not None:
        try:
            customer_items = [(str(customer), int(value)) for customer, value in customers.items()]
        except Exception:
            customer_items = []

    report_info = {
        "rows_processed": len(df),
        "forecast_days": state.get("days", 7),
        "products_covered": int(df["product_name"].nunique()) if "product_name" in df.columns else 0,
        "cleaned_rows": len(cleaned_df) if cleaned_df is not None else 0,
        "generated_on": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }

    return render_template(
        "reports.html",
        customer_items=customer_items,
        total_rows=len(df),
        report_info=report_info,
        model_name=result.get("best_model_name", "Unknown"),
        fit_diagnostics=result.get("fit_diagnostics", {}),
        training_metrics=result.get("training_metrics", {}),
        validation_metrics=result.get("accuracy", {}),
    )


@app.route("/download")
def download():
    state = load_state()
    result = state.get("result")

    if result is None:
        return redirect(url_for("home"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "final_output.csv")
    result["data"].to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True, download_name="final_output.csv")


@app.route("/download-cleaned")
def download_cleaned():
    state = load_state()
    df = state.get("df")

    if df is None:
        return redirect(url_for("home"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "cleaned_dataset.csv")
    df.to_csv(filepath, index=False)

    return send_file(filepath, as_attachment=True, download_name="cleaned_dataset.csv")

if __name__ == "__main__":
    app.run(debug=False)
