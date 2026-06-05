import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# FEATURE ENGINEERING
def feature_engineering(df):
    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["product_name", "date"]).reset_index(drop=True)
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["day_of_week_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["day_of_week_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    categorical_cols = [
        "category",
        "brand",
        "city",
        "state",
        "region",
        "customer_segment",
        "payment_method",
        "fulfillment_type",
        "return_requested",
        "device_type",
        "order_status",
    ]

    for col in categorical_cols:
        if col in data.columns:
            data[f"{col}_code"] = data[col].astype("category").cat.codes.astype(float)

    if "product_name" in data.columns:
        data["product_code"] = data["product_name"].astype("category").cat.codes.astype(float)

        grp = data.groupby("product_name", sort=False)
        lag_1 = grp["quantity_sold"].shift(1)
        lag_2 = grp["quantity_sold"].shift(2)
        lag_3 = grp["quantity_sold"].shift(3)
        lag_7 = grp["quantity_sold"].shift(7)
        rolling_3 = grp["quantity_sold"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        rolling_7 = grp["quantity_sold"].transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
        rolling_std_7 = grp["quantity_sold"].transform(lambda s: s.shift(1).rolling(window=7, min_periods=2).std())
        expanding_mean = grp["quantity_sold"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        product_avg_sales = grp["quantity_sold"].transform(lambda s: s.expanding(min_periods=1).mean()).shift(1)

        global_mean = float(data["quantity_sold"].mean()) if len(data) else 0.0
        data["lag_1"] = lag_1.fillna(global_mean)
        data["lag_2"] = lag_2.fillna(data["lag_1"])
        data["lag_3"] = lag_3.fillna(data["lag_2"])
        data["lag_7"] = lag_7.fillna(data["lag_3"])
        data["rolling_mean_3"] = rolling_3.fillna(data["lag_1"])
        data["rolling_mean_7"] = rolling_7.fillna(data["rolling_mean_3"])
        data["rolling_std_7"] = rolling_std_7.fillna(0.0)
        data["expanding_mean"] = expanding_mean.fillna(global_mean)
        data["product_avg_sales"] = product_avg_sales.fillna(global_mean)

    return data


def get_feature_columns(df):
    cols = [
        "price",
        "stock_available",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_std_7",
        "expanding_mean",
        "product_avg_sales",
    ]

    optional_numeric_cols = [
        "discount",
        "rating",
        "shipping_fee",
        "product_code",
    ]

    for col in optional_numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)

    if "product_id" in df.columns and pd.api.types.is_numeric_dtype(df["product_id"]):
        cols.append("product_id")

    categorical_code_cols = [
        "category_code",
        "brand_code",
        "city_code",
        "state_code",
        "region_code",
        "customer_segment_code",
        "payment_method_code",
        "fulfillment_type_code",
        "return_requested_code",
        "device_type_code",
        "order_status_code",
    ]

    for col in categorical_code_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)

    return cols


def calculate_accuracy_percent(y_true, y_pred):
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    denom = np.maximum(np.abs(y_true_arr), 1.0)
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / denom))
    acc = max(0.0, 100.0 - (mape * 100.0))
    return round(acc, 2)


def summarize_metrics(y_true, y_pred):
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    acc_percent = calculate_accuracy_percent(y_true_arr, y_pred_arr)

    if len(y_true_arr) > 1:
        try:
            r2 = r2_score(y_true_arr, y_pred_arr)
        except Exception:
            r2 = 0.0
    else:
        r2 = 0.0

    return {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "accuracy_percent": round(float(acc_percent), 2),
    }


def diagnose_fit(training_metrics, validation_metrics, limited_data=False):
    if limited_data or validation_metrics is None:
        return {
            "status": "Limited Data",
            "message": "Dataset is too small for a reliable holdout check, so overfitting/underfitting cannot be judged confidently.",
        }

    train_acc = float(training_metrics.get("accuracy_percent", 0.0))
    val_acc = float(validation_metrics.get("accuracy_percent", 0.0))
    train_r2 = float(training_metrics.get("r2_score", 0.0))
    val_r2 = float(validation_metrics.get("r2_score", 0.0))
    acc_gap = train_acc - val_acc
    r2_gap = train_r2 - val_r2

    if train_r2 >= 0.7 and acc_gap >= 12 and r2_gap >= 0.35:
        return {
            "status": "Possible Overfitting",
            "message": "Training performance is noticeably better than validation performance.",
        }

    if train_acc < 60 and val_acc < 60 and train_r2 < 0.25 and val_r2 < 0.25:
        return {
            "status": "Possible Underfitting",
            "message": "Both training and validation accuracy are weak, so the model may be too simple for this dataset.",
        }

    return {
        "status": "Healthy",
        "message": "Training and validation performance are reasonably aligned.",
    }


def average_metric_dict(metric_dicts):
    if not metric_dicts:
        return {"r2_score": 0.0, "mae": 0.0, "accuracy_percent": 0.0}

    return {
        "r2_score": round(float(np.mean([m["r2_score"] for m in metric_dicts])), 4),
        "mae": round(float(np.mean([m["mae"] for m in metric_dicts])), 2),
        "accuracy_percent": round(float(np.mean([m["accuracy_percent"] for m in metric_dicts])), 2),
    }


def model_selection_score(training_metrics, validation_metrics):
    train_acc = float(training_metrics.get("accuracy_percent", 0.0))
    val_acc = float(validation_metrics.get("accuracy_percent", 0.0))
    train_r2 = float(training_metrics.get("r2_score", 0.0))
    val_r2 = float(validation_metrics.get("r2_score", 0.0))
    val_mae = float(validation_metrics.get("mae", 0.0))

    gap_penalty = max(train_acc - val_acc, 0.0) * 0.7
    r2_gap_penalty = max(train_r2 - val_r2, 0.0) * 15.0
    negative_r2_penalty = abs(min(val_r2, 0.0)) * 25.0
    mae_penalty = val_mae * 5.0

    return val_acc + (max(val_r2, 0.0) * 20.0) - gap_penalty - r2_gap_penalty - negative_r2_penalty - mae_penalty


def build_time_splits(n_rows):
    if n_rows >= 18:
        splitter = TimeSeriesSplit(n_splits=2)
        return list(splitter.split(np.arange(n_rows)))

    val_size = max(1, int(np.ceil(n_rows * 0.2)))
    train_end = n_rows - val_size
    return [(np.arange(train_end), np.arange(train_end, n_rows))]


def get_candidate_models():
    return {
        "Ridge": Ridge(alpha=4.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=60,
            random_state=42,
            n_jobs=1,
            max_depth=12,
            min_samples_leaf=2,
            min_samples_split=4,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=80,
            random_state=42,
            n_jobs=1,
            max_depth=9,
            min_samples_leaf=3,
            min_samples_split=6,
        ),
    }


def fit_selected_model(df, model_name):
    data = feature_engineering(df).sort_values("date").reset_index(drop=True)
    feature_cols = get_feature_columns(data)
    X = data[feature_cols]
    y = data["quantity_sold"]

    if len(data) < 5:
        model = DummyRegressor(strategy="mean")
    else:
        model = clone(get_candidate_models()[model_name])

    model.fit(X, y)

    return {
        "best_name": model_name,
        "best_model": model,
        "feature_cols": feature_cols,
    }

# TRAIN MULTI MODEL (VALIDATED)
def train_model(df):
    data = feature_engineering(df).sort_values("date").reset_index(drop=True)
    feature_cols = get_feature_columns(data)
    X = data[feature_cols]
    y = data["quantity_sold"]
    if len(data) < 5:
        fallback_model = DummyRegressor(strategy="mean")
        fallback_model.fit(X, y)
        train_pred = fallback_model.predict(X)
        training_metrics = summarize_metrics(y, train_pred)
        fit_diagnostics = diagnose_fit(training_metrics, None, limited_data=True)
        return {
            "best_name": "MeanBaseline",
            "best_model": fallback_model,
            "feature_cols": feature_cols,
            "training": training_metrics,
            "validation": training_metrics,
            "fit_diagnostics": fit_diagnostics,
        }
    val_size = max(1, int(np.ceil(len(data) * 0.2)))
    X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]
    candidates = get_candidate_models()
    splits = build_time_splits(len(data))
    scored = []
    for name, model in candidates.items():
        fold_train_metrics = []
        fold_val_metrics = []
        for train_idx, val_idx in splits:
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            pred_train = fold_model.predict(X_fold_train)
            pred_val = fold_model.predict(X_fold_val)
            fold_train_metrics.append(summarize_metrics(y_fold_train, pred_train))
            fold_val_metrics.append(summarize_metrics(y_fold_val, pred_val))
        train_metrics = average_metric_dict(fold_train_metrics)
        val_metrics = average_metric_dict(fold_val_metrics)
        selection_score = model_selection_score(train_metrics, val_metrics)
        scored.append(
            {
                "name": name,
                "model": model,
                "training": train_metrics,
                "validation": val_metrics,
                "selection_score": round(float(selection_score), 4),
            }
        )
    scored.sort(
        key=lambda x: (
            x["selection_score"],
            x["validation"]["accuracy_percent"],
        ),
        reverse=True,
    )
    best = scored[0]

    # Refit best model on full data for final prediction/forecast.
    best_full = clone(candidates[best["name"]])
    best_full.fit(X, y)
    fit_diagnostics = diagnose_fit(best["training"], best["validation"])

    return {
        "best_name": best["name"],
        "best_model": best_full,
        "feature_cols": feature_cols,
        "training": best["training"],
        "validation": best["validation"],
        "fit_diagnostics": fit_diagnostics,
    }

# MODEL PREDICTION
def predict_demand(df, model_bundle):
    data = feature_engineering(df)

    X = data[model_bundle["feature_cols"]]
    preds = model_bundle["best_model"].predict(X)

    data["predicted_sales"] = np.clip(preds, a_min=0, a_max=None)

    return data, model_bundle["best_name"], model_bundle["best_model"]

# FUTURE FORECAST
def future_forecast(df, model_bundle, days):
    base = feature_engineering(df)
    last_date = pd.to_datetime(base["date"]).max()
    avg_price = float(base["price"].mean()) if len(base) else 0.0
    avg_stock = float(base["stock_available"].mean()) if "stock_available" in base.columns and len(base) else 0.0

    forecast_rows = []
    history_by_product = {}

    for product_name, group in base.groupby("product_name", sort=False):
        grp = group.sort_values("date").copy()
        info = {
            "sales": list(grp["quantity_sold"].astype(float)),
            "product_id": float(grp["product_id"].mode().iloc[0]) if "product_id" in grp.columns and len(grp["product_id"].dropna()) else 0.0,
            "product_code": float(grp["product_code"].iloc[-1]) if "product_code" in grp.columns else 0.0,
            "price": float(grp["price"].mean()) if "price" in grp.columns else avg_price,
            "stock_available": float(grp["stock_available"].mean()) if "stock_available" in grp.columns else avg_stock,
            "discount": float(grp["discount"].mean()) if "discount" in grp.columns else 0.0,
            "rating": float(grp["rating"].mean()) if "rating" in grp.columns else 0.0,
            "shipping_fee": float(grp["shipping_fee"].mean()) if "shipping_fee" in grp.columns else 0.0,
        }

        for col in [
            "category_code",
            "brand_code",
            "city_code",
            "state_code",
            "region_code",
            "customer_segment_code",
            "payment_method_code",
            "fulfillment_type_code",
            "return_requested_code",
            "device_type_code",
            "order_status_code",
        ]:
            info[col] = float(grp[col].mode().iloc[0]) if col in grp.columns and len(grp[col].dropna()) else 0.0

        history_by_product[product_name] = info

    for step in range(1, days + 1):
        forecast_date = last_date + pd.Timedelta(days=step)

        for product_name, info in history_by_product.items():
            sales_hist = info["sales"]
            lag_1 = sales_hist[-1] if len(sales_hist) >= 1 else 0.0
            lag_2 = sales_hist[-2] if len(sales_hist) >= 2 else lag_1
            lag_3 = sales_hist[-3] if len(sales_hist) >= 3 else lag_2
            lag_7 = sales_hist[-7] if len(sales_hist) >= 7 else lag_3
            rolling_3 = float(np.mean(sales_hist[-3:])) if sales_hist else 0.0
            rolling_7 = float(np.mean(sales_hist[-7:])) if sales_hist else rolling_3
            rolling_std_7 = float(np.std(sales_hist[-7:])) if len(sales_hist) >= 2 else 0.0
            expanding_mean = float(np.mean(sales_hist)) if sales_hist else 0.0

            row = {
                "date": forecast_date,
                "price": info["price"] if info["price"] else avg_price,
                "stock_available": info["stock_available"] if info["stock_available"] else avg_stock,
                "month": forecast_date.month,
                "day": forecast_date.day,
                "day_of_week": forecast_date.dayofweek,
                "day_of_year": forecast_date.dayofyear,
                "week_of_year": int(forecast_date.isocalendar().week),
                "is_weekend": int(forecast_date.dayofweek >= 5),
                "day_of_week_sin": float(np.sin(2 * np.pi * forecast_date.dayofweek / 7.0)),
                "day_of_week_cos": float(np.cos(2 * np.pi * forecast_date.dayofweek / 7.0)),
                "month_sin": float(np.sin(2 * np.pi * forecast_date.month / 12.0)),
                "month_cos": float(np.cos(2 * np.pi * forecast_date.month / 12.0)),
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_7": lag_7,
                "rolling_mean_3": rolling_3,
                "rolling_mean_7": rolling_7,
                "rolling_std_7": rolling_std_7,
                "expanding_mean": expanding_mean,
                "product_avg_sales": expanding_mean,
                "discount": info["discount"],
                "rating": info["rating"],
                "shipping_fee": info["shipping_fee"],
                "product_code": info["product_code"],
                "product_id": info["product_id"],
            }

            for col in [
                "category_code",
                "brand_code",
                "city_code",
                "state_code",
                "region_code",
                "customer_segment_code",
                "payment_method_code",
                "fulfillment_type_code",
                "return_requested_code",
                "device_type_code",
                "order_status_code",
            ]:
                row[col] = info.get(col, 0.0)

            X_future = pd.DataFrame([row])[model_bundle["feature_cols"]]
            predicted = float(np.clip(model_bundle["best_model"].predict(X_future)[0], a_min=0, a_max=None))
            info["sales"].append(predicted)
            forecast_rows.append({
                "date": forecast_date,
                "product_name": product_name,
                "predicted_sales": predicted,
            })

    future_df = pd.DataFrame(forecast_rows)
    if future_df.empty:
        return pd.DataFrame(columns=["date", "predicted_sales"])

    summary = (
        future_df.groupby("date", as_index=False)["predicted_sales"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return summary[["date", "predicted_sales"]]

# REVENUE CALCULATION
def calculate_revenue(df):
    data = df.copy()
    data["revenue"] = data["price"] * data["predicted_sales"]
    return data

# STOCK ANALYSIS
def stock_analysis(df):
    data = df.copy()

    conditions = []

    for _, row in data.iterrows():
        if row["predicted_sales"] > row["stock_available"]:
            conditions.append("Stockout Risk")
        elif row["stock_available"] > row["predicted_sales"] * 2:
            conditions.append("Overstock")
        else:
            conditions.append("Balanced")

    data["stock_status"] = conditions

    return data

# RECOMMENDATION SYSTEM
def recommendation(df):
    data = df.copy()

    actions = []

    for _, row in data.iterrows():
        if "Stockout" in row["stock_status"]:
            actions.append("Reorder Product")
        elif "Overstock" in row["stock_status"]:
            actions.append("Apply Discount")
        else:
            actions.append("No Action Needed")

    data["recommendation"] = actions

    return data

# ACCURACY METRICS
def accuracy_metrics(df, validation_metrics=None):
    if validation_metrics is not None:
        return validation_metrics

    y_true = df["quantity_sold"]
    y_pred = df["predicted_sales"]

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(np.maximum(np.abs(y_true), 1), y_pred)

    return {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "accuracy_percent": round(float(max(0.0, 100 - (mape * 100))), 2),
    }

# TOP/BOTTOM PRODUCTS
def top_products(df):
    return (
        df.groupby("product_name")["quantity_sold"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )


def bottom_products(df):
    return (
        df.groupby("product_name")["quantity_sold"]
        .sum()
        .sort_values(ascending=True)
        .head(5)
    )

# SEASONAL ANALYSIS
def seasonal_demand(df):
    return df.groupby("month")["quantity_sold"].sum()

# CUSTOMER BEHAVIOR
def customer_behavior(df):
    if "customer_id" in df.columns:
        return df["customer_id"].value_counts().head(5)
    return None

# FULL PIPELINE
def run_full_pipeline(df, days=7):
    df_sorted = df.copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date").reset_index(drop=True)

    all_dates = sorted(df_sorted["date"].unique())
    n_dates = len(all_dates)

    if n_dates >= 5:
        # Last 20% dates = test set (genuine holdout)
        split_idx = max(1, int(np.ceil(n_dates * 0.8)))
        train_cutoff = all_dates[split_idx - 1]  
        test_cutoff  = all_dates[split_idx]      

        df_train = df_sorted[df_sorted["date"] <= train_cutoff].reset_index(drop=True)
        df_test  = df_sorted[df_sorted["date"] >  train_cutoff].reset_index(drop=True)
    else:
        df_train = df_sorted.copy()
        df_test  = pd.DataFrame()
        train_cutoff = df_sorted["date"].max()

    model_bundle = train_model(df_train)

    train_data, best_name, best_model = predict_demand(df_train, model_bundle)
    train_data = calculate_revenue(train_data)
    train_data = stock_analysis(train_data)
    train_data = recommendation(train_data)

    test_metrics = None
    test_data    = pd.DataFrame()

    if not df_test.empty:
        test_data, _, _ = predict_demand(df_test, model_bundle)
        test_data = calculate_revenue(test_data)
        test_data = stock_analysis(test_data)
        test_data = recommendation(test_data)

        test_metrics = summarize_metrics(
            test_data["quantity_sold"].values,
            test_data["predicted_sales"].values
        )

    if not test_data.empty:
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        combined_data = combined_data.sort_values("date").reset_index(drop=True)
    else:
        combined_data = train_data.copy()

    full_bundle = fit_selected_model(df_sorted, model_bundle["best_name"])
    future = future_forecast(df_sorted, full_bundle, days)

    final_metrics = test_metrics if test_metrics else model_bundle["validation"]

    top      = top_products(combined_data)
    bottom   = bottom_products(combined_data)
    seasonal = seasonal_demand(combined_data)
    customers = customer_behavior(combined_data)

    return {
        "data":             combined_data,
        "train_data":       train_data,
        "test_data":        test_data,
        "future":           future,
        "best_model":       best_model,
        "best_model_name":  best_name,
        "accuracy":         final_metrics,
        "test_metrics":     test_metrics,
        "training_metrics": model_bundle["training"],
        "fit_diagnostics":  model_bundle["fit_diagnostics"],
        "top_products":     top,
        "bottom_products":  bottom,
        "seasonal":         seasonal,
        "customers":        customers,
        "train_cutoff":     str(train_cutoff.date()),
        "has_test_split":   not df_test.empty,
    }
