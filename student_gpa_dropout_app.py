from pathlib import Path
import json
import os
import pickle

from flask import Flask, request, render_template_string
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_dropout_dataset_v3.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

CLASS_MODEL_PATH = ARTIFACTS_DIR / "dropout_classifier.pkl"
REG_MODEL_PATH = ARTIFACTS_DIR / "numeric_gpa_regressor.pkl"
RESULTS_PATH = ARTIFACTS_DIR / "gpa_dropout_results.json"

CLASS_LABELS = {0: "Dropout: No", 1: "Dropout: Yes"}


def normalize_dropout(value):
    text = str(value).strip().lower()
    if text in {"1", "yes", "y", "true", "dropout", "dropped", "dropped out"}:
        return 1
    if text in {"0", "no", "n", "false", "not dropout", "not_dropped", "active", "continue", "continued"}:
        return 0
    return int(float(text) >= 0.5)


def find_best_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_f1 = -1
    for threshold in [i / 100 for i in range(10, 91)]:
        predicted = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predicted, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold, best_f1


def load_and_prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    required_columns = ["GPA", "Dropout"]
    missing_required = [col for col in required_columns if col not in raw_df.columns]
    if missing_required:
        raise KeyError(f"Dataset is missing required columns: {missing_required}")

    df = raw_df.dropna(subset=["GPA", "Dropout"]).copy()
    df["dropout_binary"] = df["Dropout"].apply(normalize_dropout).astype(int)

    drop_columns = [
        col
        for col in [
            "Student_ID",
            "student_id",
            "ID",
            "id",
            "GPA",
            "CGPA",
            "Semester_GPA",
            "Dropout",
            "dropout",
            "student_status",
            "Status",
            "status",
            "cgpa_binary",
            "dropout_binary",
        ]
        if col in df.columns
    ]

    X = df.drop(columns=drop_columns)
    y_class = df["dropout_binary"]
    y_reg = df["GPA"]
    return raw_df, df, X, y_class, y_reg, drop_columns


def build_preprocessor(X):
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [col for col in X.columns if col not in categorical_columns]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def train_models():
    raw_df, df, X, y_class, y_reg, drop_columns = load_and_prepare_data()
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X)

    classification_models = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        classification_models["xgboost"] = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=600,
                        max_depth=4,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, model in classification_models.items():
        fold_metrics = []
        fold_thresholds = []
        for train_idx, val_idx in cv.split(X, y_class):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y_class.iloc[train_idx], y_class.iloc[val_idx]
            model.fit(X_fold_train, y_fold_train)
            probabilities = model.predict_proba(X_fold_val)[:, 1]
            threshold, _ = find_best_threshold(y_fold_val, probabilities)
            predictions = (probabilities >= threshold).astype(int)
            fold_thresholds.append(threshold)
            fold_metrics.append(
                {
                    "accuracy": accuracy_score(y_fold_val, predictions),
                    "precision": precision_score(y_fold_val, predictions, zero_division=0),
                    "recall": recall_score(y_fold_val, predictions, zero_division=0),
                    "f1": f1_score(y_fold_val, predictions, zero_division=0),
                }
            )

        metrics_df = pd.DataFrame(fold_metrics)
        cv_results[name] = {
            "accuracy": round(float(metrics_df["accuracy"].mean()), 4),
            "precision": round(float(metrics_df["precision"].mean()), 4),
            "recall": round(float(metrics_df["recall"].mean()), 4),
            "f1": round(float(metrics_df["f1"].mean()), 4),
            "avg_threshold": round(float(pd.Series(fold_thresholds).mean()), 4),
        }

    comparison_df = pd.DataFrame(cv_results).T.sort_values("f1", ascending=False)
    best_classifier_name = comparison_df.index[0]
    best_classifier = classification_models[best_classifier_name]
    best_threshold = float(comparison_df.loc[best_classifier_name, "avg_threshold"])

    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.2, stratify=y_class, random_state=RANDOM_STATE
    )

    best_classifier.fit(X_train, y_train_class)
    y_pred_proba = best_classifier.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_proba >= best_threshold).astype(int)

    test_classification_metrics = {
        "threshold": round(best_threshold, 4),
        "accuracy": round(float(accuracy_score(y_test_class, y_pred_class)), 4),
        "precision": round(float(precision_score(y_test_class, y_pred_class, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_class, y_pred_class, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test_class, y_pred_class, zero_division=0)), 4),
    }

    regression_models = {
        "random_forest_regressor": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBRegressor is not None:
        regression_models["xgboost_regressor"] = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    regression_results = {}
    trained_regressors = {}
    for name, model in regression_models.items():
        model.fit(X_train, y_train_reg)
        pred = model.predict(X_test)
        regression_results[name] = {
            "mae": round(float(mean_absolute_error(y_test_reg, pred)), 4),
            "r2": round(float(r2_score(y_test_reg, pred)), 4),
        }
        trained_regressors[name] = model

    regression_df = pd.DataFrame(regression_results).T.sort_values("mae")
    best_regressor_name = regression_df.index[0]
    best_regressor = trained_regressors[best_regressor_name]

    results_payload = {
        "dataset": "Student Dropout Prediction Dataset",
        "data_file": str(DATA_PATH),
        "rows": int(len(df)),
        "target": "Dropout binary classification and numeric GPA regression",
        "classes": CLASS_LABELS,
        "dropped_columns": drop_columns,
        "feature_columns": list(X.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "classification_cross_validation": cv_results,
        "best_classifier": best_classifier_name,
        "test_classification_metrics": {
            **test_classification_metrics,
            "confusion_matrix": confusion_matrix(y_test_class, y_pred_class, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_test_class,
                y_pred_class,
                target_names=[CLASS_LABELS[0], CLASS_LABELS[1]],
                zero_division=0,
                output_dict=True,
            ),
        },
        "regression_results": regression_results,
        "best_regressor": best_regressor_name,
    }

    with CLASS_MODEL_PATH.open("wb") as f:
        pickle.dump(best_classifier, f)
    with REG_MODEL_PATH.open("wb") as f:
        pickle.dump(best_regressor, f)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    return {
        "raw_df": raw_df,
        "df": df,
        "X": X,
        "best_classifier": best_classifier,
        "best_regressor": best_regressor,
        "best_threshold": best_threshold,
        "results": results_payload,
    }


def load_or_train_models():
    raw_df, df, X, y_class, y_reg, drop_columns = load_and_prepare_data()
    if CLASS_MODEL_PATH.exists() and REG_MODEL_PATH.exists() and RESULTS_PATH.exists():
        with CLASS_MODEL_PATH.open("rb") as f:
            best_classifier = pickle.load(f)
        with REG_MODEL_PATH.open("rb") as f:
            best_regressor = pickle.load(f)
        with RESULTS_PATH.open("r", encoding="utf-8") as f:
            results = json.load(f)
        best_threshold = float(results["test_classification_metrics"]["threshold"])
        return {
            "raw_df": raw_df,
            "df": df,
            "X": X,
            "best_classifier": best_classifier,
            "best_regressor": best_regressor,
            "best_threshold": best_threshold,
            "results": results,
        }

    return train_models()


STATE = load_or_train_models()
X = STATE["X"]
best_classifier = STATE["best_classifier"]
best_regressor = STATE["best_regressor"]
best_threshold = STATE["best_threshold"]
RESULTS = STATE["results"]


def build_ui_fields():
    fields = []
    training_limits = {}

    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            non_null = X[column].dropna()
            training_min = float(non_null.min())
            training_max = float(non_null.max())
            is_integer_field = bool((non_null % 1 == 0).all())
            display_name = column
            display_min = int(training_min) if is_integer_field else float(training_min)
            display_max = int(training_max) if is_integer_field else float(training_max)
            step = 1 if is_integer_field else 0.01
            help_text = f"Allowed range: {display_min} to {display_max}"

            if column == "Semester":
                display_name = "Year"
                help_text = f"Enter academic year, from {display_min} to {display_max}."
            elif column == "Age":
                display_min = 17
                display_max = 30
                step = 1
                is_integer_field = True
                help_text = "Enter age in years, from 17 to 30."
            elif column == "Family_Income":
                display_min = 0
                display_max = int(training_max)
                step = 1
                is_integer_field = True
                help_text = (
                    f"Annual family income in USD. Enter ${display_min} to ${display_max}. "
                    f"Values below ${int(training_min)} are treated as ${int(training_min)} for prediction."
                )
            elif column == "Study_Hours_per_Day":
                display_name = "Study_Minutes_per_Day"
                display_min = int(training_min * 60)
                display_max = int(training_max * 60)
                step = 1
                is_integer_field = True
                help_text = f"Enter study time in minutes per day, from {display_min} to {display_max}."

            training_limits[column] = {"min": training_min, "max": training_max}
            fields.append(
                {
                    "name": column,
                    "label": display_name,
                    "type": "number",
                    "min": display_min,
                    "max": display_max,
                    "step": step,
                    "integer": is_integer_field,
                    "help": help_text,
                }
            )
        else:
            values = sorted(X[column].dropna().astype(str).unique().tolist())
            label = "Year" if column == "Semester" else column
            fields.append({"name": column, "label": label, "type": "select", "options": values})

    return fields, training_limits


UI_FIELDS, TRAINING_LIMITS = build_ui_fields()


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPA and Dropout Predictor</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #111316; color: #251912; font-family: "Trebuchet MS", Verdana, sans-serif; }
    .shell { min-height: 100vh; display: flex; }
    .app { width: 100vw; min-height: 100vh; background: #fbfaf7; display: grid; grid-template-columns: 220px 1fr; }
    .sidebar { background: #f3f2ee; border-right: 1px solid #ded9d2; padding: 22px 16px; }
    .brand { font-weight: 900; letter-spacing: .08em; margin: 0 0 26px; font-size: 16px; }
    .nav-item, .section-label { height: 34px; display: flex; align-items: center; gap: 10px; padding: 0 10px; color: #3a302a; font-size: 13px; }
    .nav-item.active { background: #e8e6e0; }
    .dot { width: 14px; height: 14px; border: 1px solid #3a302a; border-radius: 50%; display: inline-block; }
    .section-label { margin-top: 18px; color: #8a8178; text-transform: uppercase; font-size: 11px; letter-spacing: .08em; }
    .topbar { height: 52px; border-bottom: 1px solid #ece7df; display: flex; align-items: center; padding: 0 22px; color: #8a8178; font-size: 13px; }
    .content { padding: 26px 32px 28px; }
    h1 { margin: 0 0 16px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(34px, 4vw, 50px); font-weight: 400; }
    h1 em { font-style: italic; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 16px; margin-bottom: 18px; }
    .metric { background: #f4f3ef; min-height: 96px; padding: 18px 22px; border-bottom: 6px solid var(--accent); }
    .metric span { display: block; color: #9a9188; font-size: 13px; margin-bottom: 12px; }
    .metric strong { font-size: 30px; font-weight: 500; }
    .workspace { display: grid; grid-template-columns: minmax(760px, 1fr) minmax(300px, 360px); gap: 20px; align-items: start; }
    form, .panel, .name-card { background: #f4f3ef; padding: 18px; border: 1px solid #eee8df; }
    .form-title, .panel h2, .name-card h2 { margin: 0 0 14px; font-family: Georgia, "Times New Roman", serif; font-size: 24px; font-weight: 400; }
    .panel { position: sticky; top: 18px; }
    .name-card { max-width: 520px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 11px 13px; }
    label { display: flex; flex-direction: column; gap: 5px; font-size: 13px; color: #4a4039; }
    .range { color: #928a82; font-size: 11px; }
    input, select { width: 100%; border: 1px solid #d9d2c8; background: #fffdfa; color: #251912; padding: 7px 9px; font-size: 14px; outline: none; }
    .submit { margin-top: 18px; border: 0; background: #2a130b; color: white; padding: 11px 18px; border-radius: 999px; font-weight: 800; cursor: pointer; }
    .prediction { font-family: Georgia, "Times New Roman", serif; font-size: 32px; line-height: 1.05; margin: 8px 0 18px; }
    .result-line { display: flex; justify-content: space-between; border-top: 1px solid #ddd6cd; padding: 12px 0; font-size: 14px; gap: 16px; }
    .error { color: #b91c1c; font-weight: 700; margin: 0; }
    @media (max-width: 1250px) { .workspace { grid-template-columns: 1fr; } .grid { grid-template-columns: repeat(3, minmax(150px, 1fr)); } .panel { position: static; } }
    @media (max-width: 980px) { .grid { grid-template-columns: repeat(2, minmax(150px, 1fr)); } }
    @media (max-width: 760px) { .app { grid-template-columns: 1fr; } .sidebar { display: none; } .content { padding: 20px 14px; } .metrics, .workspace, .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <div class="app">
      <aside class="sidebar">
        <p class="brand">GPAWISE</p>
        <div class="nav-item active"><span class="dot"></span> Predictor</div>
        <div class="nav-item"><span class="dot"></span> Metrics</div>
        <div class="nav-item"><span class="dot"></span> Dataset</div>
        <div class="section-label">Model</div>
        <div class="nav-item"><span class="dot"></span> Dropout class</div>
        <div class="nav-item"><span class="dot"></span> Numeric GPA</div>
      </aside>
      <section>
        <div class="topbar"><span>Student performance workspace</span></div>
        <div class="content">
          <h1>Hello, <em>{{ student_name if student_name else "student" }}</em></h1>
          {% if not student_name %}
          <form class="name-card" method="post">
            <h2>Before we begin</h2>
            <label>
              <span>Student name</span>
              <input type="text" name="student_name" value="" required autofocus>
            </label>
            <button class="submit" type="submit">Continue</button>
          </form>
          {% else %}
          <div class="metrics">
            <div class="metric" style="--accent:#2a130b"><span>Features</span><strong>{{ fields|length }}</strong></div>
            <div class="metric" style="--accent:#c9ff00"><span>Classes</span><strong>2</strong></div>
            <div class="metric" style="--accent:#68b7cf"><span>GPA scale</span><strong>4.0</strong></div>
            <div class="metric" style="--accent:#ff6a2a"><span>Status</span><strong>Live</strong></div>
          </div>
          <div class="workspace">
            <form method="post">
              <input type="hidden" name="student_name" value="{{ student_name }}">
              <p class="form-title">Prediction inputs</p>
              <div class="grid">
                {% for field in fields %}
                <label>
                  <span>{{ field.label }}</span>
                  {% if field.type == 'select' %}
                  <select name="{{ field.name }}" required>
                    <option value="" disabled hidden {% if not values.get(field.name) %}selected{% endif %}></option>
                    {% for option in field.options %}
                    <option value="{{ option }}" {% if values.get(field.name) == option|string %}selected{% endif %}>{{ option }}</option>
                    {% endfor %}
                  </select>
                  {% else %}
                  <input type="number" step="{{ field.step }}" min="{{ field.min }}" max="{{ field.max }}" name="{{ field.name }}" value="{{ values.get(field.name, '') }}" required>
                  <span class="range">{{ field.help }}</span>
                  {% endif %}
                </label>
                {% endfor %}
              </div>
              <button class="submit" type="submit">Predict</button>
            </form>
            <div class="panel">
              <h2>Prediction result</h2>
              {% if error %}
              <p class="error">{{ error }}</p>
              {% elif prediction %}
              <p class="prediction">{{ prediction }}</p>
              <div class="result-line"><span>Probability Dropout: Yes</span><strong>{{ probability_percent }}%</strong></div>
              <div class="result-line"><span>Predicted numeric GPA</span><strong>{{ numeric_gpa }} / 4.0</strong></div>
              <div class="result-line"><span>Burnout</span><strong>{{ burnout }}</strong></div>
              {% else %}
              <p class="prediction">Awaiting input</p>
              <div class="result-line"><span>Every field required</span><strong>Yes</strong></div>
              <div class="result-line"><span>Range validation</span><strong>On</strong></div>
              {% endif %}
            </div>
          </div>
          {% endif %}
        </div>
      </section>
    </div>
  </div>
</body>
</html>
"""

app = Flask(__name__)


def cast_form_value(column, value):
    if pd.api.types.is_numeric_dtype(X[column]):
        numeric_value = float(value)
        matching_field = next(field for field in UI_FIELDS if field["name"] == column)
        if matching_field.get("integer") and numeric_value % 1 != 0:
            return numeric_value
        if column == "Study_Hours_per_Day":
            numeric_value = numeric_value / 60
        if column == "Family_Income":
            numeric_value = max(numeric_value, TRAINING_LIMITS[column]["min"])
        if matching_field.get("integer") and column != "Study_Hours_per_Day":
            return int(numeric_value)
        return numeric_value
    return value


def validate_display_values(values):
    errors = []
    for field in UI_FIELDS:
        if field["type"] != "number":
            continue
        raw_value = values[field["name"]]
        value = float(raw_value)
        if field.get("integer") and value % 1 != 0:
            errors.append(f"{field['label']} must be a whole number.")
        if value < field["min"] or value > field["max"]:
            errors.append(f"{field['label']} must be between {field['min']} and {field['max']}.")
    return errors


@app.route("/", methods=["GET", "POST"])
def gpa_dropout_form():
    student_name = request.form.get("student_name", "").strip()
    values = {}
    prediction = None
    probability_percent = None
    numeric_gpa = None
    burnout = None
    error = None

    if request.method == "POST" and student_name:
        values = {field["name"]: request.form.get(field["name"], "") for field in UI_FIELDS}
        missing = [field["label"] for field in UI_FIELDS if not values[field["name"]]]
        if missing:
            error = "Please complete every field before predicting."
        else:
            row = {column: cast_form_value(column, values[column]) for column in X.columns}
            range_errors = validate_display_values(values)
            if range_errors:
                error = " ".join(range_errors)
            else:
                input_df = pd.DataFrame([row])[X.columns]
                prob_dropout_yes = float(best_classifier.predict_proba(input_df)[0][1])
                class_pred = int(prob_dropout_yes >= best_threshold)
                gpa_pred = float(best_regressor.predict(input_df)[0])
                prediction = CLASS_LABELS[class_pred]
                probability_percent = round(prob_dropout_yes * 100, 2)
                numeric_gpa = round(max(0.0, min(4.0, gpa_pred)), 2)
                burnout = "Yes" if numeric_gpa < 3.0 else "No"

    return render_template_string(
        HTML_TEMPLATE,
        fields=UI_FIELDS,
        values=values,
        student_name=student_name,
        prediction=prediction,
        probability_percent=probability_percent,
        numeric_gpa=numeric_gpa,
        burnout=burnout,
        error=error,
    )


if __name__ == "__main__":
    print("Training complete.")
    print("Classifier:", RESULTS["best_classifier"], RESULTS["test_classification_metrics"])
    print("Regressor:", RESULTS["best_regressor"], RESULTS["regression_results"][RESULTS["best_regressor"]])
    print("Open http://127.0.0.1:5000 in your browser.")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
