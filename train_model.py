import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
import joblib

ROOT = Path(__file__).resolve().parent
HEART_PATH = ROOT / "heart-data.csv"
STROKE_PATH = ROOT / "stroke-data.csv"
MODEL_PATH = ROOT / "model.joblib"
EVAL_PATH = ROOT / "evaluation.txt"

HEART_FEATURES = [
    "age",
    "gender",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
HEART_CATEGORICAL = ["gender"]
HEART_NUMERICAL = [f for f in HEART_FEATURES if f not in HEART_CATEGORICAL]

STROKE_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]
STROKE_CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
STROKE_NUMERICAL = [f for f in STROKE_FEATURES if f not in STROKE_CATEGORICAL]


def build_model(numerical, categorical):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    clf = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, best_f1


def fit_binary_model(df, features, target_col, numerical, categorical):
    X = df[features]
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    model = build_model(numerical=numerical, categorical=categorical)
    model.fit(X_train, y_train)

    class_list = list(model.named_steps["clf"].classes_)
    positive_index = class_list.index(1)

    val_proba = model.predict_proba(X_val)[:, positive_index]
    threshold, val_f1 = best_threshold(y_val.to_numpy(), val_proba)

    test_proba = model.predict_proba(X_test)[:, positive_index]
    y_test_pred = (test_proba >= threshold).astype(int)

    report = classification_report(y_test, y_test_pred, digits=3)
    cm = confusion_matrix(y_test, y_test_pred)

    return {
        "model": model,
        "threshold": threshold,
        "val_f1": val_f1,
        "test_report": report,
        "test_cm": cm,
        "target_distribution": y.value_counts().to_dict(),
    }


def load_heart_data():
    heart = pd.read_csv(HEART_PATH).copy()
    heart["gender"] = heart["sex"].map({1: "Male", 0: "Female"})
    heart = heart.drop(columns=["sex"])
    return heart


def load_stroke_data():
    return pd.read_csv(STROKE_PATH).copy()


def main():
    heart = load_heart_data()
    stroke = load_stroke_data()

    heart_fit = fit_binary_model(
        df=heart,
        features=HEART_FEATURES,
        target_col="target",
        numerical=HEART_NUMERICAL,
        categorical=HEART_CATEGORICAL,
    )

    stroke_fit = fit_binary_model(
        df=stroke,
        features=STROKE_FEATURES,
        target_col="stroke",
        numerical=STROKE_NUMERICAL,
        categorical=STROKE_CATEGORICAL,
    )

    payload = {
        "version": 2,
        "heart_model": heart_fit["model"],
        "stroke_model": stroke_fit["model"],
        "heart_features": HEART_FEATURES,
        "stroke_features": STROKE_FEATURES,
        "classes": ["neither", "heart_disease", "stroke", "both"],
        "thresholds": {
            "heart": heart_fit["threshold"],
            "stroke": stroke_fit["threshold"],
        },
    }
    joblib.dump(payload, MODEL_PATH)

    eval_text = [
        "HEART MODEL (target=heart disease from heart-data.csv)",
        f"Target distribution: {heart_fit['target_distribution']}",
        f"Best threshold on validation: {heart_fit['threshold']:.3f}",
        f"Validation F1 at threshold: {heart_fit['val_f1']:.3f}",
        "Classification report on test split:",
        heart_fit["test_report"],
        "Confusion matrix on test split:",
        str(heart_fit["test_cm"]),
        "",
        "STROKE MODEL (target=stroke from stroke-data.csv)",
        f"Target distribution: {stroke_fit['target_distribution']}",
        f"Best threshold on validation: {stroke_fit['threshold']:.3f}",
        f"Validation F1 at threshold: {stroke_fit['val_f1']:.3f}",
        "Classification report on test split:",
        stroke_fit["test_report"],
        "Confusion matrix on test split:",
        str(stroke_fit["test_cm"]),
        "",
        "Final app class mapping:",
        "heart=0, stroke=0 => neither",
        "heart=1, stroke=0 => heart_disease",
        "heart=0, stroke=1 => stroke",
        "heart=1, stroke=1 => both",
    ]
    EVAL_PATH.write_text("\n".join(eval_text), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation: {EVAL_PATH}")
    print(f"Heart threshold: {heart_fit['threshold']:.3f}")
    print(f"Stroke threshold: {stroke_fit['threshold']:.3f}")


if __name__ == "__main__":
    main()
