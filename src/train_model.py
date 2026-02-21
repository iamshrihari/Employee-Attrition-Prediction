from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def train_model(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.35).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print("\n✅ Model Accuracy:", round(acc * 100, 2), "%")
    print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

    # 🔹 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/confusion_matrix.png")
    plt.close()
    print("📊 Confusion matrix saved: output/confusion_matrix.png")

    # 🔹 Feature Importance
    importances = model.feature_importances_
    feature_names = X_train.columns

    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    feat_imp.plot(kind="barh")
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("output/feature_importance.png")
    plt.close()

    print("📊 Feature importance saved: output/feature_importance.png")

    # 🔹 Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/retention_model.pkl")
    print("💾 Model saved: models/retention_model.pkl")

    return model