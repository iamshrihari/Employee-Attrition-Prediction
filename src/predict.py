import pandas as pd
import joblib
import os


def predict_attrition(X_test):
    model = joblib.load("models/retention_model.pkl")

    preds = model.predict(X_test)

    result = X_test.copy()
    result["Predicted_Attrition"] = preds

    os.makedirs("output", exist_ok=True)
    result.to_csv("output/predictions.csv", index=False)

    print("✅ Predictions saved: output/predictions.csv")