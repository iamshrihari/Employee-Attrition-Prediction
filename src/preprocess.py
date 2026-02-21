import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 Apply SMOTE ONLY on training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test