from src.preprocess import load_and_preprocess_data
from src.train_model import train_model
from src.predict import predict_attrition
from src.utils import print_section


def main():
    print_section("🚀 Employee Retention Prediction Started")

    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/hr_data.csv")
    print("✅ Data Loaded & Preprocessed!")

    train_model(X_train, X_test, y_train, y_test)

    predict_attrition(X_test)

    print_section("🔥 Project Completed Successfully!")


if __name__ == "__main__":
    main()