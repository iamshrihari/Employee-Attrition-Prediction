🚀 Employee Attrition Prediction

An end-to-end Machine Learning project to predict employee attrition using imbalance-aware classification techniques. The model helps HR teams identify employees at risk of leaving and take proactive retention measures.

📌 Problem Statement

Employee attrition increases hiring costs, reduces productivity, and impacts organizational stability.
This project builds a predictive model to identify employees who are likely to leave the company.

Dataset : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset 
renamed as hr_data.csv

🧠 Approach

Data preprocessing and feature encoding

Handling class imbalance using SMOTE

Model training using Random Forest Classifier

Threshold tuning to improve recall

Performance evaluation using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Feature importance analysis

Model persistence using pickle

📊 Model Performance

Accuracy: ~79%

Recall (Attrition class): ~59%

Improved minority class detection using imbalance-aware techniques.


⚙️ Installation
git clone https://github.com/iamshrihari/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
▶️ Run The Project
python main.py


Outputs generated:

Confusion matrix

Feature importance plot

Predictions CSV

Trained model file

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Imbalanced-learn (SMOTE)

💼 Business Impact

By identifying employees at risk of attrition, organizations can:

Improve retention strategies

Reduce recruitment costs

Increase workforce stability

👨‍💻 Author

Shrihari
Machine Learning & AI Enthusiast