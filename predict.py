import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("Employee_Performance_Retention.csv")

if "Employee_ID" in data.columns:
    data = data.drop("Employee_ID", axis=1)

encoder_dept = LabelEncoder()
data["Department"] = encoder_dept.fit_transform(data["Department"])

encoder_job = LabelEncoder()
data["Job_Satisfaction_Level"] = encoder_job.fit_transform(data["Job_Satisfaction_Level"])

encoder_promo = LabelEncoder()
data["Promotion_in_Last_2_Years"] = encoder_promo.fit_transform(data["Promotion_in_Last_2_Years"])

encoder_attrition = LabelEncoder()
data["Attrition"] = encoder_attrition.fit_transform(data["Attrition"])

features = ["Age", "Department", "Years_of_Experience", "Monthly_Working_Hours",
            "Training_Hours_per_Year", "Job_Satisfaction_Level", "Promotion_in_Last_2_Years"]

X = data[features]
y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder_attrition.classes_,
            yticklabels=encoder_attrition.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="AUC = {:.2f}".format(roc_auc))
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

sample_employee = X_test.iloc[0:1]
sample_prediction = model.predict(sample_employee)
predicted_attrition = encoder_attrition.inverse_transform(sample_prediction)[0]
print("\nExample Employee Features:")
print(sample_employee.iloc[0].to_dict())
print("Predicted Attrition:", predicted_attrition)
data.to_csv("Cleaned_Employee_Performance_Retention.csv", index=False)
