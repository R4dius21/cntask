import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Employee_Performance_Retention.csv')
df = df.drop(columns=['Employee_ID'], errors='ignore')

le_dept = LabelEncoder()
df['Department'] = le_dept.fit_transform(df['Department'])

le_js = LabelEncoder()
df['Job_Satisfaction_Level'] = le_js.fit_transform(df['Job_Satisfaction_Level'])

le_promo = LabelEncoder()
df['Promotion_in_Last_2_Years'] = le_promo.fit_transform(df['Promotion_in_Last_2_Years'])

le_attrition = LabelEncoder()
df['Attrition'] = le_attrition.fit_transform(df['Attrition'])

feature_cols = ['Age', 'Department', 'Years_of_Experience', 'Monthly_Working_Hours',
                'Training_Hours_per_Year', 'Job_Satisfaction_Level', 'Promotion_in_Last_2_Years']
X = df[feature_cols]
y = df['Attrition']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set train_test_split random_state to None for different splits each run
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, shuffle=True, random_state=None)

kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    # Remove fixed random_state here to let SVM internal randomness vary by run
    svm_clf = SVC(kernel=kernel, probability=True)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTraining SVM with {kernel} kernel:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le_attrition.classes_, yticklabels=le_attrition.classes_)
    plt.title(f'Confusion Matrix - {kernel} kernel')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    y_prob = svm_clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title(f'ROC Curve - {kernel} kernel')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Use numpy's random seed None for truly random sample selection each run
np.random.seed(None)
random_index = np.random.randint(0, X_test.shape[0])
sample = X_test[random_index].reshape(1, -1)
sample_orig = X.iloc[random_index].to_dict()
pred = svm_clf.predict(sample)
pred_label = le_attrition.inverse_transform(pred)[0]

print(f"\nSample Employee Features: {sample_orig}")
print(f"Predicted Attrition by SVM ({kernel} kernel): {pred_label}")

df.to_csv('Cleaned_Employee_Performance_Retention.csv', index=False)
