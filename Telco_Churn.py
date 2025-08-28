## Importing the libraries

import pandas as pd

## Importing dataset

df = pd.read_csv("Data/Telco-Customer-Churn.csv")
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1].values 

## Fast EAD
#print(df.head())
#print(df.info())
#print(df["Churn"].value_counts())
#print(df["Churn"].value_counts(normalize=True))
#print(df.isnull().sum())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True) 


# One-hot encoding

cat_cols = ["gender", "Partner", "Dependents", "PhoneService", 
            "MultipleLines", "InternetService", "OnlineSecurity", 
            "OnlineBackup", "DeviceProtection", "TechSupport", 
            "StreamingTV", "StreamingMovies", "Contract", 
            "PaperlessBilling", "PaymentMethod"]


num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

X_num = df[num_cols]

# Cria dummies apenas das categóricas
X_cat = pd.get_dummies(df[cat_cols], drop_first=True)

# Junta os dois
X = pd.concat([X_num, X_cat], axis=1)
#print(X)
y = df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
#print(y)
#print(X)


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Models
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(max_iter=5000)
logit.fit(X_train,y_train)

y_pred_logit = logit.predict(X_test)
y_prob_logit = logit.predict_proba(X_test)[:,1]

from sklearn.tree import DecisionTreeClassifier
arvore = DecisionTreeClassifier(max_depth=4, random_state=42)
arvore.fit(X_train, y_train)

y_pred_tree = arvore.predict(X_test)
y_prob_tree = arvore.predict_proba(X_test)[:,1]

# Metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
cm = confusion_matrix(y_test, y_pred_logit)
print(f"Matrix de Confusão - Regressão Logistica:\n {cm}")
print(f"Acuracia : {accuracy_score(y_test, y_pred_logit):.2f}")
print(f"Precisão : {precision_score(y_test, y_pred_logit):.2f}")
print(f"Recall : {recall_score(y_test, y_pred_logit):.2f}")
print(f"AUC : {roc_auc_score(y_test, y_prob_logit):.2f}")



results = pd.DataFrame({
    "Modelo": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_logit),
        accuracy_score(y_test, y_pred_tree)
    ],
    "Precision": [
        precision_score(y_test, y_pred_logit),
        precision_score(y_test, y_pred_tree)
    ],
    "Recall": [
        recall_score(y_test, y_pred_logit),
        recall_score(y_test, y_pred_tree)
    ],
    "AUC": [
        roc_auc_score(y_test, y_prob_logit),
        roc_auc_score(y_test, y_prob_tree)
    ]
})

print(results)


# Visualizing the results

from sklearn import metrics
roc_logit = metrics.roc_curve(y_test, y_prob_logit)
roc_tree = metrics.roc_curve(y_test, y_prob_tree)

auc_logit = roc_auc_score(y_test, y_prob_logit)
auc_tree = roc_auc_score(y_test, y_prob_tree)

import matplotlib.pyplot as plt

plt.plot(roc_logit[0], roc_logit[1])
plt.plot(roc_tree[0], roc_tree[1])
plt.plot([0, 1], [0, 1], 'k--')
plt.grid(True)
plt.title("ROC Curve")
plt.xlabel('Especificidade')
plt.ylabel('Recall')
plt.legend({f"Tree: {auc_tree:.2f}", f"Logit: {auc_logit:.2f}"})
plt.show()


import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()