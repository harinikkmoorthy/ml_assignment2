import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Load dataset
data = pd.read_csv("HeartDiseaseTrain-Test (1).csv")

# Encode categorical variables
categorical_cols = ["sex","chest_pain_type","fasting_blood_sugar","rest_ecg",
                    "exercise_induced_angina","slope","vessels_colored_by_flourosopy","thalassemia"]
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]


log_reg_scores = []
log_reg_scores.append(accuracy_score(y_test,y_pred))
log_reg_scores.append(roc_auc_score(y_test,y_prob))
log_reg_scores.append(precision_score(y_test,y_pred))
log_reg_scores.append(recall_score(y_test,y_pred))
log_reg_scores.append(f1_score(y_test,y_pred))
log_reg_scores.append(matthews_corrcoef(y_test,y_pred))


