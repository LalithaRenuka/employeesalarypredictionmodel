import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib

# Load data (rename 'salarydata.csv' as needed)
df = pd.read_csv('adult 3.csv')
df.columns = df.columns.str.strip()

target_col = 'income'  # Change to your column name
X = df.drop([target_col], axis=1)
y = df[target_col]

for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

y = LabelEncoder().fit_transform(y)  # 0/1 encoding

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'salary_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("Model and columns saved.")
