import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


df = pd.read_csv("train.csv")

# print(df.head())

# print(df.info())

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# print(X.head())
# print(y.head())

X_train ,X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print(X_train.head())
# print(X_train.shape)
# print(X_valid.shape)
# print(y_train.shape)
# print(y_valid.shape)

# print(X_train.isnull().sum())


numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
numeric_transformer = SimpleImputer(strategy="median")


categorical_features = ["Sex", "Embarked"]

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)
# print(X_train.isnull().sum())
# print(X_valid.isnull().sum())

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:")
print(cm)

precision = precision_score(y_valid, y_pred)
print("Precision:", precision)

recall = recall_score(y_valid, y_pred)
print("Recall:", recall)

f1 = f1_score(y_valid, y_pred)
print("F1-score:", f1)


# save the trained model by importing joblib

joblib.dump(model, "model.pkl")
print("Model saved successfully!")