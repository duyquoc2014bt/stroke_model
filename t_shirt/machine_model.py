from data_set import data
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing  import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#data handling
X = data[["price", "size", "sold_qty"]]
y = data["is_liked"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#preprocessor
preprocessor = ColumnTransformer(transformers=[("number", StandardScaler(), ["price", "sold_qty"] ),("category", OneHotEncoder(), ["size"])])

#pipeline
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(random_state=42))])

#model training
model.fit(X_train, y_train)
predict = model.predict(X_test)
result = bool(predict[0])
print(result)
#RandomForestClassifier
model_random_forest = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))])

#model training
model_random_forest.fit(X_train, y_train)
predict_random_forest = model_random_forest.predict(X_test)
result_random_forest = bool(predict_random_forest[0])
print(result_random_forest)
#XGBoost