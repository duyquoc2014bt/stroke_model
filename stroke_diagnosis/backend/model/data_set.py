import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

data = pd.read_excel("backend/model/data.xlsx", engine="openpyxl", header=0)
print(data.columns.to_list())
test_data = pd.read_excel("backend/model/test.xlsx", engine="openpyxl", header=0)
X = data.drop(columns=['results'])
X_test = test_data.drop(columns=['results'])
y = data['results']

label_transform = OrdinalEncoder()
standard_scaler = StandardScaler()
preprocessor = ColumnTransformer(transformers=[("gen", label_transform, ['gender']), ("scale", standard_scaler, ['age','bmi'])], remainder="passthrough")
X = preprocessor.fit_transform(X)
X_test = preprocessor.fit_transform(X_test)