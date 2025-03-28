import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from read_file import data

#the first way
X = data[["square_feet", "num_bedrooms", "num_bathrooms"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#the first way
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_poly, y_train)  

print(np.dot(X_test_poly,model.coef_) + model.intercept_)

# #the second way
# model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
# model.fit(X_train, y_train)

# print(model.predict(X_test))
