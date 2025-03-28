from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from read_file import data

X = data[["square_feet", "num_bedrooms", "num_bathrooms"]]
y = data["price"]

def linear_gradient(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random_state=42 cố định data test

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred) # trung bình độ lệch (tổng(y - yi)/n)
    mse = mean_squared_error(y_test, y_pred) # trung bình phương sai (tổng X^2/n)
    rmse = np.sqrt(mse)
    
    return [mae, mse, rmse]

mae, mse, rmse = linear_gradient(X, y)
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
