import numpy as np
from read_file import data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import SGDRegressor
z_score = StandardScaler()

def compute_loss(X, y, w, b):
    m = len(y)
    y_pred = X @ w + b  # Tính giá trị dự đoán
    loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)  # Công thức MSE
    return loss

X = data[["square_feet", "num_bedrooms", "num_bathrooms"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #random_state=42 cố định data test

m,n = X_train.shape
w = np.zeros(n) 
b = 0
alpha = 0.001
epochs = 290000

X_std = np.std(X_train, axis = 0, ddof=1)
X_mean = np.mean(X_train, axis = 0)
X_z_score_normalized = (X_train - X_mean)/X_std


for epoch in range(epochs):
    y_pred = np.dot(X_train, w) + b  
    error = y_pred - y_train  
    
    # Gradient Descent cập nhật w, b
    dw = (1/m) * np.dot(X_z_score_normalized.T, error) 
    db = (1/m) * np.sum(error)      

    w -= alpha * dw
    b -= alpha * db
    
    if(epoch % 100 == 0):
        loss = compute_loss(X_train, y_train, w, b)
        print(f'Epoch {epoch}: Loss = {loss}')


eq = "y = " + " + ".join([f"{w[i]:.4f} * x{i+1}" for i in range(n)]) + f" + {b:.4f}"
print("\n🔥 Phương trình hồi quy tuyến tính tối ưu:")
print(eq)

def predict(X_test):
    return np.dot(X_test, w) + b

print(predict(X_test))

# ## second way
# X_z_score_normalized = z_score.fit_transform(X_train)
# X_test_scaled = z_score.transform(X_test)


# sgd_reg = SGDRegressor(max_iter=epochs, tol=1e-3, learning_rate="constant", eta0=alpha)
# sgd_reg.fit(X_z_score_normalized, y_train)

# def predict(X_test):
#     print(sgd_reg.coef_, sgd_reg.intercept_)
#     return sgd_reg.predict(X_test)

# print(predict(X_test_scaled))
