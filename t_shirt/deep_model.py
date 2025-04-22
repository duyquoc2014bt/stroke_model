from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_set import data
from tensorflow.keras import layers, Sequential, regularizers
from sklearn.model_selection import train_test_split

#keras

X = data[["price", "size", "sold_qty"]]
y = data["is_liked"]

transform = LabelEncoder()
X['size'] = transform.fit_transform(X[['size']])
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

model = Sequential([
    layers.Dense(16, activation='relu', input_shape=(X_scaled.shape[1],), kernel_regularizer=regularizers.L2(0.01)),
    layers.Dense(8, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=8)
result = model.predict(X_test)
print("Sản phẩm bán chạy:", "Bán chạy" if bool(round(result[0][0])) else "Không bán chạy")

