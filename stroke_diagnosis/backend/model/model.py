from data_set import X, y, X_test
from keras import layers, Sequential, regularizers

model = Sequential([
    layers.Dense(16, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=regularizers.L2(0.01)),
    layers.Dense(8, activation='relu', kernel_regularizer=regularizers.L2(0.01)),
    layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=8)
model.save("model.keras")
result = model.predict(X_test)

print(result)
print("kết quả dự đoán:", ["Có khả năng đột quỵ" if item > 0.51 else "Cần theo dõi" if item < 0.51 and item > 0.49 else "Bình thường" for item in result.flatten()])