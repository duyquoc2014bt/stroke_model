import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, LSTM, Dense, Reshape, Bidirectional
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping




# Định dạng dữ liệu
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 64
MAX_TEXT_LENGTH = 64  # Độ dài tối đa của câu

# Đọc danh sách ảnh & nhãn
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "dataset")

def load_data(dataset_path):
    images = []
    labels = []
    label_dict = {}
    
    with open(os.path.join(dataset_path, "labels.txt"), "r", encoding="utf-8") as f:
        for line in f:
            filename, text = line.strip().split("|") #strip = trim in js
            label_dict[filename] = text

    for filename in os.listdir(os.path.join(dataset_path, "images")):
        if filename in label_dict:
            img_path = os.path.join(dataset_path, "images", filename)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Dùng adaptive threshold để làm nổi bật text
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Tìm contours để cắt vùng chứa text
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            # Tạo mask trắng để vẽ text region
            mask = np.ones_like(binary) * 255
            cv2.drawContours(mask, contours, -1, (0,), thickness=cv2.FILLED)
            # Cắt text từ ảnh gốc
            text_region = cv2.bitwise_and(gray, gray, mask=mask)

            # Resize ảnh về kích thước chuẩn
            img = cv2.resize(text_region, (IMAGE_WIDTH, IMAGE_HEIGHT)) #matran 32x128
            img = img/255 # Chuẩn hóa
            img = np.expand_dims(img, axis=-1)  # Thêm batch (1, 32, 128)

            images.append(img)
            labels.append(label_dict[filename])

    return np.array(images), labels
# Load dữ liệu
x_train, y_train = load_data(dataset_path)

# chuyen du lieu qua so
# Tạo bảng mã (char dictionary)

chars = set("".join(y_train))  # Lấy tất cả ký tự xuất hiện trong dataset
char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(chars))}  # enumerate tao turple array [(0, 'a'), (1, 'b'), ...]
index_to_char = {idx: char for char, idx in char_to_index.items()} #items() chuyen object ve lai turple [('a', '1), ('b', 2), ...]

# Chuyển đổi văn bản thành số
def text_to_labels(text):
    return [char_to_index[char] for char in text]
num_classes = len(char_to_index)+2
y_train_encoded = [text_to_labels(text) for text in y_train] # = y_true

# Padding các chuỗi về cùng độ dài
y_train_padded = pad_sequences(y_train_encoded, maxlen=MAX_TEXT_LENGTH, padding="post", value=num_classes-1)
print(y_train_padded)

#model

# CTC Loss function - Đã fix lỗi
def ctc_loss_lambda_func(y_true, y_pred):
    batch_size = tf.shape(y_true)[0] #y_true shape 0 sẽ là, [1,2,3,4,16,16,....] = (y_train_padded[number])
    input_length = tf.fill([batch_size, 1], IMAGE_WIDTH // 4)  # Tạo tensor lấp đầy IMAGE_WIDTH // 4 = 128/4 = 32, chia 4 vì 2 lần maxpooling 2D với kernel 2x2 giảm chiều đi, tensor batch x 1 = [[32],[32], [32], [32], ....]
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, num_classes-1), dtype="int32"), axis=-1, keepdims=True)

    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def build_crnn_model():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="image_input") #Dau vao mo hinh, chieu cao chieu dai, 1 kenh mau (Gray) (32, 128, 1)

    # CNN Backbone
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img) # tao ra 32 ma tran moi bang 32 ma tran kernel khac nhau, padding = 1 = same, padding = valid = 0, su dung relu de non-linear, (None, 32, 128, 32) = (batch, height, width, channels) 
    x = MaxPooling2D(pool_size=(2, 2))(x) # tao ra (None, 16, 64, 32)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x) #(None, 16, 64, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x) #(None, 8, 32, 64)
    x = BatchNormalization()(x) #normalize #(None, 8, 32, 64)

    # ✅ Sửa lỗi bằng cách reshape tensor trước khi đưa vào RNN
    x = Reshape((IMAGE_WIDTH // 4, (IMAGE_HEIGHT // 4) * 64))(x) #(None, 32, 512 = height * channels), x phai co so dims (None, 8, 32, 64) (batch, height, width, channels)

    # RNN (Bidirectional LSTM)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x) #dropout trannh overfiting
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)

    # Fully Connected Layer
    x = Dense(num_classes, activation="softmax", name="output")(x) # softmax la ham kich hoat activation function (kieu giong Relu, tanh, ...),  # +2 vì CTC cần một nhãn "blank" va bias

    model = Model(inputs=input_img, outputs=x)

    # Compile với CTC Loss
    model.compile(optimizer="adam", loss=ctc_loss_lambda_func) # dung thuat toan adam hoc nhanh hon vi tu dieu chinh learning rate(alpha)
    return model

# Xây dựng mô hình
model = build_crnn_model()
model.summary() #hien thi bang thong tin tung dong handle trong model

#training
# Huấn luyện mô hình
early_stopping = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
model.fit(x_train, y_train_padded, epochs=2200, batch_size=64, callbacks=[early_stopping]) #batch_size  = do dai cua mot phan tu y =  y_train_padded[number]

#Dự đoán 

def ctc_decoder(preds, beam_width=10):
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    
    # Greedy CTC Decode
    results = K.ctc_decode(preds, input_length=input_len, greedy=True)[0][0].numpy()

    decoded_texts = []
    for result in results:
        text = "".join([index_to_char.get(idx, "") for idx in result if idx < num_classes-1])  # Loại bỏ ký tự blank (-1)
        decoded_texts.append(text)

    return decoded_texts


# Đọc ảnh để test
file_path = os.path.join(current_dir, "test.png")
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))/255
img = np.expand_dims(img, axis=[0, -1])  # Thêm batch dimension
# Dự đoán
preds = model.predict(img)
decoded_text = ctc_decoder(preds) #argmax(axis=-1) lay ra index trong step (theo hang) ma dai dien gia tri co xac suat cao nhat.
print("Dự đoán:", decoded_text)
K.clear_session()

