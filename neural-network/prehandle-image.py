import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "aka.png")

def preprocess_image(image_path):
    # Đọc ảnh bằng OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Áp dụng GaussianBlur để giảm nhiễu
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Dùng adaptive threshold để làm nổi bật text
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    # Tìm contours để cắt vùng chứa text
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tạo mask trắng để vẽ text region
    mask = np.ones_like(binary) * 255
    cv2.drawContours(mask, contours, -1, (0,), thickness=cv2.FILLED)
 
    # Cắt text từ ảnh gốc
    text_region = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Chuẩn hóa kích thước ảnh
    resized = cv2.resize(text_region, (128, 32))   #W = 128 H=32
    return resized

class TextDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) #image_paths length de truyen vao idx

    def __getitem__(self, idx):
        image = preprocess_image(self.image_paths[idx])
        image = np.expand_dims(image, axis=0)  # Thêm kênh channel the hien so luong mau hinh can xu ly, them column cho b
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        label = self.labels[idx]
        return image, label

# Định nghĩa mô hình CNN + LSTM
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # tao ra 64 ma tran moi.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #trong ma tran 2x2 thi lay 1 gia tri lon nhat dien vao ma tran moi,
            # 128 ma tran co kich thuoc 8 x 32
            # 8 = H,  32 = W  
        )
        #input_size = so luong feature
        self.lstm = nn.LSTM(1024, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 3, 1)  # Định dạng (batch, height, width, channels) => batch, width, channel, height de sai cho reshape
        x = x.reshape(x.shape[0], x.shape[2], -1)  # reshape height, giu nguyen (batch, width, features) ma tran sau reshape (1, 32 none feature , 1024 feature hang)=> size 32(none-feature)x1024(feature)
        x, _ = self.lstm(x) #size 32x 1024 @ 1024(none-feature)x64(feature) = 32x64 && vi su dung bidirectional=True nen di chieu nguoc lai nen hidden*2 = 64*2 = 128 nen ma tran output 32x128
        x = self.fc(x)  #128x26 @ 32x128 = 32 x 26
        return x

# Khởi tạo mô hình
num_classes = 26  # Giả sử nhận diện ký tự từ A-Z
model = CRNN(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss()

# Ví dụ sử dụng
dataset = TextDataset([file_path], ["CTCLoss"])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for images, labels in dataloader:
    output = model(images)  # [1, 32, 26]

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-"  # "-" là ký tự blank trong CTC

#training
num_epochs = 10  # Số vòng lặp huấn luyện

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        optimizer.zero_grad()
        
        output = model(images)  # Dự đoán của mô hình (batch_size, 32, 26)

        # Chuyển label thành tensor phù hợp với CTC loss
        label_seq = torch.tensor([alphabet.index(c) for c in labels[0]], dtype=torch.long)
        label_seq = label_seq.unsqueeze(0)  # Định dạng (batch, seq_len)

        input_lengths = torch.full((1,), output.shape[1], dtype=torch.long)  # Chiều dài đầu ra
        target_lengths = torch.tensor([len(labels[0])], dtype=torch.long)  # Chiều dài nhãn

        loss = criterion(output.permute(1, 0, 2), label_seq, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

#predict
def greedy_decoder(output):
    output = torch.nn.functional.softmax(output, dim=2)  # Convert thành xác suất
    output = torch.argmax(output, dim=2)  # Lấy index có giá trị cao nhất
    decoded_text = []
    for seq in output:
        text = ""
        prev_char = None
        for idx in seq:
            char = alphabet[idx]
            if char != "-" and char != prev_char:  # Bỏ ký tự trùng do CTC
                text += char
            prev_char = char
        decoded_text.append(text)
    return decoded_text

decoded_text = greedy_decoder(output)
print(decoded_text)  
