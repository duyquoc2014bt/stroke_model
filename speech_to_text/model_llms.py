import whisper
import torch
from data import data
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
from datasets import Dataset

# Load mô hình Whisper
model = whisper.load_model("small")

# Mã hóa âm thanh thành đặc trưng
def preprocess_audio(example):
    audio_path = example["audio"]
    audio_features = whisper.load_audio(audio_path)
    return {"input_features": audio_features}

# Áp dụng preprocessing
dataset = Dataset.from_list(list(data['train'][0]['train'])).map(preprocess_audio)
# Load mô hình Whisper
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small") 
# Load processor để xử lý văn bản
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Hàm xử lý đầu vào đầu ra
def preprocess_function(batch): #chuyen doi cac dang phu hop de training trong fire-tune
    #feature_extractor chuyen input_feature => input_ids
    inputs = processor.feature_extractor(batch["input_features"], return_tensors="pt").input_features.squeeze(0)
    labels = processor.tokenizer(batch["text"], return_tensors="pt", padding="max_length",  # Padding để labels có cùng độ dài
        truncation=True,       # Cắt bớt nếu quá dài
        max_length=50 ).input_ids.squeeze(0) #squeeze(0) bo di chieu dau tien (1,50) => (50)
    return {"input_features": inputs, "labels": labels}

dataset = dataset.map(preprocess_function, remove_columns=['audio', 'text'])
eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

# Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./whisper_finetuned",  # Thư mục lưu checkpoint
    evaluation_strategy="epoch",  # Đánh giá mô hình mỗi epoch
    save_strategy="epoch",  # Lưu checkpoint mỗi epoch
    per_device_train_batch_size=2,  #số lượng mẫu dữ liệu được xử lý trên mỗi GPU/CPU trong một batch (tang neu GPU manh)
    per_device_eval_batch_size=2,  # số lượng mẫu dữ liệu được xử lý trên mỗi GPU/CPU trong một batch (tang neu GPU manh)
    learning_rate=1e-5,  # Tốc độ học (có thể giảm xuống 1e-6 nếu overfit)
    num_train_epochs=5,  # Số epoch train
    logging_dir="./logs",  # Thư mục lưu log
    logging_steps=10,  # Log loss mỗi 10 step
    save_total_limit=2,  # Chỉ giữ lại 2 checkpoint gần nhất (tránh tốn dung lượng)
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset, 
    eval_dataset=eval_dataset,
    # tokenizer=processor.tokenizer,
)
# Bắt đầu fine-tuning
trainer.train()
trainer.save_model("./whisper_finetuned")

# Load mô hình đã fine-tune
fine_tuned_model = WhisperForConditionalGeneration.from_pretrained("./whisper_finetuned")
# Load file âm thanh
audio_path = "speech_to_text/dataset/dq.m4a"
audio_features = whisper.load_audio(audio_path)
# Dự đoán
input_features = processor(audio_features, return_tensors="pt").input_features
fine_tuned_model.config.forced_decoder_ids = None
fine_tuned_model.generation_config.forced_decoder_ids = None
generated_ids = fine_tuned_model.generate(input_features)

# Chuyển kết quả sang văn bản
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
print("Kết quả nhận dạng:", transcription[0])
