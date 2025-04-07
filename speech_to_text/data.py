from datasets import load_dataset

# Load dataset từ file JSON hoặc CSV
data = load_dataset("json", data_files={"speech_to_text/dataset.json"})
