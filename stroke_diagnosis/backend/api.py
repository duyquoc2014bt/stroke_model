from fastapi import FastAPI
from keras._tf_keras.keras.models import load_model
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.model.data_set import columns, preprocessor
import pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = load_model("model.keras")

class InputData(BaseModel):
    features: object

@app.post("/predict")
def predict(req: InputData):
    values = {k: [req.features[k]] for k in columns}
    input= pd.DataFrame(data=values) 
    input_transform = preprocessor.fit_transform(input)
    prediction = model.predict(input_transform)
    result = ["Có khả năng đột quỵ" if item > 0.51 else "Cần theo dõi" if item < 0.51 and item > 0.49 else "Bình thường" for item in prediction.flatten()]
    return {"prediction": result[0]}
