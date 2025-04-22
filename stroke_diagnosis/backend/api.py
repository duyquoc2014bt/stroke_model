from fastapi import FastAPI
from keras._tf_keras.keras.models import load_model
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = load_model("model.keras")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(req: InputData):
    input_data = np.array(req.features).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"prediction": prediction}
