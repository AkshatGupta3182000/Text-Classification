## FastAPI app

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

## Load the model and vectorizer
vectorizer = joblib.load(r"C:\Users\ajayg\Downloads\Text_Classification\Artifacts\vectorizer.joblib")
model = joblib.load(r"C:\Users\ajayg\Downloads\Text_Classification\Artifacts\xgb_model.joblib")
label_encoder = joblib.load(r"C:\Users\ajayg\Downloads\Text_Classification\Artifacts\label_encoder.joblib")

## Initialize FastAPI app
app = FastAPI(title = "Text Classification API")

## Define input schema
class TextInput(BaseModel):
    text: str

## Define Route
@app.post("/predict")

def predict(input: TextInput):
    ## Preprocess and vectorize the input text
    x = vectorizer.transform([input.text])
    
    ## Make prediction
    pred = model.predict(x)[0]
    pred_class = label_encoder.inverse_transform([pred])[0]
    return {
        "text":input.text,
        "Predicted Class": pred_class,
    }

# ...existing code...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)