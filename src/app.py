## FastAPI app
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import io
from .Preprocess_v1 import preprocess_text
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File

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
@app.post("/predict_file")

async def predict_file(file: UploadFile = File(...)):

    ## Read and upload the input text
    contents = await file.read()
    if file.filename.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        df = pd.read_csv(io.BytesIO(contents))

    ## Preprocess the text data
    df['text'] = df['text'].astype(str).apply(preprocess_text)

     ## Assume the text column is named 'text'
    texts = df['text'].astype(str).tolist()
    x= vectorizer.transform(texts)
    preds = model.predict(x)
    pred_classes = label_encoder.inverse_transform(preds)

    ## Add predictions as a new column
    df['Predicted Class'] = pred_classes
    df['Confidence_score'] = model.predict_proba(x).max(axis=1)

    # Save to Excel in memory
    output = io.BytesIO()
    df.to_excel(output,index=False)
    output.seek(0)

    ## return as a downloadable file
    return StreamingResponse(
        output,
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers= {"Content-Disposition": f"attachment; filename={file.filename.split('.')[0]}_predictions.xlsx"}
    )

    
# ...existing code...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)