import os
import json
import joblib
import xgboost as xgb
import numpy as np
from Preprocess_v1 import preprocess_text
Artifact_path = ""

## Load Artifacts
print("Loading artifacts...")
vectorizer = joblib.load(os.path.join(Artifact_path,'vectorizer.joblib'))
label_encoder = joblib.load(os.path.join(Artifact_path,'label_encoder.joblib'))


## Prefer json model for stability
model = xgb.XGBClassifier()
model.load_model(os.path.join(Artifact_path,'xgb_model.json'))

## Load Metadata for sanity check
with open(os.path.join(Artifact_path,"metadata.json"),"r") as f:
    metadata = json.load(f)

print("Artifacts loaded successfully.")

## Prediction Function
def predict(texts):
    """
    Function to predict the class of the input text.
    It preprocesses the text, vectorizes it, and uses the model to predict.
    """
    if isinstance(texts,str):
        texts = [texts]

    ## Preprocess the input texts
    cleaned_texts = [preprocess_text(t) for t in texts]

    ## Vectorize the cleaned texts
    x = vectorizer.transform(cleaned_texts)

    ## Predict using the model
    preds = model.predict(x)
    probs = model.predict_proba(x)

    ## Decode the predictions
    labels = label_encoder.inverse_transform(preds)
    results = []

    for i , txt in enumerate(texts):
        results.append({
            "input":txt,
            "cleaned":cleaned_texts[i],
            "predicted_label": labels[i],
            "confidence":float(np.max(probs[i]))
        })
    return results