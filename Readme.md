#  Text Classification API  

This project is a **FastAPI-based machine learning application** that classifies text into categories using a trained **XGBoost model**.   
It supports:
-  Classifying a **single text input**
-  Uploading an **Excel file** (`.xlsx`) with text column
-  Returning predictions along with **probability scores**

---

##  Project Structure

project-root/
-- app/
----- main.py # FastAPI app (API endpoints) ## single text version
----- main_v2.py # FastAPI app (API endpoints) ## Excel file upload version
----- preprocess.py # Preprocessing pipeline

-- artifacts/
----- model.joblib # Trained ML model
----- model.json # Trained ML model (JSON)
----- vectorizer.joblib # Trained Vectorizer
----- label_encoder.joblib # Trained label encoder
----- metadata.joblib # Model metadata


-- requirements.txt # Dependencies
-- README.md # Documentation
-- .gitignore # Files to ignore in Git

### Run the FastAPI app with Uvicorn
uvicorn app.main:app --reload
Swagger Docs:  http://127.0.0.1:8000/docs
ReDoc:  http://127.0.0.1:8000/redoc

## Model Details

-Algorithm: XGBoost
-Preprocessing: Custom heavy text preprocessing (tokenization, cleaning, vectorization)
-Saved with: joblib
-Stored in: models/model.joblib

-- Built by [Akshat Gupta]
-- For demo & learning purposes  
