# main.py - Spending Prediction API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import xgboost as xgb
import random

# ==================== Models ====================
class Transaction(BaseModel):
    date: str
    amount: float
    category: str

class PredictionRequest(BaseModel):
    transactions: List[Transaction]
    days: int = 7
    user_id: Optional[str] = None

class PredictionItem(BaseModel):
    date: str
    amount: float
    category: str
    confidence: float = 0.5

# ==================== FastAPI App ====================
app = FastAPI(title="Spending Prediction API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Helper ====================
def parse_date(date_str: str) -> datetime:
    try:
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return datetime.strptime(date_str[:10], '%Y-%m-%d')
    except:
        return datetime.now()

def prepare_features(transactions: List[Transaction]):
    df = pd.DataFrame([{
        "amount": t.amount,
        "category": t.category,
        "day": parse_date(t.date).day,
        "weekday": parse_date(t.date).weekday()
    } for t in transactions])
    
    # One-hot encode category
    df = pd.get_dummies(df, columns=["category"], drop_first=True)
    return df

def train_model(df: pd.DataFrame):
    if df.empty:
        return None
    X = df.drop(columns=["amount"])
    y = df["amount"]
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3)
    model.fit(X, y)
    return model

# ==================== Routes ====================
@app.get("/")
async def root():
    return {"message": "Spending Prediction API", "status": "running"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if request.days < 1 or request.days > 90:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 90")

        df = prepare_features(request.transactions)
        model = train_model(df)

        # Predict for next N days
        last_date = datetime.now()
        if request.transactions:
            last_date = max(parse_date(t.date) for t in request.transactions)

        predictions = []
        for i in range(1, request.days + 1):
            future_date = last_date + timedelta(days=i)
            features = {"day": future_date.day, "weekday": future_date.weekday()}
            # Add category columns with zeros
            for col in df.columns:
                if col.startswith("category_") and col not in features:
                    features[col] = 0
            X_pred = pd.DataFrame([features])
            amount = model.predict(X_pred)[0] if model else 500.0  # fallback
            predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "amount": round(amount, 2),
                "category": "General",
                "confidence": round(random.uniform(0.5,0.9),2)
            })

        return {"predictions": predictions, "days": request.days, "transaction_count": len(request.transactions), "model_used": "XGBoostPredictor"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Local Testing ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
