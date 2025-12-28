# main.py - Spending Prediction API with XGBoost
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import json

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
app = FastAPI(
    title="Spending Prediction API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Helper Functions ====================
def parse_date(date_str: str) -> datetime:
    """Parse date string"""
    try:
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return datetime.strptime(date_str[:10], '%Y-%m-%d')
    except:
        return datetime.now()

def encode_category(category: str) -> int:
    """Simple category encoding"""
    mapping = {"Food": 0, "Transport": 1, "Shopping": 2, "Bills": 3, "Other": 4}
    return mapping.get(category, 4)

# ==================== Load or Train Model ====================
MODEL_PATH = "xgb_model.json"

if os.path.exists(MODEL_PATH):
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(MODEL_PATH)
else:
    # Train a dummy model if no model exists
    X_dummy = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]])
    y_dummy = np.array([100,200,150,300,250])
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_dummy, y_dummy)
    xgb_model.save_model(MODEL_PATH)

# ==================== Routes ====================
@app.get("/")
async def root():
    return {"message": "Spending Prediction API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if request.days < 1 or request.days > 90:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 90")

        # Aggregate transactions by category
        df = pd.DataFrame([{
            "days_since_start": (parse_date(t.date) - parse_date(request.transactions[0].date)).days,
            "category_encoded": encode_category(t.category),
            "amount": t.amount
        } for t in request.transactions])

        predictions = []
        last_date = max(df["days_since_start"]) if not df.empty else 0

        for i in range(1, request.days + 1):
            day_future = last_date + i
            # Predict spending amount per category
            pred_amounts = []
            for cat in df["category_encoded"].unique():
                X_test = np.array([[day_future, cat]])
                pred = xgb_model.predict(X_test)[0]
                pred_amounts.append({
                    "date": (parse_date(request.transactions[0].date) + pd.Timedelta(days=day_future)).strftime("%Y-%m-%d"),
                    "amount": round(float(pred), 2),
                    "category": list(df[df["category_encoded"]==cat]["category_encoded"].index)[0],
                    "confidence": 0.7
                })
            predictions.extend(pred_amounts)

        return {"predictions": predictions, "days": request.days, "transaction_count": len(request.transactions)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
