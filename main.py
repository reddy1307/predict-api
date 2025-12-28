# main.py - Category-wise Spending Prediction with XGBoost
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np
import xgboost as xgb

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
    category: str
    predicted_amount: float
    daily_average: float
    confidence: float
    model: str

# ==================== FastAPI App ====================
app = FastAPI(
    title="Spending Prediction API - XGBoost",
    version="1.1.0"
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
    try:
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return datetime.strptime(date_str[:10], '%Y-%m-%d')
    except:
        return datetime.now()

def predict_category_xgb(txns: List[dict], future_days: int):
    """Predict total spending for a category using XGBoost"""
    if not txns:
        return 0.0, 0.0

    txns = sorted(txns, key=lambda x: x["date"])
    X = np.array(range(len(txns))).reshape(-1, 1)  # Day index
    y = np.array([t["amount"] for t in txns])

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='reg:squarederror'
    )
    model.fit(X, y)

    future_X = np.array(range(len(txns), len(txns) + future_days)).reshape(-1, 1)
    preds = model.predict(future_X)
    preds = np.clip(preds, 0, None)

    total_pred = float(np.sum(preds))
    daily_avg = total_pred / future_days
    return total_pred, daily_avg

# ==================== Routes ====================
@app.get("/")
async def root():
    return {"message": "Spending Prediction API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if request.days not in [7, 14, 30, 90]:
        raise HTTPException(status_code=400, detail="Days must be 7, 14, 30, or 90")
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    # Group transactions by category
    categories = {}
    for t in request.transactions:
        date_obj = parse_date(t.date)
        categories.setdefault(t.category, []).append({
            "date": date_obj,
            "amount": t.amount
        })

    predictions = []
    for cat, txns in categories.items():
        total_pred, daily_avg = predict_category_xgb(txns, request.days)
        confidence = min(0.5 + (len(txns)/40), 0.95)  # More transactions → higher confidence

        predictions.append({
            "category": cat,
            "predicted_amount": round(total_pred, 2),
            "daily_average": round(daily_avg, 2),
            "confidence": round(confidence, 2),
            "model": "XGBoost"
        })

    return {
        "days": request.days,
        "categories_count": len(predictions),
        "predictions": predictions,
        "model_used": "CategoryWiseXGBoost"
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
