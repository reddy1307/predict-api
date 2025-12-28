from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# ==================== Models ====================
class Transaction(BaseModel):
    date: str
    amount: float
    category: str

class PredictionRequest(BaseModel):
    transactions: List[Transaction]
    days: int = 7
    user_id: Optional[str] = None

# ==================== FastAPI App ====================
app = FastAPI(title="Spending Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Helpers ====================
def parse_date(date_str: str):
    try:
        return datetime.fromisoformat(date_str[:10])
    except:
        return datetime.now()

def prepare_features(transactions: List[Transaction]):
    df = pd.DataFrame([{
        "day": parse_date(t.date).day,
        "weekday": parse_date(t.date).weekday(),
        "amount": t.amount
    } for t in transactions])
    return df

# ==================== Routes ====================
@app.get("/")
async def root():
    return {"message": "Spending Prediction API", "status": "running"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if request.days < 1 or request.days > 90:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 90")

    df = prepare_features(request.transactions)

    predictions = []
    if not df.empty:
        # Train simple linear regression
        X = df[["day", "weekday"]]
        y = df["amount"]
        model = LinearRegression()
        model.fit(X, y)

        # Predict for next N days
        last_date = max(parse_date(t.date) for t in request.transactions)
        for i in range(1, request.days + 1):
            future_date = last_date + timedelta(days=i)
            X_pred = [[future_date.day, future_date.weekday()]]
            amount = model.predict(X_pred)[0]
            predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "amount": round(amount, 2),
                "category": "General",
            })
    else:
        # Fallback if no transactions
        for i in range(1, request.days + 1):
            future_date = datetime.now() + timedelta(days=i)
            predictions.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "amount": 500.0,
                "category": "General",
            })

    return {"predictions": predictions, "days": request.days, "transaction_count": len(request.transactions)}
