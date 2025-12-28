from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
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

# ==================== Helpers ====================
def parse_date(date_str: str):
    """Parse date string safely"""
    try:
        return datetime.fromisoformat(date_str[:10])
    except:
        return datetime.now()

def average_category_spending(transactions: List[Transaction]):
    """Return average spending per category"""
    category_totals = {}
    category_counts = {}
    for t in transactions:
        category_totals[t.category] = category_totals.get(t.category, 0) + t.amount
        category_counts[t.category] = category_counts.get(t.category, 0) + 1

    averages = {}
    for cat in category_totals:
        averages[cat] = category_totals[cat] / category_counts[cat]
    return averages

# ==================== Routes ====================
@app.get("/")
async def root():
    return {"message": "Spending Prediction API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if request.days < 1 or request.days > 90:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 90")

    averages = average_category_spending(request.transactions)
    predictions = []

    if not averages:
        # No transactions, fallback
        predictions.append({
            "category": "General",
            "amount": 500.0,
            "note": "No transaction history, using default"
        })
    else:
        for cat, avg in averages.items():
            predictions.append({
                "category": cat,
                "amount": round(avg * (request.days / 7), 2),  # scale by period
            })

    return {
        "predictions": predictions,
        "days": request.days,
        "transaction_count": len(request.transactions)
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
