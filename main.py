# main.py - Spending Prediction API
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

class PredictionItem(BaseModel):
    date: str
    amount: float
    category: str
    confidence: float = 0.5

class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]
    days: int
    transaction_count: int
    model_used: str = "SmartPredictor"

# ==================== FastAPI App ====================
app = FastAPI(
    title="Spending Prediction API",
    version="1.0.0"
)

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

def calculate_average(transactions: List[Transaction]) -> float:
    """Calculate average transaction amount"""
    if not transactions:
        return 500.0  # Default average
    total = sum(t.amount for t in transactions)
    return total / len(transactions)

def get_most_common_category(transactions: List[Transaction]) -> str:
    """Get most common category from transactions"""
    if not transactions:
        return "Food"
    
    category_counts = {}
    for t in transactions:
        category_counts[t.category] = category_counts.get(t.category, 0) + 1
    
    return max(category_counts, key=category_counts.get)

def predict_amount(day_of_week: int, avg_amount: float, is_month_start: bool) -> float:
    """Predict amount based on day patterns"""
    base = avg_amount
    
    # Adjust for day of week
    if day_of_week >= 5:  # Weekend
        base *= 1.4
    elif day_of_week == 0:  # Monday
        base *= 0.8
    elif day_of_week == 4:  # Friday
        base *= 1.2
    
    # Adjust for month start (bills, shopping)
    if is_month_start:
        base *= 1.5
    
    # Add some randomness (±30%)
    variation = 0.7 + (random.random() * 0.6)
    return base * variation

def calculate_confidence(transaction_count: int) -> float:
    """Calculate confidence based on data quantity"""
    return min(0.3 + (transaction_count / 30), 0.9)

# ==================== Routes ====================
@app.get("/")
def root():
    return {
        "message": "Spending Prediction API",
        "status": "running",
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "POST /predict": "Get predictions"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Validate
        if request.days < 1 or request.days > 90:
            raise HTTPException(400, "Days must be between 1 and 90")
        
        # Calculate average amount
        avg_amount = calculate_average(request.transactions)
        common_category = get_most_common_category(request.transactions)
        
        # Get last date for prediction start
        last_date = datetime.now()
        if request.transactions:
            dates = [parse_date(t.date) for t in request.transactions]
            last_date = max(dates)
        
        # Generate predictions
        predictions = []
        for i in range(1, request.days + 1):
            future_date = last_date + timedelta(days=i)
            day_of_week = future_date.weekday()
            is_month_start = future_date.day <= 7
            
            # Predict amount
            amount = predict_amount(day_of_week, avg_amount, is_month_start)
            
            # Ensure minimum amount
            amount = max(10.0, amount)
            
            # Determine category
            category = common_category
            
            # Calculate confidence
            confidence = calculate_confidence(len(request.transactions))
            
            predictions.append(PredictionItem(
                date=future_date.isoformat(),
                amount=round(amount, 2),
                category=category,
                confidence=round(confidence, 2)
            ))
        
        return PredictionResponse(
            predictions=predictions,
            days=request.days,
            transaction_count=len(request.transactions),
            model_used="SmartPredictor"
        )
        
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
