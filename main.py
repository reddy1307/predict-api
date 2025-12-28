# main.py - Spending Prediction API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import random
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
        # Try different date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        return datetime.now()
    except:
        return datetime.now()

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

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Validate days
        if request.days < 1 or request.days > 90:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 90")
        
        print(f"📊 Received request: {len(request.transactions)} transactions, {request.days} days")
        
        # Calculate average amount
        avg_amount = 500.0  # Default
        if request.transactions:
            total = sum(t.amount for t in request.transactions)
            avg_amount = total / len(request.transactions)
        
        # Find most common category
        common_category = "Food"
        if request.transactions:
            categories = [t.category for t in request.transactions]
            common_category = max(set(categories), key=categories.count)
        
        # Get last date or use today
        last_date = datetime.now()
        if request.transactions:
            dates = [parse_date(t.date) for t in request.transactions]
            last_date = max(dates)
        
        # Generate predictions
        predictions = []
        for i in range(1, request.days + 1):
            future_date = last_date + timedelta(days=i)
            day_of_week = future_date.weekday()
            
            # Base prediction logic
            amount = avg_amount
            
            # Adjust for day patterns
            if day_of_week >= 5:  # Weekend
                amount *= 1.3
            elif day_of_week == 0:  # Monday
                amount *= 0.9
            elif day_of_week == 4:  # Friday
                amount *= 1.2
            
            # Adjust for month start
            if future_date.day <= 7:
                amount *= 1.5
            
            # Add randomness
            amount *= (0.8 + random.random() * 0.4)
            
            # Ensure minimum
            amount = max(10.0, amount)
            
            # Calculate confidence
            confidence = 0.5
            if request.transactions:
                confidence = min(0.3 + (len(request.transactions) / 30), 0.9)
            
            predictions.append({
                "date": future_date.isoformat(),
                "amount": round(amount, 2),
                "category": common_category,
                "confidence": round(confidence, 2)
            })
        
        # Build response
        response = {
            "predictions": predictions,
            "days": request.days,
            "transaction_count": len(request.transactions),
            "model_used": "SmartPredictor"
        }
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==================== Test Endpoint ====================
@app.get("/test")
def test_endpoint():
    """Test endpoint to verify API is working"""
    test_transactions = [
        Transaction(date="2024-01-01", amount=500.0, category="Food"),
        Transaction(date="2024-01-02", amount=1500.0, category="Shopping")
    ]
    
    test_request = PredictionRequest(
        transactions=test_transactions,
        days=3
    )
    
    # Call predict function directly
    import asyncio
    response = asyncio.run(predict(test_request))
    
    return {
        "test_request": test_request.dict(),
        "test_response": response
    }

# For local development
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Spending Prediction API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
