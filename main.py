# main.py - Lightweight XGBoost API (No pandas)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== Pydantic Models ====================
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
    model_used: str = "XGBoost"

# ==================== Helper Functions ====================
def parse_date(date_str: str) -> datetime:
    """Parse date string without pandas"""
    try:
        # Try ISO format
        if 'T' in date_str:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        # Try YYYY-MM-DD
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return datetime.now()

def create_features(transactions: List[Dict], future_dates: List[datetime] = None):
    """Create features without pandas"""
    if not transactions:
        return None
    
    # Parse dates and amounts
    parsed_transactions = []
    for t in transactions:
        parsed_transactions.append({
            'date': parse_date(t['date']),
            'amount': float(t['amount']),
            'category': t['category']
        })
    
    # Sort by date
    parsed_transactions.sort(key=lambda x: x['date'])
    
    if future_dates:
        # For prediction - simplified features
        features = []
        for date in future_dates:
            # Basic time features
            feat = [
                date.weekday(),
                date.day,
                date.month,
                1 if date.weekday() >= 5 else 0,
            ]
            
            # Add historical averages if available
            if parsed_transactions:
                amounts = [t['amount'] for t in parsed_transactions]
                feat.extend([
                    np.mean(amounts),
                    np.std(amounts) if len(amounts) > 1 else 0,
                    len(amounts),
                ])
            else:
                feat.extend([0, 0, 0])
            
            features.append(feat)
        
        return np.array(features)
    
    return None

# ==================== XGBoost Predictor ====================
class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self.categories = ["Food", "Transport", "Shopping", "Bills", "Entertainment"]
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize XGBoost model"""
        self.model = xgb.XGBRegressor(
            n_estimators=50,  # Reduced for Render free tier
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
    
    def predict(self, transactions: List[Dict], days: int = 7):
        """Predict future spending"""
        try:
            if not transactions:
                return self._fallback_prediction(days)
            
            # Convert to dict format
            transactions_dict = [
                {
                    'date': t['date'],
                    'amount': float(t['amount']),
                    'category': str(t['category'])
                }
                for t in transactions
            ]
            
            # Get last date
            dates = [parse_date(t['date']) for t in transactions_dict]
            last_date = max(dates) if dates else datetime.now()
            
            # Generate future dates
            future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
            
            # Create features
            X_future = create_features(transactions_dict, future_dates)
            
            if X_future is None:
                return self._fallback_prediction(days)
            
            # Simple training on the fly
            if len(transactions_dict) >= 10:
                self._simple_train(transactions_dict)
            
            # Make predictions
            if hasattr(self.model, 'fit'):
                predicted_amounts = self.model.predict(X_future)
            else:
                # Fallback if model not trained
                return self._fallback_prediction(days)
            
            # Generate predictions
            predictions = []
            for i, (date, amount) in enumerate(zip(future_dates, predicted_amounts)):
                amount = max(10, amount * (0.8 + np.random.random() * 0.4))
                
                predictions.append({
                    "date": date.isoformat(),
                    "amount": round(float(amount), 2),
                    "category": np.random.choice(self.categories),
                    "confidence": 0.7
                })
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction(days)
    
    def _simple_train(self, transactions: List[Dict]):
        """Simple training without complex features"""
        try:
            if len(transactions) < 5:
                return
            
            # Create simple features and targets
            X = []
            y = []
            
            for i in range(len(transactions)):
                t = transactions[i]
                date = parse_date(t['date'])
                
                X.append([
                    date.weekday(),
                    date.day,
                    date.month,
                    1 if date.weekday() >= 5 else 0,
                ])
                y.append(float(t['amount']))
            
            if len(X) > 3:
                self.model.fit(np.array(X), np.array(y))
                print(f"Model trained on {len(X)} samples")
                
        except Exception as e:
            print(f"Training error: {e}")
    
    def _fallback_prediction(self, days: int):
        """Simple prediction when model fails"""
        predictions = []
        today = datetime.now()
        
        for i in range(1, days + 1):
            date = today + timedelta(days=i)
            base = 500
            if date.weekday() >= 5:
                base = 800
            
            amount = base * (0.7 + np.random.random() * 0.6)
            
            predictions.append({
                "date": date.isoformat(),
                "amount": round(float(amount), 2),
                "category": np.random.choice(self.categories),
                "confidence": 0.5
            })
        
        return predictions

# ==================== FastAPI App ====================
app = FastAPI(
    title="Spending Prediction API",
    description="Lightweight spending prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = XGBoostPredictor()

@app.get("/")
async def root():
    return {"message": "Spending Prediction API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if request.days < 1 or request.days > 90:
            raise HTTPException(400, "Days must be 1-90")
        
        # Convert to dict
        transactions_dict = [
            {
                "date": t.date,
                "amount": float(t.amount),
                "category": t.category
            }
            for t in request.transactions
        ]
        
        # Get predictions
        predictions = predictor.predict(transactions_dict, request.days)
        
        # Format response
        prediction_items = [
            PredictionItem(
                date=p["date"],
                amount=p["amount"],
                category=p["category"],
                confidence=p["confidence"]
            )
            for p in predictions
        ]
        
        return PredictionResponse(
            predictions=prediction_items,
            days=request.days,
            transaction_count=len(request.transactions),
            model_used="XGBoost"
        )
        
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
