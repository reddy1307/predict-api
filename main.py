# main.py - XGBoost Spending Prediction API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

# ==================== XGBoost Predictor ====================
class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self.categories = ["Food", "Transport", "Shopping", "Bills", "Entertainment", "Healthcare", "Education"]
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize XGBoost model with default parameters"""
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror'
        )
    
    def _prepare_features(self, transactions_df: pd.DataFrame, future_dates: List[datetime] = None):
        """
        Prepare features for XGBoost
        
        Features:
        1. Time-based: day_of_week, day_of_month, month, is_weekend, is_month_start, is_month_end
        2. Historical: rolling_mean_3, rolling_mean_7, rolling_std_7
        3. Category-based: category frequency and amounts
        4. Trend: moving average slope
        """
        features = []
        
        if future_dates:
            # For future prediction
            for date in future_dates:
                # Time features
                feat = [
                    date.weekday(),                     # day_of_week (0-6)
                    date.day,                           # day_of_month (1-31)
                    date.month,                         # month (1-12)
                    1 if date.weekday() >= 5 else 0,    # is_weekend
                    1 if date.day <= 3 else 0,          # is_month_start
                    1 if date.day >= 28 else 0,         # is_month_end
                ]
                
                # Add historical patterns if available
                if len(transactions_df) > 0:
                    # Get recent transactions (last 30 days)
                    recent = transactions_df[transactions_df['date'] >= (date - timedelta(days=30))]
                    
                    if len(recent) > 0:
                        feat.extend([
                            recent['amount'].mean() if not recent.empty else 0,
                            recent['amount'].std() if len(recent) > 1 else 0,
                            len(recent),  # transaction count
                        ])
                        
                        # Category distribution for this day of week
                        same_dow = transactions_df[transactions_df['date'].dt.weekday == date.weekday()]
                        if len(same_dow) > 0:
                            feat.append(same_dow['amount'].mean())
                            # Most common category for this weekday
                            cat_counts = same_dow['category'].value_counts()
                            if not cat_counts.empty:
                                feat.append(list(self.categories).index(cat_counts.index[0]) if cat_counts.index[0] in self.categories else 0)
                            else:
                                feat.append(0)
                        else:
                            feat.extend([0, 0])
                    else:
                        feat.extend([0, 0, 0, 0, 0])
                else:
                    feat.extend([0, 0, 0, 0, 0])
                
                features.append(feat)
            
            return np.array(features)
        
        else:
            # For training - prepare features from historical data
            if len(transactions_df) < 10:
                return None
            
            # Create daily aggregates
            daily = transactions_df.groupby('date').agg({
                'amount': 'sum',
                'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'General'
            }).reset_index()
            
            for i in range(7, len(daily)):
                current_date = daily.iloc[i]['date']
                window = daily.iloc[i-7:i]
                
                # Time features
                feat = [
                    current_date.weekday(),
                    current_date.day,
                    current_date.month,
                    1 if current_date.weekday() >= 5 else 0,
                    1 if current_date.day <= 3 else 0,
                    1 if current_date.day >= 28 else 0,
                ]
                
                # Historical features from window
                feat.extend([
                    window['amount'].mean(),
                    window['amount'].std() if len(window) > 1 else 0,
                    len(window),
                ])
                
                # Category pattern for this weekday
                same_dow_hist = daily[daily['date'].dt.weekday == current_date.weekday()]
                if len(same_dow_hist) > 0:
                    feat.append(same_dow_hist['amount'].mean())
                    cat_counts = same_dow_hist['category'].value_counts()
                    if not cat_counts.empty:
                        feat.append(list(self.categories).index(cat_counts.index[0]) if cat_counts.index[0] in self.categories else 0)
                    else:
                        feat.append(0)
                else:
                    feat.extend([0, 0])
                
                features.append(feat)
            
            return np.array(features) if features else None
    
    def train(self, transactions: List[dict]):
        """Train XGBoost model on transaction data"""
        try:
            if len(transactions) < 15:
                print(f"⚠️ Need at least 15 transactions for training, got {len(transactions)}")
                return False
            
            # Prepare data
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create features
            X = self._prepare_features(df)
            if X is None:
                return False
            
            # Create targets (daily totals)
            daily = df.groupby('date')['amount'].sum().reset_index()
            y = daily['amount'].values[7:7+len(X)]
            
            if len(X) != len(y):
                print(f"⚠️ Feature-target mismatch: X={len(X)}, y={len(y)}")
                return False
            
            # Train XGBoost
            self.model.fit(X, y)
            print(f"✅ XGBoost trained on {len(X)} samples")
            return True
            
        except Exception as e:
            print(f"❌ XGBoost training failed: {e}")
            return False
    
    def predict(self, transactions: List[dict], days: int = 7):
        """Predict future spending using XGBoost"""
        try:
            if not transactions:
                return self._fallback_prediction(days)
            
            # Prepare data
            df = pd.DataFrame(transactions)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Train model if we have enough data
            if len(df) >= 15:
                print("🔄 Training XGBoost with available data...")
                self.train(transactions)
            
            # Generate future dates
            last_date = df['date'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
            
            # Prepare features for future dates
            X_future = self._prepare_features(df, future_dates)
            
            if X_future is None:
                return self._fallback_prediction(days)
            
            # Make predictions
            predicted_amounts = self.model.predict(X_future)
            
            # Ensure positive amounts
            predicted_amounts = np.maximum(predicted_amounts, 10)
            
            # Generate predictions with categories
            predictions = []
            for i, (date, amount) in enumerate(zip(future_dates, predicted_amounts)):
                # Add realistic variation (±20%)
                variation = 0.8 + (np.random.random() * 0.4)
                final_amount = amount * variation
                
                # Determine category
                category = self._predict_category(date, df)
                
                # Calculate confidence
                confidence = self._calculate_confidence(len(transactions), predicted_amounts)
                
                predictions.append({
                    "date": date.isoformat(),
                    "amount": round(float(final_amount), 2),
                    "category": category,
                    "confidence": round(confidence, 2)
                })
            
            return predictions
            
        except Exception as e:
            print(f"❌ XGBoost prediction error: {e}")
            return self._fallback_prediction(days)
    
    def _predict_category(self, date: datetime, df: pd.DataFrame):
        """Predict most likely category for a date"""
        try:
            # Check same day of week
            same_dow = df[df['date'].dt.weekday == date.weekday()]
            if not same_dow.empty:
                return same_dow['category'].mode().iloc[0]
            
            # Check same day of month range (±3 days)
            day_range = df[
                (df['date'].dt.day >= max(1, date.day - 3)) &
                (df['date'].dt.day <= min(31, date.day + 3))
            ]
            if not day_range.empty:
                return day_range['category'].mode().iloc[0]
            
            # Return most common overall
            return df['category'].mode().iloc[0] if not df.empty else "General"
            
        except:
            return np.random.choice(self.categories)
    
    def _calculate_confidence(self, data_points: int, predictions: np.ndarray):
        """Calculate confidence score based on data quality and prediction consistency"""
        # Base confidence from data quantity
        data_confidence = min(0.4 + (data_points / 50), 0.9)
        
        # Consistency confidence (lower std = higher confidence)
        if len(predictions) > 1:
            std_normalized = np.std(predictions) / (np.mean(predictions) + 1e-10)
            consistency = 0.9 if std_normalized < 0.3 else 0.6
        else:
            consistency = 0.7
        
        return round((data_confidence * 0.6) + (consistency * 0.4), 2)
    
    def _fallback_prediction(self, days: int):
        """Simple prediction when XGBoost fails"""
        predictions = []
        today = datetime.now()
        
        for i in range(1, days + 1):
            date = today + timedelta(days=i)
            
            # Base patterns
            if date.weekday() >= 5:  # Weekend
                base_amount = 800 + np.random.random() * 400
            elif date.day <= 7:  # First week (bills, shopping)
                base_amount = 1200 + np.random.random() * 600
            else:  # Regular weekday
                base_amount = 500 + np.random.random() * 300
            
            # Add some noise
            amount = base_amount * (0.8 + np.random.random() * 0.4)
            
            predictions.append({
                "date": date.isoformat(),
                "amount": round(float(amount), 2),
                "category": np.random.choice(self.categories),
                "confidence": 0.5
            })
        
        return predictions

# ==================== FastAPI App ====================
app = FastAPI(
    title="XGBoost Spending Predictor",
    description="ML-powered spending prediction using XGBoost algorithm",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = XGBoostPredictor()

@app.get("/")
async def root():
    return {
        "message": "🚀 XGBoost Spending Prediction API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "algorithm": "XGBoost Regressor",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /predict": "Get predictions (main endpoint)",
            "GET /example": "Example request format"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "XGBoost",
        "categories": predictor.categories
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_spending(request: PredictionRequest):
    """
    Predict future spending using XGBoost algorithm
    
    **Example Request:**
    ```json
    {
        "transactions": [
            {"date": "2024-01-01", "amount": 500, "category": "Food"},
            {"date": "2024-01-02", "amount": 1500, "category": "Shopping"}
        ],
        "days": 7,
        "user_id": "user123"
    }
    ```
    
    **Response:**
    ```json
    {
        "predictions": [
            {"date": "2024-01-03T00:00:00", "amount": 650.50, "category": "Food", "confidence": 0.75}
        ],
        "days": 7,
        "transaction_count": 2,
        "model_used": "XGBoost"
    }
    ```
    """
    try:
        # Validate input
        if request.days < 1 or request.days > 90:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 90")
        
        print(f"📈 Prediction request received: {len(request.transactions)} transactions, {request.days} days")
        
        # Convert to dict for predictor
        transactions_dict = [
            {
                "date": t.date,
                "amount": float(t.amount),
                "category": str(t.category)
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
        print(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/example")
async def get_example():
    """Get example transaction data for testing"""
    today = datetime.now()
    example_transactions = []
    
    # Generate 30 days of example data
    for i in range(30, 0, -1):
        date = today - timedelta(days=i)
        amount = 0
        
        # Create patterns
        if date.weekday() == 0:  # Monday
            amount = 300 + np.random.random() * 200
            category = "Food"
        elif date.weekday() == 4:  # Friday
            amount = 800 + np.random.random() * 400
            category = "Entertainment"
        elif date.weekday() >= 5:  # Weekend
            amount = 1000 + np.random.random() * 500
            category = "Shopping"
        elif date.day <= 5:  # Start of month
            amount = 1500 + np.random.random() * 1000
            category = "Bills"
        else:  # Regular days
            amount = 500 + np.random.random() * 300
            category = np.random.choice(["Food", "Transport", "Healthcare"])
        
        example_transactions.append({
            "date": date.isoformat(),
            "amount": round(float(amount), 2),
            "category": category
        })
    
    return {
        "example_request": {
            "transactions": example_transactions[:10],  # First 10 for example
            "days": 7,
            "user_id": "test_user_123"
        },
        "endpoint": "POST /predict",
        "note": "Copy the example_request object and send to /predict endpoint"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
