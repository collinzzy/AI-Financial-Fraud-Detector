import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    def __init__(self):
        self.models = {}
        self.performance_metrics = {}
        
    def detect_fraud(self, model, X_processed, original_df, risk_threshold=0.5):
        """Detect fraud using trained model"""
        # Get predictions
        if hasattr(model, 'predict_proba'):
            # Classification models
            fraud_proba = model.predict_proba(X_processed)[:, 1]
            predictions = (fraud_proba >= risk_threshold).astype(int)
        else:
            # Anomaly detection models (like Isolation Forest)
            predictions = model.predict(X_processed)
            # Convert isolation forest predictions (1 for normal, -1 for anomaly)
            predictions = (predictions == -1).astype(int)
            fraud_proba = model.decision_function(X_processed)
            # Convert to probability-like scores
            fraud_proba = 1 / (1 + np.exp(-fraud_proba))
        
        # Create results dataframe
        results_df = original_df.copy()
        results_df['risk_score'] = fraud_proba
        results_df['is_fraud'] = predictions
        results_df['risk_level'] = results_df['risk_score'].apply(self._get_risk_level)
        
        return results_df
    
    def _get_risk_level(self, risk_score):
        """Convert risk score to risk level"""
        if risk_score >= 0.8:
            return 'Very High'
        elif risk_score >= 0.6:
            return 'High'
        elif risk_score >= 0.4:
            return 'Medium'
        elif risk_score >= 0.2:
            return 'Low'
        else:
            return 'Very Low'
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred = (y_pred == -1).astype(int)
            y_pred_proba = model.decision_function(X_test)
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_score': auc_score,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def analyze_fraud_patterns(self, results_df):
        """Analyze patterns in detected fraud"""
        fraud_df = results_df[results_df['is_fraud'] == 1]
        
        if len(fraud_df) == 0:
            return {"message": "No fraud detected"}
        
        patterns = {
            'top_fraud_categories': fraud_df['category'].value_counts().head(5).to_dict(),
            'top_fraud_merchants': fraud_df['merchant'].value_counts().head(5).to_dict(),
            'fraud_by_hour': fraud_df['hour_of_day'].value_counts().sort_index().to_dict(),
            'avg_fraud_amount': fraud_df['amount'].mean(),
            'total_fraud_amount': fraud_df['amount'].sum(),
            'fraud_rate_by_location': fraud_df['location'].value_counts(normalize=True).to_dict()
        }
        
        return patterns
    
    def generate_fraud_report(self, results_df):
        """Generate comprehensive fraud report"""
        fraud_cases = results_df[results_df['is_fraud'] == 1]
        
        report = {
            'summary': {
                'total_transactions': len(results_df),
                'fraud_cases': len(fraud_cases),
                'fraud_rate': len(fraud_cases) / len(results_df),
                'total_fraud_amount': fraud_cases['amount'].sum(),
                'avg_fraud_amount': fraud_cases['amount'].mean()
            },
            'high_risk_cases': fraud_cases.nlargest(10, 'risk_score')[['transaction_id', 'amount', 'risk_score', 'merchant']].to_dict('records'),
            'risk_distribution': results_df['risk_level'].value_counts().to_dict()
        }
        
        return report
    
    def save_model(self, model, filepath):
        """Save trained model to file"""
        joblib.dump(model, filepath)
    
    def load_model(self, filepath):
        """Load trained model from file"""
        return joblib.load(filepath)