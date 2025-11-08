import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        
    def train_model(self, X_processed, model_type='Isolation Forest', test_size=0.2, 
                   n_estimators=100, random_state=42):
        """Train fraud detection model"""
        
        # Since we don't have real labels in sample data, we'll simulate them
        # In a real scenario, you'd have labeled fraud data
        y = self._generate_synthetic_labels(X_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self._handle_imbalance(X_train, y_train)
        
        # Train model based on type
        if model_type == "Isolation Forest":
            model = self._train_isolation_forest(X_train_resampled, n_estimators, random_state)
        elif model_type == "Random Forest":
            model = self._train_random_forest(X_train_resampled, y_train_resampled, n_estimators, random_state)
        elif model_type == "XGBoost":
            model = self._train_xgboost(X_train_resampled, y_train_resampled, n_estimators, random_state)
        elif model_type == "Logistic Regression":
            model = self._train_logistic_regression(X_train_resampled, y_train_resampled, random_state)
        elif model_type == "Ensemble":
            model = self._train_ensemble(X_train_resampled, y_train_resampled, n_estimators, random_state)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate model
        performance = self._evaluate_model(model, X_test, y_test, model_type)
        
        # Store feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_type] = dict(zip(X_processed.columns, model.feature_importances_))
        
        self.models[model_type] = model
        self.best_model = model
        
        return model, performance
    
    def _generate_synthetic_labels(self, X_processed):
        """Generate synthetic fraud labels based on feature patterns"""
        np.random.seed(42)
        
        # Create labels based on feature combinations that might indicate fraud
        n_samples = len(X_processed)
        y = np.zeros(n_samples)
        
        # Fraud pattern 1: High amount with unusual time
        pattern1_mask = (
            (X_processed['amount'] > 1.5) &  # High standardized amount
            ((X_processed['hour_of_day'] < 6) | (X_processed['hour_of_day'] > 22))  # Unusual hours
        )
        
        # Fraud pattern 2: High frequency with large amounts
        pattern2_mask = (
            (X_processed['transaction_count_24h'] > 1.5) &  # High frequency
            (X_processed['amount_to_avg_ratio'] > 2.0)  # Much larger than average
        )
        
        # Fraud pattern 3: New customer with large transaction
        pattern3_mask = (
            (X_processed['customer_age_days'] < -0.5) &  # New customer (standardized)
            (X_processed['amount'] > 1.0)  # Medium to large amount
        )
        
        # Combine patterns
        fraud_mask = pattern1_mask | pattern2_mask | pattern3_mask
        
        # Set fraud labels with some noise
        y[fraud_mask] = np.random.choice([0, 1], size=fraud_mask.sum(), p=[0.3, 0.7])
        
        # Add some random fraud cases
        n_random_fraud = max(1, int(0.02 * n_samples))  # 2% random fraud
        random_indices = np.random.choice(np.where(~fraud_mask)[0], n_random_fraud, replace=False)
        y[random_indices] = 1
        
        return y.astype(int)
    
    def _handle_imbalance(self, X_train, y_train):
        """Handle class imbalance using SMOTE and undersampling"""
        # Since fraud is rare, we'll use a combination of over and undersampling
        over = SMOTE(sampling_strategy=0.1, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        
        pipeline = Pipeline([
            ('over', over),
            ('under', under)
        ])
        
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
        
        return X_resampled, y_resampled
    
    def _train_isolation_forest(self, X_train, n_estimators, random_state):
        """Train Isolation Forest for anomaly detection"""
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=0.1,  # Expected proportion of anomalies
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train)
        return model
    
    def _train_random_forest(self, X_train, y_train, n_estimators, random_state):
        """Train Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_xgboost(self, X_train, y_train, n_estimators, random_state):
        """Train XGBoost classifier"""
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_logistic_regression(self, X_train, y_train, random_state):
        """Train Logistic Regression model"""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000,
            C=0.1
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_ensemble(self, X_train, y_train, n_estimators, random_state):
        """Train ensemble of models"""
        from sklearn.ensemble import VotingClassifier
        
        # Define individual models
        models = [
            ('rf', RandomForestClassifier(n_estimators=n_estimators//2, random_state=random_state)),
            ('xgb', XGBClassifier(n_estimators=n_estimators//2, random_state=random_state)),
            ('lr', LogisticRegression(class_weight='balanced', random_state=random_state))
        ]
        
        # Create ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def _evaluate_model(self, model, X_test, y_test, model_type):
        """Evaluate model performance"""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            # For Isolation Forest
            y_pred = model.predict(X_test)
            y_pred = (y_pred == -1).astype(int)
            y_pred_proba = model.decision_function(X_test)
            # Convert to probability-like scores
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
        
        # Calculate metrics
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
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'auc_score': round(auc_score, 4),
            'confusion_matrix': cm,
            'model_type': model_type
        }
        
        return metrics
    
    def get_feature_importance(self, model_type=None):
        """Get feature importance for the specified model"""
        if model_type and model_type in self.feature_importance:
            return self.feature_importance[model_type]
        elif self.best_model and hasattr(self.best_model, 'feature_importances_'):
            return dict(zip(self.best_model.feature_names_in_, self.best_model.feature_importances_))
        else:
            return {}
    
    def compare_models(self, X_test, y_test):
        """Compare performance of all trained models"""
        comparison = {}
        
        for model_name, model in self.models.items():
            performance = self._evaluate_model(model, X_test, y_test, model_name)
            comparison[model_name] = performance
        
        return comparison