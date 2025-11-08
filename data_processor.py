import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def generate_sample_transactions(self, n_transactions=10000, fraud_rate=0.03):
        """Generate realistic sample transaction data with fraud patterns"""
        np.random.seed(42)
        
        # Base transaction data
        data = {
            'transaction_id': [f'TXN_{i:06d}' for i in range(n_transactions)],
            'customer_id': [f'CUST_{np.random.randint(1000, 9999)}' for _ in range(n_transactions)],
            'amount': np.random.exponential(100, n_transactions).round(2),
            'merchant': np.random.choice([
                'Amazon', 'Walmart', 'Target', 'Starbucks', 'Apple_Store',
                'Gas_Station', 'Restaurant', 'Hotel', 'Airlines', 'Online_Retailer'
            ], n_transactions),
            'category': np.random.choice([
                'Shopping', 'Food', 'Travel', 'Entertainment', 'Utilities', 'Other'
            ], n_transactions),
            'location': np.random.choice(['US', 'UK', 'EU', 'Asia', 'Other'], n_transactions),
            'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_transactions),
            'ip_address': [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" 
                          for _ in range(n_transactions)],
            'transaction_time': [datetime.now() - timedelta(hours=random.randint(0, 8760)) 
                               for _ in range(n_transactions)]
        }
        
        df = pd.DataFrame(data)
        
        # Add features that help detect fraud
        df['hour_of_day'] = df['transaction_time'].dt.hour
        df['day_of_week'] = df['transaction_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Generate fraud labels with realistic patterns
        df['is_fraud'] = 0
        
        # Fraud pattern 1: Unusually large amounts
        large_amount_mask = df['amount'] > df['amount'].quantile(0.95)
        df.loc[large_amount_mask, 'is_fraud'] = np.random.choice([0, 1], size=large_amount_mask.sum(), p=[0.7, 0.3])
        
        # Fraud pattern 2: Unusual hours (2 AM - 5 AM)
        unusual_hours_mask = (df['hour_of_day'] >= 2) & (df['hour_of_day'] <= 5)
        df.loc[unusual_hours_mask, 'is_fraud'] = np.random.choice([0, 1], size=unusual_hours_mask.sum(), p=[0.8, 0.2])
        
        # Fraud pattern 3: High-frequency transactions from same customer
        customer_tx_counts = df['customer_id'].value_counts()
        high_freq_customers = customer_tx_counts[customer_tx_counts > 10].index
        high_freq_mask = df['customer_id'].isin(high_freq_customers)
        df.loc[high_freq_mask, 'is_fraud'] = np.random.choice([0, 1], size=high_freq_mask.sum(), p=[0.9, 0.1])
        
        # Fraud pattern 4: International transactions from domestic customers
        intl_mask = (df['location'] != 'US') & (df['customer_id'].str.contains('CUST_1'))
        df.loc[intl_mask, 'is_fraud'] = np.random.choice([0, 1], size=intl_mask.sum(), p=[0.6, 0.4])
        
        # Ensure fraud rate is approximately correct
        current_fraud_rate = df['is_fraud'].mean()
        if current_fraud_rate < fraud_rate:
            # Add more fraud cases
            additional_fraud_needed = int((fraud_rate - current_fraud_rate) * n_transactions)
            non_fraud_indices = df[df['is_fraud'] == 0].index
            fraud_indices = np.random.choice(non_fraud_indices, additional_fraud_needed, replace=False)
            df.loc[fraud_indices, 'is_fraud'] = 1
        
        # Add some noise and realistic features
        df['previous_chargebacks'] = np.random.poisson(0.1, n_transactions)
        df['customer_age_days'] = np.random.exponential(365, n_transactions).astype(int)
        df['transaction_count_24h'] = np.random.poisson(2, n_transactions)
        
        # Fraud transactions tend to have higher previous chargebacks
        df.loc[df['is_fraud'] == 1, 'previous_chargebacks'] = np.random.poisson(2, df['is_fraud'].sum())
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess transaction data for machine learning"""
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Feature engineering
        df_processed = self._create_features(df_processed)
        
        # Select features for modeling
        feature_columns = [
            'amount', 'hour_of_day', 'day_of_week', 'is_weekend',
            'previous_chargebacks', 'customer_age_days', 'transaction_count_24h',
            'amount_to_avg_ratio', 'time_since_last_tx'
        ]
        
        # Add encoded categorical features
        categorical_columns = ['merchant', 'category', 'location', 'device_type']
        for col in categorical_columns:
            if col in df_processed.columns:
                encoded_col = f'{col}_encoded'
                df_processed[encoded_col] = self._encode_column(df_processed[col])
                feature_columns.append(encoded_col)
        
        # Handle missing values
        X = df_processed[feature_columns].fillna(0)
        
        # Scale numerical features
        numerical_features = ['amount', 'previous_chargebacks', 'customer_age_days', 
                            'transaction_count_24h', 'amount_to_avg_ratio', 'time_since_last_tx']
        
        for feature in numerical_features:
            if feature in X.columns:
                X[feature] = self.scaler.fit_transform(X[[feature]])
        
        return X, feature_columns
    
    def _create_features(self, df):
        """Create additional features for fraud detection"""
        df_temp = df.copy()
        
        # Transaction amount relative to customer's average
        customer_avg = df_temp.groupby('customer_id')['amount'].transform('mean')
        df_temp['amount_to_avg_ratio'] = df_temp['amount'] / (customer_avg + 1e-6)
        
        # Time since last transaction (simulated)
        df_temp = df_temp.sort_values('transaction_time')
        df_temp['time_since_last_tx'] = df_temp.groupby('customer_id')['transaction_time'].diff().dt.total_seconds().fillna(3600)
        
        # Binary features
        df_temp['is_large_amount'] = (df_temp['amount'] > df_temp['amount'].quantile(0.95)).astype(int)
        df_temp['is_high_frequency'] = (df_temp['transaction_count_24h'] > 10).astype(int)
        
        return df_temp
    
    def _encode_column(self, series):
        """Encode categorical column"""
        if series.name not in self.label_encoders:
            self.label_encoders[series.name] = LabelEncoder()
        
        return self.label_encoders[series.name].fit_transform(series.astype(str))
    
    def validate_transaction_data(self, df):
        """Validate transaction data for required columns and quality"""
        required_columns = ['amount', 'customer_id', 'transaction_time']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(df['amount']):
            raise ValueError("Amount column must be numeric")
        
        # Check for negative amounts
        if (df['amount'] < 0).any():
            raise ValueError("Negative amounts found in transaction data")
        
        return True