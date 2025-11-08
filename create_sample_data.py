# create_sample_data.py
from data_processor import DataProcessor
import pandas as pd

def create_sample_data():
    processor = DataProcessor()
    df = processor.generate_sample_transactions(5000)
    df.to_csv('data/sample_fraud_data.csv', index=False)
    print("Sample data created with 5000 transactions")

if __name__ == "__main__":
    create_sample_data()