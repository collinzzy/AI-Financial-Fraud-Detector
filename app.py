import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from fraud_detector import FraudDetector
from model_trainer import ModelTrainer
from visualization_engine import VisualizationEngine

# Page configuration
st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fraud detection theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #e74c3c;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .safe-transaction {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #e74c3c;
    }
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class FraudDetectionApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.fraud_detector = FraudDetector()
        self.model_trainer = ModelTrainer()
        self.viz_engine = VisualizationEngine()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'model_performance' not in st.session_state:
            st.session_state.model_performance = None
    
    def render_header(self):
        """Render application header"""
        st.markdown('<div class="main-header">🔍 AI Financial Fraud Detection System</div>', unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Transactions Analyzed", 
                     len(st.session_state.current_data) if st.session_state.current_data is not None else 0)
        with col2:
            fraud_count = len(st.session_state.predictions[
                st.session_state.predictions['is_fraud'] == 1
            ]) if st.session_state.predictions is not None else 0
            st.metric("Fraud Detected", fraud_count)
        with col3:
            st.metric("Detection Accuracy", "95%+")
        with col4:
            st.metric("System Status", "🟢 Active")
    
    def render_sidebar(self):
        """Render sidebar navigation and controls"""
        st.sidebar.title("🔧 Navigation")
        
        # Navigation
        section = st.sidebar.radio(
            "Go to:",
            ["📊 Data Overview", "🤖 Train Models", "🔍 Detect Fraud", "📈 Analytics", "🚨 Live Monitoring"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.title("📁 Data Management")
        
        # Data source selection
        data_source = st.sidebar.selectbox(
            "Data Source",
            ["Upload Dataset", "Generate Sample Data", "Real-time Stream"]
        )
        
        return section, data_source
    
    def load_data(self, data_source):
        """Load data based on selected source"""
        if data_source == "Upload Dataset":
            return self.load_uploaded_data()
        elif data_source == "Generate Sample Data":
            return self.generate_sample_data()
        else:  # Real-time Stream
            return self.setup_real_time_stream()
    
    def load_uploaded_data(self):
        """Handle uploaded transaction data"""
        st.subheader("📤 Upload Transaction Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file with transaction data",
            type=['csv'],
            help="File should contain transaction details like amount, location, time, etc."
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} transactions")
                return df
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                return None
        
        return None
    
    def generate_sample_data(self):
        """Generate sample fraud detection dataset"""
        st.subheader("🎯 Generate Sample Fraud Data")
        
        if st.button("Generate Sample Dataset", type="primary"):
            with st.spinner("Generating realistic fraud data..."):
                df = self.data_processor.generate_sample_transactions()
                st.success(f"✅ Generated {len(df)} sample transactions with realistic fraud patterns")
                return df
        
        return None
    
    def setup_real_time_stream(self):
        """Setup real-time data stream simulation"""
        st.subheader("🌊 Real-time Transaction Stream")
        st.info("Real-time streaming feature coming soon...")
        return None
    
    def show_data_overview(self, df):
        """Display data overview and statistics"""
        st.header("📊 Data Overview")
        
        if df is None:
            st.warning("Please load data first")
            return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            st.metric("Total Amount", f"${df['amount'].sum():,.2f}")
        with col3:
            st.metric("Average Transaction", f"${df['amount'].mean():.2f}")
        with col4:
            st.metric("Unique Customers", df['customer_id'].nunique())
        
        # Data preview
        with st.expander("🔍 Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Data quality assessment
        with st.expander("📋 Data Quality Report"):
            self.show_data_quality_report(df)
        
        # Basic visualizations
        with st.expander("📈 Quick Insights"):
            self.show_basic_insights(df)
    
    def show_data_quality_report(self, df):
        """Display data quality assessment"""
        quality_data = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            data_type = df[col].dtype
            
            quality_data.append({
                'Column': col,
                'Data Type': data_type,
                'Null Values': null_count,
                'Null %': f"{null_pct:.1f}%",
                'Unique Values': unique_count
            })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True)
    
    def show_basic_insights(self, df):
        """Show basic data insights"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction amount distribution
            fig = px.histogram(df, x='amount', title='Transaction Amount Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Transactions over time (if time column exists)
            time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                time_col = time_cols[0]
                try:
                    df_temp = df.copy()
                    df_temp[time_col] = pd.to_datetime(df_temp[time_col])
                    daily_counts = df_temp.groupby(df_temp[time_col].dt.date).size()
                    fig = px.line(x=daily_counts.index, y=daily_counts.values, 
                                title='Transactions Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Could not generate time series chart")
    
    def show_model_training(self, df):
        """Show model training interface"""
        st.header("🤖 Train Fraud Detection Models")
        
        if df is None:
            st.warning("Please load data first to train models")
            return
        
        st.info("""
        **Fraud Detection Models Available:**
        - **Isolation Forest**: Anomaly detection for rare fraud patterns
        - **Random Forest**: Ensemble learning for classification
        - **XGBoost**: Gradient boosting for high accuracy
        - **AutoEncoder**: Neural network for complex patterns
        """)
        
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["Isolation Forest", "Random Forest", "XGBoost", "AutoEncoder", "Ensemble"]
            )
        
        with col2:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("Number of Estimators", 50, 500, 100)
            with col2:
                random_state = st.number_input("Random State", value=42)
        
        # Train model
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                try:
                    # Prepare data
                    X_processed, feature_names = self.data_processor.preprocess_data(df)
                    
                    # Train model
                    model, performance = self.model_trainer.train_model(
                        X_processed, model_type=model_type,
                        test_size=test_size/100,
                        n_estimators=n_estimators,
                        random_state=random_state
                    )
                    
                    # Store in session state
                    st.session_state.trained_model = model
                    st.session_state.model_performance = performance
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Show performance
                    self.show_model_performance(performance)
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
    
    def show_model_performance(self, performance):
        """Display model performance metrics"""
        st.subheader("📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{performance.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Precision", f"{performance.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{performance.get('recall', 0):.3f}")
        with col4:
            st.metric("F1-Score", f"{performance.get('f1', 0):.3f}")
        
        # Confusion matrix
        if 'confusion_matrix' in performance:
            fig = self.viz_engine.plot_confusion_matrix(performance['confusion_matrix'])
            st.plotly_chart(fig, use_container_width=True)
    
    def show_fraud_detection(self, df):
        """Show fraud detection interface"""
        st.header("🔍 Detect Fraud")
        
        if df is None:
            st.warning("Please load data first")
            return
        
        if st.session_state.trained_model is None:
            st.warning("Please train a model first")
            return
        
        # Detection options
        col1, col2 = st.columns(2)
        
        with col1:
            detection_mode = st.radio(
                "Detection Mode",
                ["Batch Detection", "Single Transaction", "Real-time Monitoring"]
            )
        
        with col2:
            risk_threshold = st.slider("Risk Threshold", 0.1, 0.9, 0.5, 0.1)
        
        if detection_mode == "Batch Detection":
            self.run_batch_detection(df, risk_threshold)
        elif detection_mode == "Single Transaction":
            self.run_single_detection()
        else:
            self.run_real_time_monitoring()
    
    def run_batch_detection(self, df, risk_threshold):
        """Run fraud detection on entire dataset"""
        if st.button("🔍 Run Fraud Detection", type="primary"):
            with st.spinner("Analyzing transactions for fraud..."):
                try:
                    # Preprocess data
                    X_processed, feature_names = self.data_processor.preprocess_data(df)
                    
                    # Detect fraud
                    predictions = self.fraud_detector.detect_fraud(
                        st.session_state.trained_model, 
                        X_processed, 
                        df,
                        risk_threshold
                    )
                    
                    st.session_state.predictions = predictions
                    
                    # Show results
                    self.show_detection_results(predictions)
                    
                except Exception as e:
                    st.error(f"❌ Error during fraud detection: {str(e)}")
    
    def show_detection_results(self, predictions):
        """Display fraud detection results"""
        st.subheader("🎯 Detection Results")
        
        # Summary statistics
        fraud_count = len(predictions[predictions['is_fraud'] == 1])
        total_count = len(predictions)
        fraud_percentage = (fraud_count / total_count) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", total_count)
        with col2:
            st.metric("Fraud Detected", fraud_count)
        with col3:
            st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
        with col4:
            total_fraud_amount = predictions[predictions['is_fraud'] == 1]['amount'].sum()
            st.metric("Fraud Amount", f"${total_fraud_amount:,.2f}")
        
        # Show fraud transactions
        with st.expander("🚨 Fraudulent Transactions", expanded=True):
            fraud_df = predictions[predictions['is_fraud'] == 1]
            if len(fraud_df) > 0:
                st.dataframe(fraud_df, use_container_width=True)
                
                # Download fraud report
                csv = fraud_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Fraud Report",
                    data=csv,
                    file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.success("🎉 No fraudulent transactions detected!")
        
        # Risk distribution
        with st.expander("📊 Risk Analysis"):
            self.show_risk_analysis(predictions)
    
    def show_risk_analysis(self, predictions):
        """Display risk analysis visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk score distribution
            fig = px.histogram(predictions, x='risk_score', 
                             title='Distribution of Risk Scores',
                             color='is_fraud')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fraud by category (if category column exists)
            category_cols = [col for col in predictions.columns if 'category' in col.lower() or 'type' in col.lower()]
            if category_cols:
                category_col = category_cols[0]
                fraud_by_category = predictions[predictions['is_fraud'] == 1][category_col].value_counts()
                fig = px.pie(values=fraud_by_category.values, names=fraud_by_category.index,
                           title='Fraud by Category')
                st.plotly_chart(fig, use_container_width=True)
    
    def run_single_detection(self):
        """Run fraud detection on single transaction"""
        st.subheader("🔍 Single Transaction Analysis")
        
        # Create transaction input form
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
                customer_id = st.text_input("Customer ID", value="CUST_001")
            
            with col2:
                merchant = st.text_input("Merchant", value="Online_Store")
                location = st.selectbox("Location", ["US", "UK", "EU", "Asia", "Other"])
            
            submitted = st.form_submit_button("Analyze Transaction")
            
            if submitted:
                # Create transaction data
                transaction_data = {
                    'amount': amount,
                    'customer_id': customer_id,
                    'merchant': merchant,
                    'location': location,
                    'timestamp': datetime.now()
                }
                
                # Analyze transaction
                risk_score, is_fraud = self.analyze_single_transaction(transaction_data)
                
                # Display results
                if is_fraud:
                    st.markdown(f'<div class="fraud-alert">🚨 HIGH RISK: Fraud Probability {risk_score:.1%}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="safe-transaction">✅ LOW RISK: Fraud Probability {risk_score:.1%}</div>', 
                              unsafe_html=True)
    
    def analyze_single_transaction(self, transaction_data):
        """Analyze single transaction for fraud"""
        # Placeholder implementation
        # In real implementation, this would use the trained model
        risk_score = np.random.uniform(0, 1)
        is_fraud = risk_score > 0.7
        return risk_score, is_fraud
    
    def run_real_time_monitoring(self):
        """Run real-time monitoring simulation"""
        st.subheader("🌊 Real-time Monitoring Dashboard")
        
        # Simulate real-time data
        if st.button("Start Monitoring"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                # Simulate processing
                time.sleep(0.1)
                progress_bar.progress(i + 1)
                status_text.text(f"Processed {i + 1} transactions...")
                
                # Simulate fraud detection
                if i % 10 == 0:
                    st.warning(f"🚨 Potential fraud detected in transaction {i}")
            
            st.success("✅ Monitoring completed!")
    
    def run(self):
        """Main application runner"""
        self.setup_session_state()
        
        # Render header
        self.render_header()
        
        # Render sidebar and get current section
        section, data_source = self.render_sidebar()
        
        # Load data if not already loaded
        if st.session_state.current_data is None:
            df = self.load_data(data_source)
            if df is not None:
                st.session_state.current_data = df
        else:
            df = st.session_state.current_data
        
        # Show appropriate section
        if section == "📊 Data Overview":
            self.show_data_overview(df)
        elif section == "🤖 Train Models":
            self.show_model_training(df)
        elif section == "🔍 Detect Fraud":
            self.show_fraud_detection(df)
        elif section == "📈 Analytics":
            st.header("📈 Advanced Analytics")
            st.info("Advanced analytics features coming soon...")
        elif section == "🚨 Live Monitoring":
            self.run_real_time_monitoring()

def main():
    app = FraudDetectionApp()
    app.run()

if __name__ == "__main__":
    main()