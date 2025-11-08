import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from plotly.offline import plot
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationEngine:
    def __init__(self):
        self.color_scheme = {
            'fraud': '#e74c3c',
            'legitimate': '#2ecc71',
            'high_risk': '#e74c3c',
            'medium_risk': '#f39c12',
            'low_risk': '#2ecc71'
        }
    
    def plot_fraud_distribution(self, df):
        """Plot distribution of fraud vs legitimate transactions"""
        fraud_count = df['is_fraud'].sum()
        legit_count = len(df) - fraud_count
        
        fig = px.pie(
            values=[legit_count, fraud_count],
            names=['Legitimate', 'Fraud'],
            title='Fraud vs Legitimate Transactions',
            color=['Legitimate', 'Fraud'],
            color_discrete_map={'Legitimate': self.color_scheme['legitimate'], 
                              'Fraud': self.color_scheme['fraud']}
        )
        
        return fig
    
    def plot_risk_score_distribution(self, results_df):
        """Plot distribution of risk scores"""
        fig = px.histogram(
            results_df, 
            x='risk_score',
            color='is_fraud',
            title='Distribution of Risk Scores',
            color_discrete_map={0: self.color_scheme['legitimate'], 1: self.color_scheme['fraud']},
            nbins=50
        )
        
        fig.update_layout(
            xaxis_title='Risk Score',
            yaxis_title='Number of Transactions',
            showlegend=True
        )
        
        return fig
    
    def plot_fraud_by_feature(self, results_df, feature):
        """Plot fraud distribution by specific feature"""
        fraud_by_feature = results_df[results_df['is_fraud'] == 1][feature].value_counts().head(10)
        
        fig = px.bar(
            x=fraud_by_feature.index,
            y=fraud_by_feature.values,
            title=f'Fraud Distribution by {feature.title()}',
            color=fraud_by_feature.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis_title=feature.title(),
            yaxis_title='Number of Fraud Cases',
            showlegend=False
        )
        
        return fig
    
    def plot_transaction_timeline(self, results_df):
        """Plot transaction timeline with fraud highlights"""
        if 'transaction_time' not in results_df.columns:
            return self._create_empty_plot("No time data available")
        
        # Group by hour and fraud status
        results_df['hour'] = pd.to_datetime(results_df['transaction_time']).dt.hour
        hourly_fraud = results_df.groupby(['hour', 'is_fraud']).size().reset_index(name='count')
        
        fig = px.line(
            hourly_fraud[hourly_fraud['is_fraud'] == 1],
            x='hour',
            y='count',
            title='Fraud Cases by Hour of Day',
            color_discrete_sequence=[self.color_scheme['fraud']]
        )
        
        fig.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Number of Fraud Cases',
            showlegend=False
        )
        
        return fig
    
    def plot_amount_distribution(self, results_df):
        """Plot transaction amount distribution by fraud status"""
        fig = px.box(
            results_df,
            x='is_fraud',
            y='amount',
            color='is_fraud',
            title='Transaction Amount Distribution by Fraud Status',
            color_discrete_map={0: self.color_scheme['legitimate'], 1: self.color_scheme['fraud']}
        )
        
        fig.update_layout(
            xaxis_title='Is Fraud',
            yaxis_title='Transaction Amount ($)',
            showlegend=False
        )
        
        return fig
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale='Blues',
            title='Confusion Matrix',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Legitimate', 'Fraud']),
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
        )
        
        return fig
    
    def plot_model_performance(self, performance_metrics):
        """Plot model performance comparison"""
        models = list(performance_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [performance_metrics[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(name=metric.title(), x=models, y=values))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group'
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance, top_n=15):
        """Plot feature importance"""
        if not feature_importance:
            return self._create_empty_plot("No feature importance data")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importance = zip(*sorted_features)
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title=f'Top {top_n} Most Important Features for Fraud Detection',
            color=importance,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title='Feature Importance',
            yaxis_title='Features',
            showlegend=False
        )
        
        return fig
    
    def plot_geographic_fraud(self, results_df):
        """Plot fraud distribution by geography"""
        if 'location' not in results_df.columns:
            return self._create_empty_plot("No location data available")
        
        fraud_by_location = results_df[results_df['is_fraud'] == 1]['location'].value_counts()
        
        fig = px.choropleth(
            locations=fraud_by_location.index,
            locationmode='country names',
            color=fraud_by_location.values,
            title='Fraud Distribution by Location',
            color_continuous_scale='Reds'
        )
        
        return fig
    
    def plot_real_time_monitoring(self, monitoring_data):
        """Plot real-time monitoring dashboard"""
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=['Transactions per Minute', 'Fraud Alerts', 'Risk Score Distribution', 'Amount Trends'],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Add sample data for demonstration
        fig.add_trace(go.Scatter(x=monitoring_data.get('time', []), 
                               y=monitoring_data.get('transactions', []),
                               name='Transactions'), row=1, col=1)
        
        fig.add_trace(go.Bar(x=monitoring_data.get('alerts_time', []), 
                           y=monitoring_data.get('alerts', []),
                           name='Fraud Alerts'), row=1, col=2)
        
        fig.add_trace(go.Histogram(x=monitoring_data.get('risk_scores', []),
                                 name='Risk Scores'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=monitoring_data.get('amount_time', []), 
                               y=monitoring_data.get('amounts', []),
                               name='Transaction Amounts'), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Real-time Fraud Monitoring Dashboard")
        
        return fig
    
    def _create_empty_plot(self, message):
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def create_comprehensive_dashboard(self, results_df, performance_metrics, feature_importance):
        """Create a comprehensive fraud detection dashboard"""
        # Create subplots
        fig = sp.make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Fraud Distribution', 'Risk Score Distribution', 
                'Transaction Amounts', 'Fraud by Hour',
                'Feature Importance', 'Performance Metrics'
            ],
            specs=[
                [{"type": "pie"}, {"type": "histogram"}],
                [{"type": "box"}, {"type": "line"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Fraud distribution
        fraud_count = results_df['is_fraud'].sum()
        legit_count = len(results_df) - fraud_count
        fig.add_trace(go.Pie(labels=['Legitimate', 'Fraud'], 
                           values=[legit_count, fraud_count],
                           marker_colors=[self.color_scheme['legitimate'], self.color_scheme['fraud']]),
                    row=1, col=1)
        
        # Risk score distribution
        fig.add_trace(go.Histogram(x=results_df['risk_score'], name='Risk Scores'),
                    row=1, col=2)
        
        # Transaction amounts
        fig.add_trace(go.Box(y=results_df[results_df['is_fraud']==0]['amount'], name='Legitimate'),
                    row=2, col=1)
        fig.add_trace(go.Box(y=results_df[results_df['is_fraud']==1]['amount'], name='Fraud'),
                    row=2, col=1)
        
        # Fraud by hour
        if 'hour_of_day' in results_df.columns:
            fraud_by_hour = results_df[results_df['is_fraud']==1]['hour_of_day'].value_counts().sort_index()
            fig.add_trace(go.Scatter(x=fraud_by_hour.index, y=fraud_by_hour.values, name='Fraud by Hour'),
                        row=2, col=2)
        
        # Feature importance
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importance = zip(*top_features)
            fig.add_trace(go.Bar(x=importance, y=features, orientation='h', name='Feature Importance'),
                        row=3, col=1)
        
        # Performance metrics
        if performance_metrics:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            values = [performance_metrics.get(metric, 0) for metric in metrics]
            fig.add_trace(go.Bar(x=metrics, y=values, name='Performance'),
                        row=3, col=2)
        
        fig.update_layout(height=900, title_text="Comprehensive Fraud Detection Dashboard")
        
        return fig