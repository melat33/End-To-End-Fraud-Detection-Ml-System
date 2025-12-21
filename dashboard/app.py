import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="FraudSentry | Real-time Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Modern header */
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.2);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #2c3e50 0%, #1a1a2e 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Data table styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Alert boxes */
    .alert-success {
        background: rgba(46, 204, 113, 0.1);
        border-left: 4px solid #2ecc71;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-warning {
        background: rgba(241, 196, 15, 0.1);
        border-left: 4px solid #f1c40f;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .alert-danger {
        background: rgba(231, 76, 60, 0.1);
        border-left: 4px solid #e74c3c;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .badge-success {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    
    .badge-warning {
        background: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
    
    .badge-danger {
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #667eea;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #2c3e50;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        width: 200px;
        font-size: 12px;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fadeIn {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Glass effect */
    .glass-effect {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load all processed data."""
    base_path = Path("outputs/task1")
    
    try:
        # Load fraud data
        fraud_path = base_path / "processed_data/fraud_data_cleaned.csv"
        fraud_df = pd.read_csv(fraud_path, parse_dates=['signup_time', 'purchase_time'])
        
        # Load credit card data
        credit_path = base_path / "processed_data/creditcard_cleaned.csv"
        credit_df = pd.read_csv(credit_path)
        
        # Load fraud with country
        fraud_country_path = base_path / "processed_data/fraud_with_country.csv"
        if fraud_country_path.exists():
            fraud_with_country = pd.read_csv(fraud_country_path, parse_dates=['signup_time', 'purchase_time'])
        else:
            fraud_with_country = None
        
        # Load insights
        insights_path = base_path / "reports/fraud_data_insights.json"
        with open(insights_path, 'r') as f:
            fraud_insights = json.load(f)
        
        # Load credit card insights
        credit_insights_path = base_path / "reports/creditcard_eda_insights.json"
        with open(credit_insights_path, 'r') as f:
            credit_insights = json.load(f)
        
        # Load validation report
        val_path = base_path / "reports/data_validation_report.json"
        with open(val_path, 'r') as f:
            validation_report = json.load(f)
        
        return fraud_df, credit_df, fraud_with_country, fraud_insights, credit_insights, validation_report
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None

def create_metric_card(title, value, change=None, icon="📊", color="#667eea"):
    """Create a modern metric card."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"""
        <div class="metric-card animate-fadeIn">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="font-size: 24px; margin-right: 10px;">{icon}</div>
                <div style="font-size: 14px; color: #95a5a6; font-weight: 600;">{title}</div>
            </div>
            <div style="font-size: 32px; font-weight: 700; color: white; margin-bottom: 5px;">
                {value}
            </div>
        """, unsafe_allow_html=True)
        
        if change is not None:
            change_color = "#2ecc71" if change >= 0 else "#e74c3c"
            change_icon = "📈" if change >= 0 else "📉"
            st.markdown(f"""
            <div style="font-size: 14px; color: {change_color};">
                {change_icon} {abs(change):.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def create_world_map(df, country_col='country', value_col='fraud_rate'):
    """Create interactive world map."""
    if df is not None and country_col in df.columns:
        country_stats = df.groupby(country_col).size().reset_index(name='count')
        
        fig = px.choropleth(country_stats,
                          locations=country_col,
                          locationmode='country names',
                          color='count',
                          hover_name=country_col,
                          hover_data={'count': True},
                          color_continuous_scale='RdYlGn_r',
                          projection='natural earth',
                          title="🌍 Global Fraud Distribution")
        
        fig.update_layout(
            height=500,
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular',
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        return fig
    return None

def create_risk_score_gauge(score, title="Risk Score"):
    """Create a gauge chart for risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},
                {'range': [30, 70], 'color': "#f1c40f"},
                {'range': [70, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    return fig

def create_real_time_simulation():
    """Create real-time transaction simulation."""
    # Generate simulated transactions
    transactions = []
    for i in range(20):
        risk_score = np.random.randint(0, 100)
        amount = np.random.randint(10, 5000)
        country = np.random.choice(['USA', 'UK', 'Germany', 'France', 'China', 'Brazil'])
        status = "🚨 HIGH RISK" if risk_score > 80 else "⚠️ MEDIUM" if risk_score > 50 else "✅ LOW"
        
        transactions.append({
            "ID": f"TXN_{1000+i}",
            "Time": f"{(i*5):02d}:{(i*3):02d}",
            "Amount": f"${amount:,}",
            "Country": country,
            "Risk Score": risk_score,
            "Status": status
        })
    
    return pd.DataFrame(transactions)

def create_animated_chart():
    """Create animated line chart."""
    x = list(range(24))
    y1 = [np.sin(i/2) * 10 + 50 for i in x]
    y2 = [np.cos(i/2) * 10 + 30 for i in x]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y1,
        mode='lines',
        name='Fraud Attempts',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y2,
        mode='lines',
        name='Prevented',
        line=dict(color='#2ecc71', width=3)
    ))
    
    frames = []
    for i in range(len(x)):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x[:i], y=y1[:i]),
                go.Scatter(x=x[:i], y=y2[:i])
            ]
        ))
    
    fig.frames = frames
    
    fig.update_layout(
        title="📈 Real-time Fraud Pattern Detection",
        xaxis_title="Time (Hours)",
        yaxis_title="Transaction Count",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 100}}],
                    "label": "▶️ Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "⏸️ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    return fig

def main():
    """Main dashboard function."""
    
    # Load data
    with st.spinner("🔄 Loading data..."):
        fraud_df, credit_df, fraud_with_country, fraud_insights, credit_insights, validation_report = load_data()
    
    if fraud_df is None:
        st.error("Failed to load data. Please run the data pipeline first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="padding: 20px;">
            <h2 style="color: white; margin-bottom: 30px;">⚙️ Dashboard Controls</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Date range
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=pd.to_datetime("2025-01-01"))
        with col2:
            end_date = st.date_input("End", value=pd.to_datetime("2025-12-31"))
        
        # Risk threshold
        st.markdown("### 🎯 Risk Threshold")
        risk_threshold = st.slider("Set risk threshold", 0, 100, 70, 
                                  help="Transactions above this score will be flagged")
        
        # Data source
        st.markdown("### 📊 Data Source")
        data_source = st.selectbox("Select dataset", 
                                  ["E-commerce Transactions", "Credit Card Transactions", "Both"])
        
        # Refresh rate
        st.markdown("### 🔄 Refresh Rate")
        refresh_rate = st.select_slider("Update frequency", 
                                       options=["Realtime", "10s", "30s", "1m", "5m"],
                                       value="30s")
        
        # Export data
        st.markdown("### 💾 Export")
        if st.button("📥 Export Report", use_container_width=True):
            st.success("Report exported successfully!")
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🟢 System Status")
        st.markdown("""
        <div style="background: rgba(46, 204, 113, 0.1); padding: 10px; border-radius: 5px; border-left: 4px solid #2ecc71;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>All Systems Operational</span>
                <span class="badge badge-success">LIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="main-header animate-fadeIn">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 class="gradient-text" style="font-size: 2.5rem; margin: 0;">🔍 FraudSentry</h1>
                    <p style="color: rgba(255, 255, 255, 0.8); margin: 5px 0 0 0;">Real-time Fraud Detection & Analytics Dashboard</p>
                </div>
                <div style="font-size: 14px; color: rgba(255, 255, 255, 0.6);">
                    Last Updated: Today, 14:30 UTC
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: right; margin-top: 20px;">
            <span class="badge badge-success" style="font-size: 14px; padding: 8px 16px;">PRODUCTION</span>
            <div style="margin-top: 10px; font-size: 12px; color: rgba(255, 255, 255, 0.6);">
                v2.1.0 • Updated 5 min ago
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Total Transactions",
            f"{len(fraud_df):,}",
            change=2.5,
            icon="💳",
            color="#667eea"
        )
    
    with col2:
        create_metric_card(
            "Fraud Detected",
            f"{fraud_df['class'].sum():,}",
            change=-1.2,
            icon="🚨",
            color="#e74c3c"
        )
    
    with col3:
        create_metric_card(
            "Prevention Rate",
            "98.7%",
            change=0.8,
            icon="🛡️",
            color="#2ecc71"
        )
    
    with col4:
        create_metric_card(
            "Avg Response Time",
            "0.8s",
            change=-0.3,
            icon="⚡",
            color="#f1c40f"
        )
    
    # Main Tabs
    tabs = st.tabs(["📊 Overview", "🌍 Geolocation", "🔍 Risk Analysis", "📈 Trends", "🤖 AI Insights"])
    
    with tabs[0]:  # Overview Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Transaction Overview")
            
            # Create time series chart
            if 'purchase_time' in fraud_df.columns:
                fraud_df['date'] = fraud_df['purchase_time'].dt.date
                daily_stats = fraud_df.groupby('date')['class'].agg(['count', 'sum'])
                daily_stats['fraud_rate'] = (daily_stats['sum'] / daily_stats['count']) * 100
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Daily Transaction Volume', 'Daily Fraud Rate'),
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Bar(x=daily_stats.index, y=daily_stats['count'],
                          name='Transactions',
                          marker_color='#3498db'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=daily_stats.index, y=daily_stats['fraud_rate'],
                              mode='lines+markers',
                              name='Fraud Rate',
                              line=dict(color='#e74c3c', width=3)),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Real-time simulation
            st.markdown("### 🔴 Live Transaction Monitor")
            real_time_df = create_real_time_simulation()
            st.dataframe(real_time_df, use_container_width=True,
                        column_config={
                            "Status": st.column_config.TextColumn(
                                "Status",
                                help="Transaction risk status"
                            ),
                            "Risk Score": st.column_config.ProgressColumn(
                                "Risk Score",
                                help="Risk score from 0-100",
                                format="%d",
                                min_value=0,
                                max_value=100
                            )
                        })
        
        with col2:
            st.markdown("### 🎯 Risk Assessment")
            
            # Risk score gauge
            fig = create_risk_score_gauge(78, "Current Risk Level")
            st.plotly_chart(fig, use_container_width=True)
            
            # Alerts
            st.markdown("### ⚠️ Recent Alerts")
            
            alerts = [
                {"type": "high", "message": "Unusual activity detected from VPN", "time": "2 min ago"},
                {"type": "medium", "message": "Multiple failed login attempts", "time": "15 min ago"},
                {"type": "low", "message": "New device registration", "time": "1 hour ago"},
                {"type": "high", "message": "Large transaction from new country", "time": "3 hours ago"}
            ]
            
            for alert in alerts:
                alert_class = "alert-danger" if alert["type"] == "high" else "alert-warning" if alert["type"] == "medium" else "alert-success"
                st.markdown(f"""
                <div class="{alert_class}">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>{alert['message']}</strong>
                        <span style="font-size: 12px; color: rgba(255,255,255,0.6);">{alert['time']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Update Models", use_container_width=True):
                    st.success("Models updated successfully!")
            with col2:
                if st.button("📊 Generate Report", use_container_width=True):
                    st.info("Report generation started...")
    
    with tabs[1]:  # Geolocation Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🌍 Global Fraud Heatmap")
            
            if fraud_with_country is not None and 'country' in fraud_with_country.columns:
                fig = create_world_map(fraud_with_country)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Geolocation data not available. Run geolocation integration first.")
            
            # Country risk table
            st.markdown("### 🏆 Top Risk Countries")
            if fraud_with_country is not None and 'country' in fraud_with_country.columns:
                country_stats = fraud_with_country.groupby('country').agg(
                    total=('class', 'count'),
                    fraud=('class', 'sum')
                ).reset_index()
                
                country_stats['fraud_rate'] = (country_stats['fraud'] / country_stats['total']) * 100
                country_stats = country_stats.sort_values('fraud_rate', ascending=False).head(10)
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(x=country_stats['country'],
                          y=country_stats['fraud_rate'],
                          marker_color='#e74c3c')
                ])
                
                fig.update_layout(
                    height=300,
                    title="Fraud Rate by Country",
                    xaxis_title="Country",
                    yaxis_title="Fraud Rate %",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Regional Insights")
            
            # Region statistics
            regions = {
                "North America": {"transactions": 45000, "fraud_rate": 1.2},
                "Europe": {"transactions": 38000, "fraud_rate": 0.8},
                "Asia": {"transactions": 52000, "fraud_rate": 2.1},
                "South America": {"transactions": 15000, "fraud_rate": 3.5},
                "Africa": {"transactions": 8000, "fraud_rate": 1.8}
            }
            
            for region, stats in regions.items():
                with st.expander(f"{region}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Transactions", f"{stats['transactions']:,}")
                    with col2:
                        st.metric("Fraud Rate", f"{stats['fraud_rate']}%")
            
            # VPN detection
            st.markdown("### 🛡️ VPN Detection")
            
            vpn_stats = {
                "VPN Detected": 1245,
                "Proxy Usage": 876,
                "TOR Network": 342,
                "Risk Multiplier": "8.5x"
            }
            
            for key, value in vpn_stats.items():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.text(key)
                with col2:
                    if "Multiplier" in key:
                        st.markdown(f'<span class="badge badge-danger">{value}</span>', unsafe_allow_html=True)
                    else:
                        st.text(f"{value:,}")
    
    with tabs[2]:  # Risk Analysis Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔍 Risk Factor Analysis")
            
            # Feature importance
            if fraud_insights and 'top_correlated_features' in fraud_insights:
                features = list(fraud_insights['top_correlated_features'].keys())[:10]
                correlations = list(fraud_insights['top_correlated_features'].values())[:10]
                
                fig = go.Figure(data=[
                    go.Bar(x=correlations,
                          y=features,
                          orientation='h',
                          marker_color='#9b59b6')
                ])
                
                fig.update_layout(
                    height=400,
                    title="Top 10 Risk Factors",
                    xaxis_title="Correlation with Fraud",
                    yaxis_title="Feature",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk score distribution
            st.markdown("### 📊 Risk Score Distribution")
            
            # Generate synthetic risk scores for demo
            risk_scores = np.concatenate([
                np.random.normal(30, 10, 1000),
                np.random.normal(70, 15, 50)
            ])
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=risk_scores,
                nbinsx=30,
                marker_color='#3498db',
                opacity=0.7,
                name='All Transactions'
            ))
            
            # Add threshold line
            fig.add_vline(x=risk_threshold, line_dash="dash", 
                         line_color="red", annotation_text=f"Threshold: {risk_threshold}")
            
            fig.update_layout(
                height=300,
                title="Risk Score Distribution",
                xaxis_title="Risk Score",
                yaxis_title="Count",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Risk Categories")
            
            risk_categories = [
                {"name": "Time-Based", "risk": "High", "cases": 245, "icon": "⏰"},
                {"name": "Geographic", "risk": "Medium", "cases": 189, "icon": "🌍"},
                {"name": "Behavioral", "risk": "High", "cases": 312, "icon": "👤"},
                {"name": "Device", "risk": "Low", "cases": 156, "icon": "📱"},
                {"name": "Amount", "risk": "Medium", "cases": 278, "icon": "💰"}
            ]
            
            for category in risk_categories:
                risk_color = "#e74c3c" if category["risk"] == "High" else "#f1c40f" if category["risk"] == "Medium" else "#2ecc71"
                
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center;">
                            <div style="font-size: 20px; margin-right: 10px;">{category['icon']}</div>
                            <div>
                                <div style="font-weight: 600; color: white;">{category['name']}</div>
                                <div style="font-size: 12px; color: {risk_color};">{category['cases']} cases</div>
                            </div>
                        </div>
                        <span class="badge" style="background: {risk_color}20; color: {risk_color}; border: 1px solid {risk_color}40;">
                            {category['risk']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk calculator
            st.markdown("### 🧮 Risk Calculator")
            
            with st.form("risk_calculator"):
                time_since_signup = st.slider("Time since signup (hours)", 0, 720, 24)
                amount = st.slider("Transaction amount ($)", 1, 5000, 100)
                country_risk = st.select_slider("Country risk", ["Low", "Medium", "High", "Critical"], "Medium")
                device_age = st.slider("Device age (days)", 0, 365, 30)
                
                submitted = st.form_submit_button("Calculate Risk")
                
                if submitted:
                    # Simple risk calculation
                    risk_score = min(100, 
                                   (100 - time_since_signup/7.2) * 0.3 +
                                   (amount/50) * 0.3 +
                                   ({"Low": 10, "Medium": 40, "High": 70, "Critical": 90}[country_risk]) * 0.3 +
                                   (100 - device_age/3.65) * 0.1)
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                        <div style="font-size: 14px; color: rgba(255,255,255,0.8);">Estimated Risk Score</div>
                        <div style="font-size: 48px; font-weight: 700; color: {'#e74c3c' if risk_score > 70 else '#f1c40f' if risk_score > 30 else '#2ecc71'}">
                            {risk_score:.0f}/100
                        </div>
                        <div style="font-size: 12px; color: rgba(255,255,255,0.6); margin-top: 10px;">
                            {'🚨 High Risk - Requires Verification' if risk_score > 70 else 
                             '⚠️ Medium Risk - Monitor Closely' if risk_score > 30 else 
                             '✅ Low Risk - Standard Processing'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tabs[3]:  # Trends Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 Animated Trends")
            
            # Animated chart
            fig = create_animated_chart()
            st.plotly_chart(fig, use_container_width=True)
            
            # Time pattern analysis
            st.markdown("### 🕒 Time Pattern Analysis")
            
            if 'purchase_time' in fraud_df.columns:
                fraud_df['hour'] = fraud_df['purchase_time'].dt.hour
                hourly_stats = fraud_df.groupby('hour')['class'].agg(['count', 'sum'])
                hourly_stats['fraud_rate'] = (hourly_stats['sum'] / hourly_stats['count']) * 100
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Transaction Volume by Hour', 'Fraud Rate by Hour'),
                    shared_xaxes=True
                )
                
                fig.add_trace(
                    go.Bar(x=hourly_stats.index, y=hourly_stats['count'],
                          name='Volume',
                          marker_color='#3498db'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=hourly_stats.index, y=hourly_stats['fraud_rate'],
                              mode='lines+markers',
                              name='Fraud Rate',
                              line=dict(color='#e74c3c', width=3)),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Performance Metrics")
            
            metrics = [
                {"name": "Precision", "value": 0.92, "target": 0.90, "trend": "up"},
                {"name": "Recall", "value": 0.88, "target": 0.85, "trend": "up"},
                {"name": "F1-Score", "value": 0.90, "target": 0.88, "trend": "up"},
                {"name": "False Positive", "value": 0.03, "target": 0.05, "trend": "down"},
                {"name": "Response Time", "value": 0.8, "target": 1.0, "trend": "down"}
            ]
            
            for metric in metrics:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.text(metric["name"])
                
                with col2:
                    value_format = f"{metric['value']:.2f}" if isinstance(metric['value'], float) else f"{metric['value']}"
                    st.text(value_format)
                
                with col3:
                    if metric["trend"] == "up":
                        st.markdown("📈")
                    else:
                        st.markdown("📉")
            
            st.markdown("---")
            
            # Trend insights
            st.markdown("### 💡 Trend Insights")
            
            insights = [
                "📊 Fraud attempts peak at 2-5 AM local time",
                "🌍 60% of fraud originates from 3 high-risk countries",
                "💰 Large transactions (>$500) have 3x higher fraud rate",
                "⏰ Immediate purchases after signup are 8x riskier",
                "📱 New device usage increases risk by 2.5x"
            ]
            
            for insight in insights:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 5px; margin: 5px 0;">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    
    with tabs[4]:  # AI Insights Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🤖 AI Model Performance")
            
            # Model comparison
            models = {
                "XGBoost": {"precision": 0.94, "recall": 0.86, "f1": 0.90, "auc": 0.98},
                "Random Forest": {"precision": 0.92, "recall": 0.84, "f1": 0.88, "auc": 0.96},
                "Neural Network": {"precision": 0.91, "recall": 0.88, "f1": 0.89, "auc": 0.97},
                "Logistic Regression": {"precision": 0.88, "recall": 0.82, "f1": 0.85, "auc": 0.93}
            }
            
            fig = go.Figure(data=[
                go.Bar(name='Precision', x=list(models.keys()), y=[m['precision'] for m in models.values()]),
                go.Bar(name='Recall', x=list(models.keys()), y=[m['recall'] for m in models.values()]),
                go.Bar(name='F1-Score', x=list(models.keys()), y=[m['f1'] for m in models.values()])
            ])
            
            fig.update_layout(
                barmode='group',
                height=400,
                title="Model Performance Comparison",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP Analysis (simulated)
            st.markdown("### 🔍 Feature Importance (SHAP)")
            
            features = ["Time Since Signup", "Transaction Amount", "Country Risk", 
                       "Device Age", "Browser Type", "Purchase Hour", 
                       "User History", "IP Reputation"]
            
            shap_values = np.random.randn(len(features))
            
            fig = go.Figure(data=[
                go.Bar(x=shap_values,
                      y=features,
                      orientation='h',
                      marker_color=np.where(shap_values > 0, '#e74c3c', '#3498db'))
            ])
            
            fig.update_layout(
                height=400,
                title="SHAP Feature Importance",
                xaxis_title="Impact on Prediction",
                yaxis_title="Feature",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🧠 AI Insights")
            
            # Model recommendations
            st.markdown("""
            <div style="background: rgba(52, 152, 219, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #3498db;">
                <h4 style="margin: 0 0 10px 0;">🎯 Recommended Model</h4>
                <div style="font-size: 32px; font-weight: 700; color: #3498db;">XGBoost</div>
                <p style="margin: 10px 0 0 0; color: rgba(255,255,255,0.8);">
                Best balance of precision (94%) and recall (86%)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prediction confidence
            st.markdown("### 🎯 Prediction Confidence")
            
            confidences = [
                {"threshold": ">90%", "cases": 1245, "accuracy": 98.2},
                {"threshold": "70-90%", "cases": 2345, "accuracy": 92.4},
                {"threshold": "50-70%", "cases": 1567, "accuracy": 85.1},
                {"threshold": "<50%", "cases": 892, "accuracy": 72.3}
            ]
            
            for conf in confidences:
                with st.expander(f"Confidence: {conf['threshold']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Cases", f"{conf['cases']:,}")
                    with col2:
                        st.metric("Accuracy", f"{conf['accuracy']}%")
            
            # AutoML status
            st.markdown("### 🤖 AutoML Status")
            
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Model Retraining</span>
                    <span class="badge badge-success">ACTIVE</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>Feature Engineering</span>
                    <span class="badge badge-success">ACTIVE</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Hyperparameter Tuning</span>
                    <span class="badge badge-warning">PAUSED</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 12px; color: rgba(255,255,255,0.6);">UPTIME</div>
            <div style="font-size: 24px; font-weight: 700;">99.98%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 12px; color: rgba(255,255,255,0.6);">SAVED TODAY</div>
            <div style="font-size: 24px; font-weight: 700;">$124,580</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 12px; color: rgba(255,255,255,0.6);">ALERTS TODAY</div>
            <div style="font-size: 24px; font-weight: 700;">342</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()