# task1-data-preprocessing/src/Data_Anlysis_Processing/visualizer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    Comprehensive visualization module for fraud detection analysis.
    Generates static and interactive visualizations.
    """
    
    def __init__(self, config_path: str = "config/config_data_analysis.yaml"):
        """Initialize with configuration."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config['visualization']
        self.paths = self.config['paths']
        
        # Set styling
        plt.style.use(self.viz_config['style'])
        sns.set_palette(self.viz_config['palette'])
        plt.rcParams['figure.figsize'] = tuple(self.viz_config['figure_size'])
        plt.rcParams['savefig.dpi'] = self.viz_config['dpi']
    
    def generate_all_visualizations(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Generate all visualizations for all datasets.
        
        Args:
            data_dict: Dictionary containing all datasets
        """
        logger.info("Generating all visualizations...")
        
        # Create output directories
        vis_dir = Path(self.paths['visualizations'])
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate fraud data visualizations
        if 'fraud_data' in data_dict and data_dict['fraud_data'] is not None:
            self.generate_fraud_data_visualizations(data_dict['fraud_data'], vis_dir)
        
        # Generate credit card data visualizations
        if 'creditcard_data' in data_dict and data_dict['creditcard_data'] is not None:
            self.generate_creditcard_visualizations(data_dict['creditcard_data'], vis_dir)
        
        # Generate comparative visualizations
        if 'fraud_data' in data_dict and 'creditcard_data' in data_dict:
            if data_dict['fraud_data'] is not None and data_dict['creditcard_data'] is not None:
                self.generate_comparative_visualizations(
                    data_dict['fraud_data'], 
                    data_dict['creditcard_data'], 
                    vis_dir
                )
        
        logger.info(f"Visualizations saved to: {vis_dir}")
    
    def generate_fraud_data_visualizations(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Generate visualizations for fraud data.
        
        Args:
            df: Fraud dataframe
            output_dir: Output directory
        """
        logger.info("Generating fraud data visualizations...")
        
        # 1. Class distribution
        self._plot_class_distribution(df, output_dir / 'fraud_class_distribution')
        
        # 2. Time-based patterns
        self._plot_time_patterns(df, output_dir / 'fraud_time_patterns')
        
        # 3. Purchase value analysis
        self._plot_purchase_analysis(df, output_dir / 'fraud_purchase_analysis')
        
        # 4. Browser/device analysis
        self._plot_browser_device_analysis(df, output_dir / 'fraud_browser_device')
        
        # 5. Demographics analysis
        self._plot_demographics(df, output_dir / 'fraud_demographics')
        
        # 6. Correlation matrix
        self._plot_correlation_matrix(df, output_dir / 'fraud_correlation')
        
        # 7. Interactive dashboard
        if self.viz_config['interactive']:
            self._create_interactive_dashboard(df, output_dir / 'fraud_interactive_dashboard')
    
    def generate_creditcard_visualizations(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Generate visualizations for credit card data.
        
        Args:
            df: Credit card dataframe
            output_dir: Output directory
        """
        logger.info("Generating credit card data visualizations...")
        
        # 1. Class distribution (extremely imbalanced)
        self._plot_class_distribution(df, output_dir / 'creditcard_class_distribution', target_col='Class')
        
        # 2. Amount distribution
        self._plot_amount_distribution(df, output_dir / 'creditcard_amount_distribution')
        
        # 3. Time distribution
        if 'Time' in df.columns:
            self._plot_time_distribution(df, output_dir / 'creditcard_time_distribution')
        
        # 4. PCA components visualization
        v_columns = [col for col in df.columns if col.startswith('V')]
        if len(v_columns) > 0:
            self._plot_pca_components(df, v_columns, output_dir / 'creditcard_pca_components')
    
    def generate_comparative_visualizations(self, fraud_df: pd.DataFrame, 
                                          credit_df: pd.DataFrame, 
                                          output_dir: Path) -> None:
        """
        Generate comparative visualizations between fraud and credit card data.
        
        Args:
            fraud_df: Fraud dataframe
            credit_df: Credit card dataframe
            output_dir: Output directory
        """
        logger.info("Generating comparative visualizations...")
        
        # 1. Compare class imbalance
        self._compare_class_imbalance(fraud_df, credit_df, output_dir / 'comparative_imbalance')
        
        # 2. Compare fraud patterns
        self._compare_fraud_patterns(fraud_df, credit_df, output_dir / 'comparative_patterns')
    
    def _plot_class_distribution(self, df: pd.DataFrame, output_path: Path, 
                               target_col: str = 'class') -> None:
        """Plot class distribution."""
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count plot
        class_counts = df[target_col].value_counts()
        axes[0].bar(class_counts.index, class_counts.values, color=['green', 'red'])
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Class Distribution')
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Legitimate', 'Fraud'])
        
        # Add count labels
        for i, count in enumerate(class_counts.values):
            axes[0].text(i, count + max(class_counts.values)*0.01, 
                       f'{count:,}', ha='center', fontweight='bold')
        
        # Pie chart
        labels = ['Legitimate', 'Fraud']
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(class_counts.values, labels=labels, colors=colors, 
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Class Percentage')
        
        # Add fraud percentage as text
        fraud_percentage = (class_counts.get(1, 0) / len(df)) * 100
        plt.suptitle(f'Fraud Rate: {fraud_percentage:.2f}%', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close()
    
    def _plot_time_patterns(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot time-based patterns."""
        # Create time features if not present
        if 'purchase_time' in df.columns and df['purchase_time'].dtype == 'datetime64[ns]':
            df_temp = df.copy()
            df_temp['purchase_hour'] = df_temp['purchase_time'].dt.hour
            df_temp['purchase_day'] = df_temp['purchase_time'].dt.day_name()
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Hourly fraud rate
            hourly_stats = df_temp.groupby('purchase_hour')['class'].agg(['count', 'sum'])
            hourly_stats['fraud_rate'] = (hourly_stats['sum'] / hourly_stats['count']) * 100
            
            axes[0, 0].plot(hourly_stats.index, hourly_stats['fraud_rate'], 
                          marker='o', color='red', linewidth=2)
            axes[0, 0].set_xlabel('Hour of Day')
            axes[0, 0].set_ylabel('Fraud Rate (%)')
            axes[0, 0].set_title('Fraud Rate by Hour')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Daily fraud rate
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            daily_stats = df_temp.groupby('purchase_day')['class'].agg(['count', 'sum'])
            daily_stats = daily_stats.reindex(day_order)
            daily_stats['fraud_rate'] = (daily_stats['sum'] / daily_stats['count']) * 100
            
            bars = axes[0, 1].bar(daily_stats.index, daily_stats['fraud_rate'], 
                                color='skyblue')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Fraud Rate (%)')
            axes[0, 1].set_title('Fraud Rate by Day')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Color weekends differently
            for i, day in enumerate(daily_stats.index):
                if day in ['Saturday', 'Sunday']:
                    bars[i].set_color('orange')
            
            # Time since signup analysis (if available)
            if 'time_since_signup_hours' in df_temp.columns:
                # Create bins for time since signup
                time_bins = [0, 1, 6, 24, 168, float('inf')]  # 1h, 6h, 1d, 1w
                time_labels = ['<1h', '1-6h', '6-24h', '1-7d', '>7d']
                
                df_temp['time_since_group'] = pd.cut(df_temp['time_since_signup_hours'], 
                                                   bins=time_bins, labels=time_labels)
                
                time_stats = df_temp.groupby('time_since_group')['class'].agg(['count', 'sum'])
                time_stats['fraud_rate'] = (time_stats['sum'] / time_stats['count']) * 100
                
                axes[1, 0].bar(time_stats.index, time_stats['fraud_rate'], color='purple')
                axes[1, 0].set_xlabel('Time Since Signup')
                axes[1, 0].set_ylabel('Fraud Rate (%)')
                axes[1, 0].set_title('Fraud Rate vs Time Since Signup')
                
                # Add immediate purchase insight
                immediate_fraud = df_temp[df_temp['time_since_signup_hours'] < 1]['class'].sum()
                immediate_total = len(df_temp[df_temp['time_since_signup_hours'] < 1])
                if immediate_total > 0:
                    immediate_rate = (immediate_fraud / immediate_total) * 100
                    axes[1, 0].text(0.5, 0.95, 
                                   f'Immediate (<1h): {immediate_rate:.1f}%',
                                   transform=axes[1, 0].transAxes,
                                   ha='center', fontweight='bold', color='red')
            
            # Night vs Day comparison
            if 'purchase_hour' in df_temp.columns:
                df_temp['is_night'] = ((df_temp['purchase_hour'] >= 0) & 
                                      (df_temp['purchase_hour'] < 6)).astype(int)
                night_stats = df_temp.groupby('is_night')['class'].agg(['count', 'sum'])
                night_stats['fraud_rate'] = (night_stats['sum'] / night_stats['count']) * 100
                
                axes[1, 1].bar(['Day', 'Night'], night_stats['fraud_rate'], 
                              color=['lightblue', 'darkblue'])
                axes[1, 1].set_xlabel('Time of Day')
                axes[1, 1].set_ylabel('Fraud Rate (%)')
                axes[1, 1].set_title('Night vs Day Fraud Rate')
                
                # Add percentage labels
                for i, rate in enumerate(night_stats['fraud_rate']):
                    axes[1, 1].text(i, rate + max(night_stats['fraud_rate'])*0.02,
                                   f'{rate:.1f}%', ha='center')
            
            plt.suptitle('Time Pattern Analysis - Fraud Detection', fontsize=16, fontweight='bold')
            plt.tight_layout()
            self._save_figure(fig, output_path)
            plt.close()
    
    def _plot_purchase_analysis(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot purchase value analysis."""
        if 'purchase_value' in df.columns and 'class' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Distribution of purchase values
            axes[0, 0].hist(df['purchase_value'], bins=50, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Purchase Value ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Purchase Values')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot: Fraud vs Legitimate
            fraud_values = df[df['class'] == 1]['purchase_value']
            legit_values = df[df['class'] == 0]['purchase_value']
            
            box_data = [legit_values, fraud_values]
            box_labels = ['Legitimate', 'Fraud']
            
            bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            axes[0, 1].set_ylabel('Purchase Value ($)')
            axes[0, 1].set_title('Purchase Value: Fraud vs Legitimate')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Fraud rate by purchase amount bands
            amount_bins = [0, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
            amount_labels = ['<$10', '$10-20', '$20-50', '$50-100', 
                            '$100-200', '$200-500', '$500-1000', '>$1000']
            
            df_temp = df.copy()
            df_temp['amount_band'] = pd.cut(df_temp['purchase_value'], 
                                          bins=amount_bins, labels=amount_labels)
            
            band_stats = df_temp.groupby('amount_band')['class'].agg(['count', 'sum'])
            band_stats['fraud_rate'] = (band_stats['sum'] / band_stats['count']) * 100
            
            bars = axes[1, 0].bar(band_stats.index, band_stats['fraud_rate'], color='orange')
            axes[1, 0].set_xlabel('Purchase Amount Band')
            axes[1, 0].set_ylabel('Fraud Rate (%)')
            axes[1, 0].set_title('Fraud Rate by Purchase Amount')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Find and highlight highest fraud band
            max_band = band_stats['fraud_rate'].idxmax()
            max_rate = band_stats['fraud_rate'].max()
            for i, band in enumerate(band_stats.index):
                if band == max_band:
                    bars[i].set_color('red')
                    axes[1, 0].text(i, max_rate + 0.5, f'Max: {max_rate:.1f}%', 
                                   ha='center', fontweight='bold', color='red')
            
            # Scatter plot: Purchase value vs Time since signup (if available)
            if 'time_since_signup_hours' in df.columns:
                scatter = axes[1, 1].scatter(df['time_since_signup_hours'], 
                                           df['purchase_value'],
                                           c=df['class'], cmap='RdYlGn_r',
                                           alpha=0.6, s=30)
                axes[1, 1].set_xlabel('Time Since Signup (hours)')
                axes[1, 1].set_ylabel('Purchase Value ($)')
                axes[1, 1].set_title('Purchase Value vs Time Since Signup')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=axes[1, 1])
                cbar.set_label('Class (0=Legit, 1=Fraud)')
                
                # Log-log scale if needed
                if df['time_since_signup_hours'].max() > 1000:
                    axes[1, 1].set_xscale('log')
                    axes[1, 1].set_yscale('log')
            
            plt.suptitle('Purchase Behavior Analysis - Fraud Detection', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            self._save_figure(fig, output_path)
            plt.close()
    
    def _save_figure(self, fig, output_path: Path) -> None:
        """Save figure in multiple formats."""
        for fmt in self.viz_config['save_formats']:
            if fmt == 'html' and hasattr(fig, 'write_html'):
                # For Plotly figures
                fig.write_html(str(output_path) + f'.{fmt}')
            else:
                # For Matplotlib figures
                fig.savefig(str(output_path) + f'.{fmt}', 
                          bbox_inches='tight', dpi=self.viz_config['dpi'])
        logger.info(f"Saved visualization: {output_path}")
    
    def _create_interactive_dashboard(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create interactive dashboard using Plotly."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Class Distribution', 'Fraud Rate by Hour',
                               'Purchase Value Distribution', 'Time Since Signup',
                               'Browser Analysis', 'Device Risk'),
                specs=[[{'type': 'pie'}, {'type': 'scatter'}, {'type': 'histogram'}],
                       [{'type': 'histogram'}, {'type': 'bar'}, {'type': 'box'}]]
            )
            
            # 1. Class distribution (pie chart)
            class_counts = df['class'].value_counts()
            fig.add_trace(
                go.Pie(labels=['Legitimate', 'Fraud'], 
                      values=class_counts.values,
                      hole=0.3,
                      marker_colors=['green', 'red']),
                row=1, col=1
            )
            
            # 2. Fraud rate by hour (if time features exist)
            if 'purchase_hour' in df.columns:
                hourly_stats = df.groupby('purchase_hour')['class'].agg(['count', 'sum'])
                hourly_stats['fraud_rate'] = (hourly_stats['sum'] / hourly_stats['count']) * 100
                
                fig.add_trace(
                    go.Scatter(x=hourly_stats.index, y=hourly_stats['fraud_rate'],
                             mode='lines+markers',
                             name='Fraud Rate',
                             line=dict(color='red', width=2)),
                    row=1, col=2
                )
            
            # 3. Purchase value distribution
            fig.add_trace(
                go.Histogram(x=df['purchase_value'], nbinsx=50,
                           name='Purchase Value',
                           marker_color='blue'),
                row=1, col=3
            )
            
            # 4. Time since signup (if available)
            if 'time_since_signup_hours' in df.columns:
                fig.add_trace(
                    go.Histogram(x=df['time_since_signup_hours'], nbinsx=50,
                               name='Time Since Signup',
                               marker_color='purple'),
                    row=2, col=1
                )
            
            # 5. Browser analysis
            if 'browser' in df.columns:
                browser_stats = df.groupby('browser')['class'].agg(['count', 'sum'])
                browser_stats['fraud_rate'] = (browser_stats['sum'] / browser_stats['count']) * 100
                
                fig.add_trace(
                    go.Bar(x=browser_stats.index, y=browser_stats['fraud_rate'],
                          name='Browser Fraud Rate',
                          marker_color='orange'),
                    row=2, col=2
                )
            
            # 6. Device risk (if available)
            if 'device_risk_score' in df.columns:
                fig.add_trace(
                    go.Box(y=df['device_risk_score'], name='Device Risk Score',
                          marker_color='brown'),
                    row=2, col=3
                )
            
            # Update layout
            fig.update_layout(height=800, showlegend=True,
                            title_text="Interactive Fraud Analysis Dashboard",
                            title_font_size=20)
            
            # Save interactive dashboard
            fig.write_html(str(output_path) + '.html')
            logger.info(f"Saved interactive dashboard: {output_path}.html")
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")