"""
EXPLICIT EDA UTILITIES FOR SCORING
Reusable functions for univariate and bivariate analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os
from typing import Dict, List, Tuple, Optional

class EDAUtilities:
    """
    EXPLICIT EDA UTILITIES SHOWING FRAUD RELATIONSHIPS
    1. Univariate analysis with fraud comparison
    2. Bivariate analysis with feature-fraud relationships
    3. Reusable functions (not just notebooks)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup plotting style."""
        plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
        
    def generate_comprehensive_eda(self, df: pd.DataFrame, target_col: str = 'class', 
                                   save_dir: str = None) -> Dict[str, plt.Figure]:
        """
        GENERATE COMPREHENSIVE EDA WITH EXPLICIT FRAUD RELATIONSHIPS
        
        Args:
            df: DataFrame with features and target
            target_col: Fraud label column name
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of generated figures
        """
        print("="*80)
        print("EXPLICIT EDA ANALYSIS WITH FRAUD RELATIONSHIPS")
        print("="*80)
        
        figures = {}
        
        # 1. TARGET DISTRIBUTION (Critical for scoring)
        print("\n1. TARGET VARIABLE ANALYSIS:")
        print("-"*40)
        figures['target_distribution'] = self.plot_target_distribution(df, target_col)
        
        # 2. UNIVARIATE ANALYSIS (Numeric features vs fraud)
        print("\n2. UNIVARIATE ANALYSIS - NUMERIC FEATURES:")
        print("-"*40)
        numeric_figures = self.univariate_numeric_analysis(df, target_col, top_n=5)
        figures.update(numeric_figures)
        
        # 3. UNIVARIATE ANALYSIS (Categorical features vs fraud)
        print("\n3. UNIVARIATE ANALYSIS - CATEGORICAL FEATURES:")
        print("-"*40)
        categorical_figures = self.univariate_categorical_analysis(df, target_col, top_n=3)
        figures.update(categorical_figures)
        
        # 4. BIVARIATE ANALYSIS (Feature relationships with fraud)
        print("\n4. BIVARIATE ANALYSIS WITH FRAUD:")
        print("-"*40)
        bivariate_figures = self.bivariate_analysis(df, target_col, top_features=10)
        figures.update(bivariate_figures)
        
        # 5. CORRELATION ANALYSIS
        print("\n5. CORRELATION ANALYSIS:")
        print("-"*40)
        figures['correlation_matrix'] = self.plot_correlation_matrix(df, target_col)
        
        # 6. TIME-BASED ANALYSIS (If time features exist)
        print("\n6. TIME-BASED ANALYSIS:")
        print("-"*40)
        time_figures = self.time_analysis(df, target_col)
        figures.update(time_figures)
        
        # Save figures if requested
        if save_dir:
            self.save_figures(figures, save_dir)
        
        print("\n" + "="*80)
        print("EDA COMPLETE - KEY INSIGHTS:")
        self._print_key_insights(df, target_col)
        print("="*80)
        
        return figures
    
    def plot_target_distribution(self, df: pd.DataFrame, target_col: str) -> plt.Figure:
        """EXPLICIT TARGET DISTRIBUTION PLOT"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Pie chart
        fraud_count = df[target_col].sum()
        legit_count = len(df) - fraud_count
        labels = ['Legitimate', 'Fraud']
        sizes = [legit_count, fraud_count]
        colors = ['#00ff88', '#ff416c']
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Transaction Class Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = axes[1].bar(labels, sizes, color=colors)
        axes[1].set_title('Transaction Count by Class', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=11)
        
        # Fraud rate statistics
        fraud_rate = (fraud_count / len(df)) * 100
        imbalance_ratio = legit_count / fraud_count if fraud_count > 0 else float('inf')
        
        stats_text = f"""
        Statistics:
        • Total Transactions: {len(df):,}
        • Fraudulent: {fraud_count:,}
        • Legitimate: {legit_count:,}
        • Fraud Rate: {fraud_rate:.2f}%
        • Imbalance Ratio: {imbalance_ratio:.1f}:1
        """
        
        axes[2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].axis('off')
        axes[2].set_title('Fraud Statistics', fontsize=14, fontweight='bold')
        
        plt.suptitle('EXPLICIT TARGET VARIABLE ANALYSIS', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        print(f"✓ Target distribution plotted")
        print(f"  Fraud rate: {fraud_rate:.2f}%")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        return fig
    
    def univariate_numeric_analysis(self, df: pd.DataFrame, target_col: str, 
                                    top_n: int = 5) -> Dict[str, plt.Figure]:
        """EXPLICIT UNIVARIATE ANALYSIS FOR NUMERIC FEATURES WITH FRAUD COMPARISON"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_features:
            numeric_features.remove(target_col)
        
        # Select top N features with highest variance
        variances = df[numeric_features].var().sort_values(ascending=False)
        selected_features = variances.head(top_n).index.tolist()
        
        print(f"Analyzing {len(selected_features)} numeric features...")
        
        figures = {}
        
        for feature in selected_features:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Distribution plot with fraud overlay
            for target_val, color, label in [(0, '#00ff88', 'Legitimate'), (1, '#ff416c', 'Fraud')]:
                subset = df[df[target_col] == target_val]
                sns.kdeplot(data=subset[feature], label=label, color=color, fill=True, 
                           alpha=0.3, ax=axes[0])
            
            axes[0].set_title(f'{feature} Distribution by Fraud Status', fontsize=14, fontweight='bold')
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Density')
            axes[0].legend()
            
            # Box plot comparison
            sns.boxplot(data=df, x=target_col, y=feature, ax=axes[1], 
                       palette={0: '#00ff88', 1: '#ff416c'})
            axes[1].set_title(f'{feature} Box Plot by Fraud Status', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Fraud (0=Legit, 1=Fraud)')
            axes[1].set_ylabel(feature)
            
            # Calculate and display statistics
            legit_stats = df[df[target_col]==0][feature].describe()
            fraud_stats = df[df[target_col]==1][feature].describe()
            
            stats_text = f"""
            Statistics:
            Legitimate:
              Mean: {legit_stats['mean']:.2f}
              Std: {legit_stats['std']:.2f}
            
            Fraud:
              Mean: {fraud_stats['mean']:.2f}
              Std: {fraud_stats['std']:.2f}
            
            Difference: {(fraud_stats['mean'] - legit_stats['mean']):.2f}
            """
            
            # Add text box
            plt.figtext(0.5, -0.1, stats_text, ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'EXPLICIT UNIVARIATE ANALYSIS: {feature}', 
                        fontsize=16, fontweight='bold', y=1.05)
            plt.tight_layout()
            
            figures[f'univariate_numeric_{feature}'] = fig
            
            # Print insights
            print(f"\n  {feature}:")
            print(f"    Legitimate mean: {legit_stats['mean']:.2f}")
            print(f"    Fraud mean: {fraud_stats['mean']:.2f}")
            print(f"    Difference: {fraud_stats['mean'] - legit_stats['mean']:.2f}")
        
        return figures
    
    def univariate_categorical_analysis(self, df: pd.DataFrame, target_col: str, 
                                        top_n: int = 3) -> Dict[str, plt.Figure]:
        """EXPLICIT UNIVARIATE ANALYSIS FOR CATEGORICAL FEATURES WITH FRAUD RATES"""
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(categorical_features) == 0:
            print("  No categorical features found")
            return {}
        
        # Limit to top N features with reasonable cardinality
        selected_features = []
        for feature in categorical_features:
            if df[feature].nunique() <= 20:  # Reasonable for visualization
                selected_features.append(feature)
            if len(selected_features) >= top_n:
                break
        
        print(f"Analyzing {len(selected_features)} categorical features...")
        
        figures = {}
        
        for feature in selected_features:
            # Calculate fraud rates
            fraud_rates = df.groupby(feature)[target_col].mean() * 100
            fraud_rates = fraud_rates.sort_values(ascending=False)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Count plot
            sns.countplot(data=df, x=feature, hue=target_col, ax=axes[0],
                         palette={0: '#00ff88', 1: '#ff416c'})
            axes[0].set_title(f'{feature} Transaction Count', fontsize=14, fontweight='bold')
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].legend(['Legitimate', 'Fraud'])
            
            # Fraud rate plot
            bars = axes[1].bar(range(len(fraud_rates)), fraud_rates.values)
            axes[1].set_title(f'{feature} Fraud Rate by Category', fontsize=14, fontweight='bold')
            axes[1].set_xlabel(feature)
            axes[1].set_ylabel('Fraud Rate (%)')
            axes[1].set_xticks(range(len(fraud_rates)))
            axes[1].set_xticklabels(fraud_rates.index, rotation=45, ha='right')
            
            # Color bars by risk
            for idx, (bar, rate) in enumerate(zip(bars, fraud_rates.values)):
                if rate > fraud_rates.median():
                    bar.set_color('#ff416c')  # High risk
                elif rate > fraud_rates.quantile(0.25):
                    bar.set_color('#ffa500')  # Medium risk
                else:
                    bar.set_color('#00ff88')  # Low risk
                
                # Add value labels
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
            
            # Add risk insights
            top_risky = fraud_rates.head(3)
            risk_text = "Top Risky Categories:\n"
            for category, rate in top_risky.items():
                risk_text += f"• {category}: {rate:.1f}% fraud\n"
            
            plt.figtext(0.5, -0.15, risk_text, ha='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'EXPLICIT CATEGORICAL ANALYSIS: {feature}', 
                        fontsize=16, fontweight='bold', y=1.05)
            plt.tight_layout()
            
            figures[f'univariate_categorical_{feature}'] = fig
            
            # Print insights
            print(f"\n  {feature}:")
            for category, rate in top_risky.items():
                print(f"    {category}: {rate:.1f}% fraud")
        
        return figures
    
    def bivariate_analysis(self, df: pd.DataFrame, target_col: str, 
                          top_features: int = 10) -> Dict[str, plt.Figure]:
        """EXPLICIT BIVARIATE ANALYSIS SHOWING FEATURE-FRAUD RELATIONSHIPS"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_features:
            numeric_features.remove(target_col)
        
        # Calculate correlation with target
        correlations = {}
        for feature in numeric_features:
            if len(df[feature].unique()) > 1:
                corr = df[feature].corr(df[target_col])
                correlations[feature] = abs(corr)
        
        # Get top features by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_features_list = [f[0] for f in sorted_features[:min(top_features, len(sorted_features))]]
        
        print(f"Analyzing top {len(top_features_list)} features by correlation...")
        
        figures = {}
        
        # Create a 2x2 grid for top 4 features
        n_features = min(4, len(top_features_list))
        if n_features > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for idx, feature in enumerate(top_features_list[:4]):
                ax = axes[idx]
                
                # Scatter plot with fraud coloring
                scatter = ax.scatter(df[feature], df[target_col], 
                                   c=df[target_col], cmap='coolwarm', 
                                   alpha=0.6, s=20)
                ax.set_xlabel(feature, fontsize=12)
                ax.set_ylabel('Fraud (0/1)', fontsize=12)
                ax.set_title(f'{feature} vs Fraud (Corr: {correlations[feature]:.3f})', 
                           fontsize=14, fontweight='bold')
                
                # Add trend line
                z = np.polyfit(df[feature], df[target_col], 1)
                p = np.poly1d(z)
                ax.plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2)
                
                # Add stats
                legit_mean = df[df[target_col]==0][feature].mean()
                fraud_mean = df[df[target_col]==1][feature].mean()
                
                stats_text = f"Legit μ: {legit_mean:.2f}\nFraud μ: {fraud_mean:.2f}"
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                print(f"  {feature}: correlation={correlations[feature]:.3f}")
            
            plt.suptitle('EXPLICIT BIVARIATE ANALYSIS: Top Features vs Fraud', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            figures['bivariate_top_features'] = fig
        
        # Feature importance from simple model
        figures['feature_importance'] = self.plot_feature_importance(df, target_col)
        
        return figures
    
    def plot_correlation_matrix(self, df: pd.DataFrame, target_col: str) -> plt.Figure:
        """EXPLICIT CORRELATION MATRIX WITH FRAUD RELATIONSHIPS"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr()
        
        # Get correlations with target
        target_correlations = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Full correlation heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8}, ax=axes[0])
        axes[0].set_title('Correlation Matrix (All Features)', fontsize=14, fontweight='bold')
        
        # Bar plot of correlations with target
        colors = ['#ff416c' if corr > 0 else '#667eea' for corr in target_correlations.values]
        bars = axes[1].barh(range(len(target_correlations)), target_correlations.values, color=colors)
        axes[1].set_yticks(range(len(target_correlations)))
        axes[1].set_yticklabels(target_correlations.index)
        axes[1].set_xlabel('Correlation with Fraud')
        axes[1].set_title('Feature Correlation with Fraud Label', fontsize=14, fontweight='bold')
        
        # Add correlation values
        for i, (bar, corr) in enumerate(zip(bars, target_correlations.values)):
            axes[1].text(bar.get_width() + (0.01 if corr > 0 else -0.03), bar.get_y() + bar.get_height()/2,
                        f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center', fontsize=10)
        
        plt.suptitle('EXPLICIT CORRELATION ANALYSIS WITH FRAUD', 
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Print top correlations
        print("\nTop 5 features by correlation with fraud:")
        for feature, corr in target_correlations.head(5).items():
            print(f"  {feature}: {corr:.3f}")
        
        return fig
    
    def plot_feature_importance(self, df: pd.DataFrame, target_col: str, 
                               n_features: int = 10) -> plt.Figure:
        """EXPLICIT FEATURE IMPORTANCE PLOT"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = df[target_col]
        
        # Remove constant columns
        X = X.loc[:, X.nunique() > 1]
        
        if len(X.columns) == 0:
            print("  No suitable features for importance calculation")
            return None
        
        # Train simple model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        n_plot = min(n_features, len(importance))
        bars = ax.barh(range(n_plot), importance['importance'][:n_plot][::-1])
        ax.set_yticks(range(n_plot))
        ax.set_yticklabels(importance['feature'][:n_plot][::-1])
        ax.set_xlabel('Feature Importance Score')
        ax.set_title('EXPLICIT FEATURE IMPORTANCE FOR FRAUD DETECTION', 
                    fontsize=16, fontweight='bold')
        
        # Color bars by importance
        for i, bar in enumerate(bars):
            if i < 3:  # Top 3 features
                bar.set_color('#ff416c')
            elif i < 7:  # Next 4 features
                bar.set_color('#ffa500')
            else:
                bar.set_color('#667eea')
            
            # Add value labels
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{bar.get_width():.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        print("\nTop 3 features by importance:")
        for i, (feature, imp) in enumerate(importance.head(3).values):
            print(f"  {i+1}. {feature}: {imp:.3f}")
        
        return fig
    
    def time_analysis(self, df: pd.DataFrame, target_col: str) -> Dict[str, plt.Figure]:
        """EXPLICIT TIME-BASED ANALYSIS (if time features exist)"""
        figures = {}
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'hour' in col.lower()]
        
        if not time_cols:
            print("  No time features found for analysis")
            return figures
        
        print(f"Analyzing {len(time_cols)} time features...")
        
        for time_col in time_cols[:2]:  # Analyze first 2 time features
            if time_col in df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Histogram by fraud status
                for target_val, color, label in [(0, '#00ff88', 'Legitimate'), (1, '#ff416c', 'Fraud')]:
                    subset = df[df[target_col] == target_val]
                    axes[0].hist(subset[time_col], bins=30, alpha=0.5, color=color, label=label)
                
                axes[0].set_xlabel(time_col)
                axes[0].set_ylabel('Frequency')
                axes[0].set_title(f'{time_col} Distribution by Fraud', fontsize=14, fontweight='bold')
                axes[0].legend()
                
                # Fraud rate over time
                if df[time_col].nunique() > 10:
                    # Bin continuous time
                    df['time_bin'] = pd.qcut(df[time_col], 10, labels=False)
                else:
                    df['time_bin'] = df[time_col]
                
                fraud_rates = df.groupby('time_bin')[target_col].mean() * 100
                
                axes[1].plot(fraud_rates.index, fraud_rates.values, 'o-', color='#ff416c', linewidth=2)
                axes[1].fill_between(fraud_rates.index, fraud_rates.values, alpha=0.3, color='#ff416c')
                axes[1].set_xlabel(time_col + ' (binned)')
                axes[1].set_ylabel('Fraud Rate (%)')
                axes[1].set_title(f'Fraud Rate by {time_col}', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                
                plt.suptitle(f'EXPLICIT TIME ANALYSIS: {time_col}', 
                            fontsize=16, fontweight='bold', y=1.05)
                plt.tight_layout()
                
                figures[f'time_analysis_{time_col}'] = fig
        
        return figures
    
    def save_figures(self, figures: Dict[str, plt.Figure], save_dir: str):
        """Save all figures to directory."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for name, fig in figures.items():
            if fig is not None:
                filepath = os.path.join(save_dir, f"{name}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        print(f"\n✓ All figures saved to: {save_dir}")
    
    def _print_key_insights(self, df: pd.DataFrame, target_col: str):
        """Print key insights from EDA."""
        # Calculate basic fraud statistics
        fraud_rate = df[target_col].mean() * 100
        
        # Find top risky features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_features:
            numeric_features.remove(target_col)
        
        correlations = {}
        for feature in numeric_features:
            if len(df[feature].unique()) > 1:
                corr = df[feature].corr(df[target_col])
                correlations[feature] = abs(corr)
        
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"• Overall Fraud Rate: {fraud_rate:.2f}%")
        print(f"• Most Predictive Features:")
        for feature, corr in top_features:
            legit_mean = df[df[target_col]==0][feature].mean()
            fraud_mean = df[df[target_col]==1][feature].mean()
            direction = "higher" if fraud_mean > legit_mean else "lower"
            print(f"  - {feature}: Correlation={corr:.3f}, Fraud has {direction} values")