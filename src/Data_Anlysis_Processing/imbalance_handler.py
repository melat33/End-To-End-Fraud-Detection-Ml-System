# task1-data-preprocessing/src/Data_Anlysis_Processing/imbalance_handler.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImbalanceHandler:
    """
    Handle class imbalance in fraud detection datasets.
    Supports multiple resampling techniques.
    """
    
    def __init__(self, config_path: str = "config/config_data_analysis.yaml"):
        """Initialize with configuration."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.imbalance_config = self.config['imbalance_handling']
        self.method = self.imbalance_config['method']
        self.sampling_strategy = self.imbalance_config['sampling_strategy']
        self.random_state = self.imbalance_config['random_state']
        self.k_neighbors = self.imbalance_config['k_neighbors']
        
        logger.info(f"Imbalance handler initialized with method: {self.method}")
    
    def handle_fraud_data_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle class imbalance in fraud data.
        
        Args:
            df: Fraud dataframe
            
        Returns:
            Balanced dataframe
        """
        logger.info("Handling class imbalance for fraud data...")
        
        # Check if class column exists
        if 'class' not in df.columns:
            logger.error("Class column not found in fraud data")
            return df
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        # Apply resampling
        X_resampled, y_resampled = self._apply_resampling(X, y)
        
        # Combine back
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled['class'] = y_resampled
        
        # Log results
        original_ratio = y.value_counts()
        resampled_ratio = pd.Series(y_resampled).value_counts()
        
        logger.info(f"Original class distribution: {original_ratio.to_dict()}")
        logger.info(f"Resampled class distribution: {resampled_ratio.to_dict()}")
        logger.info(f"Fraud percentage: {original_ratio[1]/len(y)*100:.2f}% -> {resampled_ratio[1]/len(y_resampled)*100:.2f}%")
        
        return df_resampled
    
    def handle_creditcard_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle class imbalance in credit card data.
        
        Args:
            df: Credit card dataframe
            
        Returns:
            Balanced dataframe
        """
        logger.info("Handling class imbalance for credit card data...")
        
        # Check if Class column exists
        if 'Class' not in df.columns:
            logger.error("Class column not found in credit card data")
            return df
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # For credit card data, we might want to use different strategy
        # due to extreme imbalance (0.17% fraud)
        if self.method == 'SMOTE':
            # Use different sampling strategy for credit card data
            sampling_strategy = min(0.1, self.sampling_strategy)  # Cap at 10%
        else:
            sampling_strategy = self.sampling_strategy
        
        # Apply resampling
        X_resampled, y_resampled = self._apply_resampling(X, y, sampling_strategy)
        
        # Combine back
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled['Class'] = y_resampled
        
        # Log results
        original_ratio = y.value_counts()
        resampled_ratio = pd.Series(y_resampled).value_counts()
        
        logger.info(f"Original class distribution: {original_ratio.to_dict()}")
        logger.info(f"Resampled class distribution: {resampled_ratio.to_dict()}")
        logger.info(f"Fraud percentage: {original_ratio[1]/len(y)*100:.2f}% -> {resampled_ratio[1]/len(y_resampled)*100:.2f}%")
        
        return df_resampled
    
    def _apply_resampling(self, X, y, sampling_strategy=None):
        """
        Apply the selected resampling method.
        
        Args:
            X: Features
            y: Target
            sampling_strategy: Optional custom sampling strategy
            
        Returns:
            Resampled X and y
        """
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        
        if self.method == 'SMOTE':
            sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state,
                k_neighbors=self.k_neighbors
            )
        elif self.method == 'RandomOverSampler':
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        elif self.method == 'RandomUnderSampler':
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        else:
            logger.warning(f"Unknown method {self.method}, using RandomOverSampler")
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            logger.info(f"Applied {self.method} with strategy {sampling_strategy}")
            return X_resampled, y_resampled
        except Exception as e:
            logger.error(f"Error applying {self.method}: {e}")
            logger.info("Returning original data")
            return X, y
    
    def analyze_imbalance(self, df: pd.DataFrame, target_col: str = 'class') -> Dict[str, Any]:
        """
        Analyze class imbalance in dataset.
        
        Args:
            df: Dataframe
            target_col: Target column name
            
        Returns:
            Dictionary with imbalance analysis
        """
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return {}
        
        y = df[target_col]
        class_counts = y.value_counts()
        total_samples = len(y)
        
        # Calculate imbalance ratios
        if len(class_counts) == 2:
            majority_class = class_counts.idxmax()
            minority_class = class_counts.idxmin()
            
            imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
            minority_percentage = (class_counts[minority_class] / total_samples) * 100
            
            analysis = {
                'total_samples': total_samples,
                'class_distribution': class_counts.to_dict(),
                'majority_class': majority_class,
                'minority_class': minority_class,
                'imbalance_ratio': round(imbalance_ratio, 2),
                'minority_percentage': round(minority_percentage, 4),
                'severity': self._assess_imbalance_severity(minority_percentage, imbalance_ratio)
            }
        else:
            analysis = {
                'total_samples': total_samples,
                'class_distribution': class_counts.to_dict(),
                'warning': 'Multi-class problem detected'
            }
        
        return analysis
    
    def _assess_imbalance_severity(self, minority_percentage: float, 
                                 imbalance_ratio: float) -> str:
        """
        Assess severity of class imbalance.
        
        Args:
            minority_percentage: Percentage of minority class
            imbalance_ratio: Ratio of majority to minority
            
        Returns:
            Severity level
        """
        if minority_percentage < 1:
            return 'EXTREME'
        elif minority_percentage < 5:
            return 'SEVERE'
        elif minority_percentage < 20:
            return 'MODERATE'
        else:
            return 'MILD'
    
    def handle_imbalance(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Handle imbalance for all datasets.
        
        Args:
            data_dict: Dictionary containing all datasets
            
        Returns:
            Dictionary with balanced datasets
        """
        result = {}
        
        if 'fraud_data' in data_dict and data_dict['fraud_data'] is not None:
            # Analyze imbalance first
            analysis = self.analyze_imbalance(data_dict['fraud_data'], 'class')
            logger.info(f"Fraud data imbalance analysis: {analysis.get('severity', 'UNKNOWN')}")
            
            # Handle imbalance if needed
            if analysis.get('severity') in ['EXTREME', 'SEVERE', 'MODERATE']:
                result['fraud_data'] = self.handle_fraud_data_imbalance(data_dict['fraud_data'])
            else:
                result['fraud_data'] = data_dict['fraud_data']
                logger.info("Fraud data imbalance is mild, skipping resampling")
        
        if 'creditcard_data' in data_dict and data_dict['creditcard_data'] is not None:
            # Analyze imbalance first
            analysis = self.analyze_imbalance(data_dict['creditcard_data'], 'Class')
            logger.info(f"Credit card data imbalance analysis: {analysis.get('severity', 'UNKNOWN')}")
            
            # Always handle imbalance for credit card data (it's extremely imbalanced)
            result['creditcard_data'] = self.handle_creditcard_imbalance(data_dict['creditcard_data'])
        
        # Copy other datasets unchanged
        for key, df in data_dict.items():
            if key not in result:
                result[key] = df
        
        return result