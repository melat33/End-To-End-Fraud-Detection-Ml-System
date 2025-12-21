"""
EXPLICIT DATA TRANSFORMATION PIPELINE FOR SCORING
Handles scaling and encoding with proper train/test split
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import logging
from typing import List, Dict, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransformer:
    """
    EXPLICIT TRANSFORMATION PIPELINE FOR SCORING:
    1. Numeric feature scaling (StandardScaler)
    2. Categorical feature encoding (OneHotEncoder)
    3. Proper train/test split handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []
        self.is_fitted = False
        
    def create_preprocessing_pipeline(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """CREATE EXPLICIT PREPROCESSING PIPELINE"""
        logger.info("="*60)
        logger.info("CREATING EXPLICIT TRANSFORMATION PIPELINE")
        logger.info("="*60)
        
        # Identify feature types
        self.numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numeric_features)} numeric features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")
        
        # Create transformers based on config
        numeric_transformer = self._create_numeric_transformer()
        categorical_transformer = self._create_categorical_transformer()
        
        # Create column transformer
        transformers = []
        
        if self.numeric_features:
            transformers.append(('numeric', numeric_transformer, self.numeric_features))
            logger.info(f"Added numeric transformer with {len(self.numeric_features)} features")
        
        if self.categorical_features:
            transformers.append(('categorical', categorical_transformer, self.categorical_features))
            logger.info(f"Added categorical transformer with {len(self.categorical_features)} features")
        
        if not transformers:
            raise ValueError("No features identified for transformation!")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        logger.info("✓ Transformation pipeline created successfully")
        logger.info(f"Total features to transform: {len(self.numeric_features) + len(self.categorical_features)}")
        
        return self.preprocessor
    
    def _create_numeric_transformer(self) -> Pipeline:
        """CREATE EXPLICIT NUMERIC TRANSFORMER WITH SCALING"""
        scaling_method = self.config['data_transformation']['scaling']['method']
        
        logger.info(f"Creating numeric transformer with {scaling_method} scaling")
        
        steps = []
        
        # 1. Imputation (if needed - though data should be clean)
        steps.append(('imputer', SimpleImputer(strategy='median')))
        
        # 2. Scaling (EXPLICIT AS PER REQUIREMENTS)
        if scaling_method == 'standard':
            scaler = StandardScaler()
            logger.info("  Using StandardScaler (z-score normalization)")
        elif scaling_method == 'robust':
            scaler = RobustScaler()
            logger.info("  Using RobustScaler (robust to outliers)")
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            logger.info("  Using MinMaxScaler (range [0, 1])")
        else:
            scaler = StandardScaler()
            logger.warning(f"  Unknown scaler {scaling_method}, using StandardScaler")
        
        steps.append(('scaler', scaler))
        
        return Pipeline(steps=steps)
    
    def _create_categorical_transformer(self) -> Pipeline:
        """CREATE EXPLICIT CATEGORICAL TRANSFORMER WITH ENCODING"""
        encoding_method = self.config['data_transformation']['encoding']['method']
        handle_unknown = self.config['data_transformation']['encoding']['handle_unknown']
        drop_first = self.config['data_transformation']['encoding']['drop_first']
        
        logger.info(f"Creating categorical transformer with {encoding_method} encoding")
        
        # Handle high cardinality features differently
        high_cardinality_threshold = self.config['data_transformation']['encoding']['high_cardinality_threshold']
        frequency_encoding_for = self.config['data_transformation']['encoding']['frequency_encoding_for']
        
        if encoding_method == 'onehot':
            encoder = OneHotEncoder(
                handle_unknown=handle_unknown,
                drop='first' if drop_first else None,
                sparse_output=False
            )
            logger.info("  Using OneHotEncoder (binary columns)")
            
        elif encoding_method == 'ordinal':
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            logger.info("  Using OrdinalEncoder (integer encoding)")
            
        else:
            encoder = OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )
            logger.warning(f"  Unknown encoder {encoding_method}, using OneHotEncoder")
        
        steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', encoder)
        ]
        
        return Pipeline(steps=steps)
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> 'DataTransformer':
        """FIT TRANSFORMER ON TRAINING DATA ONLY (PREVENT DATA LEAKAGE)"""
        logger.info("\nFITTING TRANSFORMER ON TRAINING DATA...")
        
        if self.preprocessor is None:
            self.create_preprocessing_pipeline(X_train)
        
        logger.info("Training data statistics before fitting:")
        logger.info(f"  Shape: {X_train.shape}")
        logger.info(f"  Numeric features: {self.numeric_features[:3]}...")
        logger.info(f"  Categorical features: {self.categorical_features[:3]}...")
        
        # FIT ONLY ON TRAINING DATA
        self.preprocessor.fit(X_train, y_train)
        self.is_fitted = True
        
        # Get feature names
        self.feature_names = self.get_feature_names()
        
        logger.info("✓ Transformer fitted successfully")
        logger.info(f"  Transformed features: {len(self.feature_names)}")
        logger.info(f"  Feature names: {self.feature_names[:5]}...")
        
        return self
    
    def transform(self, X_data: pd.DataFrame) -> np.ndarray:
        """TRANSFORM DATA USING FITTED TRANSFORMER"""
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transforming data!")
        
        logger.info(f"Transforming data with shape: {X_data.shape}")
        
        # TRANSFORM USING FITTED TRANSFORMER
        X_transformed = self.preprocessor.transform(X_data)
        
        logger.info(f"✓ Data transformed to shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> np.ndarray:
        """FIT AND TRANSFORM TRAINING DATA"""
        logger.info("\nFITTING AND TRANSFORMING TRAINING DATA...")
        
        self.fit(X_train, y_train)
        X_transformed = self.transform(X_train)
        
        return X_transformed
    
    def transform_with_names(self, X_data: pd.DataFrame) -> pd.DataFrame:
        """Transform data and return as DataFrame with feature names"""
        X_transformed = self.transform(X_data)
        
        if self.feature_names and len(self.feature_names) == X_transformed.shape[1]:
            return pd.DataFrame(X_transformed, columns=self.feature_names)
        else:
            return pd.DataFrame(X_transformed)
    
    def get_feature_names(self) -> List[str]:
        """Get names of transformed features"""
        if self.preprocessor is None:
            return []
        
        try:
            return self.preprocessor.get_feature_names_out().tolist()
        except:
            # Fallback for sklearn versions
            feature_names = []
            
            # Numeric features
            for feature in self.numeric_features:
                feature_names.append(f"num_{feature}")
            
            # Categorical features (for OneHotEncoder)
            if self.preprocessor.named_transformers_['categorical'] is not None:
                encoder = self.preprocessor.named_transformers_['categorical'].named_steps['encoder']
                if hasattr(encoder, 'get_feature_names_out'):
                    cat_features = encoder.get_feature_names_out(self.categorical_features)
                    feature_names.extend(cat_features.tolist())
            
            return feature_names
    
    def get_transformation_summary(self) -> Dict:
        """Get summary of transformation"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        summary = {
            "status": "fitted",
            "fitted_on_training": True,
            "transformation_methods": {
                "numeric_scaling": self.config['data_transformation']['scaling']['method'],
                "categorical_encoding": self.config['data_transformation']['encoding']['method']
            },
            "feature_counts": {
                "original_numeric": len(self.numeric_features),
                "original_categorical": len(self.categorical_features),
                "transformed_total": len(self.feature_names)
            },
            "preprocessor_details": str(self.preprocessor)
        }
        
        return summary
    
    def save_transformer(self, filepath: str) -> None:
        """Save fitted transformer to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted transformer!")
        
        save_data = {
            'preprocessor': self.preprocessor,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"✓ Transformer saved to: {filepath}")
    
    def load_transformer(self, filepath: str) -> 'DataTransformer':
        """Load fitted transformer from disk"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.preprocessor = save_data['preprocessor']
        self.numeric_features = save_data['numeric_features']
        self.categorical_features = save_data['categorical_features']
        self.feature_names = save_data['feature_names']
        self.config = save_data['config']
        self.is_fitted = save_data['is_fitted']
        
        logger.info(f"✓ Transformer loaded from: {filepath}")
        
        return self