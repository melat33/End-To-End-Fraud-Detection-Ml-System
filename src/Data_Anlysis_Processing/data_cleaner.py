"""
EXPLICIT DATA CLEANING UTILITIES FOR SCORING
Handles missing values, duplicates, and data type corrections for BOTH datasets
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Comprehensive data cleaning with EXPLICIT methods for:
    1. Missing value treatment
    2. Duplicate removal  
    3. Data type corrections
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.cleaning_stats = {}
        self.dataset_type = None
        
    def clean_fraud_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXECUTE COMPLETE CLEANING PIPELINE FOR FRAUD DATA"""
        logger.info("="*60)
        logger.info("STARTING EXPLICIT FRAUD DATA CLEANING")
        logger.info("="*60)
        
        self.dataset_type = 'fraud'
        df_copy = df.copy()
        original_shape = df_copy.shape
        
        # 1. EXPLICIT DATA TYPE CORRECTIONS
        logger.info("\n1. EXPLICIT DATA TYPE CORRECTIONS:")
        df_copy = self._correct_datatypes_fraud(df_copy)
        
        # 2. EXPLICIT MISSING VALUE TREATMENT
        logger.info("\n2. EXPLICIT MISSING VALUE TREATMENT:")
        df_copy = self._handle_missing_values_fraud(df_copy)
        
        # 3. EXPLICIT DUPLICATE REMOVAL
        logger.info("\n3. EXPLICIT DUPLICATE REMOVAL:")
        df_copy, dup_stats = self._remove_duplicates_fraud(df_copy)
        
        # 4. DATA VALIDATION
        logger.info("\n4. DATA VALIDATION:")
        df_copy = self._validate_cleaning_fraud(df_copy)
        
        # 5. GENERATE CLEANING REPORT
        final_shape = df_copy.shape
        logger.info("\n" + "="*60)
        logger.info("CLEANING COMPLETE - SUMMARY:")
        logger.info(f"Original shape: {original_shape}")
        logger.info(f"Final shape: {final_shape}")
        logger.info(f"Rows removed: {original_shape[0] - final_shape[0]}")
        logger.info(f"Columns changed: {original_shape[1] - final_shape[1]}")
        logger.info("="*60)
        
        self._generate_cleaning_report(original_shape, final_shape)
        
        return df_copy
    
    def clean_credit_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXECUTE COMPLETE CLEANING PIPELINE FOR CREDIT CARD DATA"""
        logger.info("="*60)
        logger.info("STARTING EXPLICIT CREDIT CARD DATA CLEANING")
        logger.info("="*60)
        
        self.dataset_type = 'credit'
        df_copy = df.copy()
        original_shape = df_copy.shape
        
        # 1. EXPLICIT DATA TYPE CORRECTIONS
        logger.info("\n1. EXPLICIT DATA TYPE CORRECTIONS:")
        df_copy = self._correct_datatypes_credit(df_copy)
        
        # 2. EXPLICIT MISSING VALUE TREATMENT
        logger.info("\n2. EXPLICIT MISSING VALUE TREATMENT:")
        df_copy = self._handle_missing_values_credit(df_copy)
        
        # 3. EXPLICIT DUPLICATE REMOVAL
        logger.info("\n3. EXPLICIT DUPLICATE REMOVAL:")
        df_copy, dup_stats = self._remove_duplicates_credit(df_copy)
        
        # 4. DATA VALIDATION
        logger.info("\n4. DATA VALIDATION:")
        df_copy = self._validate_cleaning_credit(df_copy)
        
        # 5. GENERATE CLEANING REPORT
        final_shape = df_copy.shape
        logger.info("\n" + "="*60)
        logger.info("CLEANING COMPLETE - SUMMARY:")
        logger.info(f"Original shape: {original_shape}")
        logger.info(f"Final shape: {final_shape}")
        logger.info(f"Rows removed: {original_shape[0] - final_shape[0]}")
        logger.info("="*60)
        
        self._generate_cleaning_report(original_shape, final_shape)
        
        return df_copy
    
    def _correct_datatypes_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXPLICIT DATA TYPE CORRECTIONS FOR FRAUD DATA"""
        logger.info("Performing data type corrections...")
        
        # Datetime conversions
        datetime_cols = self.config['data_cleaning']['data_type_correction']['fraud_dataset']['datetime_columns']
        for col in datetime_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"  ✓ Converted {col} to datetime")
                except Exception as e:
                    logger.error(f"  ✗ Failed to convert {col}: {e}")
        
        # Categorical conversions
        cat_cols = self.config['data_cleaning']['data_type_correction']['fraud_dataset']['categorical_columns']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                logger.info(f"  ✓ Converted {col} to categorical")
        
        # Numeric conversions
        num_cols = self.config['data_cleaning']['data_type_correction']['fraud_dataset']['numeric_columns']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"  ✓ Converted {col} to numeric")
        
        # IP address conversion
        ip_col = self.config['data_cleaning']['data_type_correction']['fraud_dataset']['ip_column']
        if ip_col in df.columns:
            try:
                df[ip_col] = df[ip_col].astype(np.int64)
                logger.info(f"  ✓ Converted {ip_col} to integer")
            except:
                logger.warning(f"  ⚠ Could not convert {ip_col} to integer, keeping as string")
        
        return df
    
    def _handle_missing_values_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXPLICIT MISSING VALUE TREATMENT FOR FRAUD DATA"""
        logger.info("Treating missing values...")
        
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before treatment: {missing_before}")
        
        # Get strategies from config
        strategies = self.config['data_cleaning']['missing_value_strategy']['fraud_dataset']
        
        for column, strategy in strategies.items():
            if column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    if strategy == 'median':
                        fill_value = df[column].median()
                    elif strategy == 'mean':
                        fill_value = df[column].mean()
                    elif strategy == 'mode':
                        fill_value = df[column].mode()[0] if len(df[column].mode()) > 0 else 'Unknown'
                    elif strategy == 'unknown':
                        fill_value = 'Unknown'
                    elif strategy == 'drop':
                        df = df.dropna(subset=[column])
                        fill_value = None
                        logger.info(f"  ✓ Dropped {missing_count} rows with missing {column}")
                    else:
                        fill_value = strategy
                    
                    if fill_value is not None:
                        df[column] = df[column].fillna(fill_value)
                        logger.info(f"  ✓ Filled {missing_count} missing values in {column} with {strategy}")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after treatment: {missing_after}")
        logger.info(f"Reduction: {missing_before - missing_after} values")
        
        self.cleaning_stats['missing_values'] = {
            'before': int(missing_before),
            'after': int(missing_after),
            'reduction': int(missing_before - missing_after)
        }
        
        return df
    
    def _remove_duplicates_fraud(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """EXPLICIT DUPLICATE REMOVAL FOR FRAUD DATA"""
        logger.info("Removing duplicates...")
        
        duplicates_before = df.duplicated().sum()
        logger.info(f"Exact duplicates before removal: {duplicates_before}")
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # Remove near-duplicates based on business logic
        key_columns = self.config['data_cleaning']['duplicate_removal']['fraud_key_columns']
        available_cols = [col for col in key_columns if col in df.columns]
        
        if available_cols:
            duplicates_near_before = df.duplicated(subset=available_cols).sum()
            logger.info(f"Near-duplicates before removal: {duplicates_near_before}")
            
            df = df.drop_duplicates(subset=available_cols, keep='first')
            duplicates_near_after = df.duplicated(subset=available_cols).sum()
            near_duplicates_removed = duplicates_near_before - duplicates_near_after
            logger.info(f"Near-duplicates removed: {near_duplicates_removed}")
        else:
            near_duplicates_removed = 0
        
        duplicates_after = df.duplicated().sum()
        total_duplicates_removed = duplicates_before - duplicates_after + near_duplicates_removed
        
        logger.info(f"Total duplicates removed: {total_duplicates_removed}")
        logger.info(f"Remaining duplicates: {duplicates_after}")
        
        self.cleaning_stats['duplicates'] = {
            'exact_before': int(duplicates_before),
            'exact_after': int(duplicates_after),
            'near_removed': int(near_duplicates_removed),
            'total_removed': int(total_duplicates_removed)
        }
        
        return df, self.cleaning_stats['duplicates']
    
    def _validate_cleaning_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """VALIDATE CLEANING RESULTS FOR FRAUD DATA"""
        logger.info("Validating cleaning results...")
        
        # Check for remaining missing values
        missing_vals = df.isnull().sum().sum()
        if missing_vals > 0:
            logger.warning(f"  ⚠ {missing_vals} missing values remain")
        else:
            logger.info("  ✓ No missing values remain")
        
        # Check data types
        datetime_cols = self.config['data_cleaning']['data_type_correction']['fraud_dataset']['datetime_columns']
        for col in datetime_cols:
            if col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    logger.info(f"  ✓ {col} is correctly datetime")
                else:
                    logger.error(f"  ✗ {col} is NOT datetime: {df[col].dtype}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            logger.info("  ✓ No exact duplicates remain")
        else:
            logger.warning(f"  ⚠ {duplicates} exact duplicates remain")
        
        logger.info("Validation complete")
        return df
    
    def _correct_datatypes_credit(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXPLICIT DATA TYPE CORRECTIONS FOR CREDIT CARD DATA"""
        logger.info("Performing data type corrections...")
        
        # All V columns should be float
        v_columns = [col for col in df.columns if col.startswith('V')]
        for col in v_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"  ✓ Converted {col} to numeric")
        
        # Time and Amount columns
        if 'Time' in df.columns:
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            logger.info("  ✓ Converted Time to numeric")
        
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            logger.info("  ✓ Converted Amount to numeric")
        
        # Class should be int
        if 'Class' in df.columns and self.config['data_cleaning']['data_type_correction']['credit_dataset']['class_to_int']:
            df['Class'] = df['Class'].astype(int)
            logger.info("  ✓ Converted Class to integer")
        
        return df
    
    def _handle_missing_values_credit(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXPLICIT MISSING VALUE TREATMENT FOR CREDIT CARD DATA"""
        logger.info("Treating missing values...")
        
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before treatment: {missing_before}")
        
        if missing_before > 0:
            # For V columns, use median
            v_columns = [col for col in df.columns if col.startswith('V')]
            for col in v_columns:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"  ✓ Filled missing values in {col} with median")
            
            # For Amount, use median
            if 'Amount' in df.columns and df['Amount'].isnull().any():
                df['Amount'] = df['Amount'].fillna(df['Amount'].median())
                logger.info("  ✓ Filled missing Amount with median")
            
            # For Time, use median
            if 'Time' in df.columns and df['Time'].isnull().any():
                df['Time'] = df['Time'].fillna(df['Time'].median())
                logger.info("  ✓ Filled missing Time with median")
        else:
            logger.info("  ✓ No missing values found")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after treatment: {missing_after}")
        
        self.cleaning_stats['missing_values'] = {
            'before': int(missing_before),
            'after': int(missing_after),
            'reduction': int(missing_before - missing_after)
        }
        
        return df
    
    def _remove_duplicates_credit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """EXPLICIT DUPLICATE REMOVAL FOR CREDIT CARD DATA"""
        logger.info("Removing duplicates...")
        
        duplicates_before = df.duplicated().sum()
        logger.info(f"Exact duplicates before removal: {duplicates_before}")
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        # For credit card, check for transaction duplicates
        # (same Time, Amount, and V features)
        v_columns = [col for col in df.columns if col.startswith('V')]
        key_columns = ['Time', 'Amount'] + v_columns[:5]  # Use first 5 V features
        
        available_cols = [col for col in key_columns if col in df.columns]
        
        if len(available_cols) >= 3:  # Need at least Time, Amount and one V feature
            duplicates_near_before = df.duplicated(subset=available_cols).sum()
            logger.info(f"Near-duplicates before removal: {duplicates_near_before}")
            
            df = df.drop_duplicates(subset=available_cols, keep='first')
            duplicates_near_after = df.duplicated(subset=available_cols).sum()
            near_duplicates_removed = duplicates_near_before - duplicates_near_after
            logger.info(f"Near-duplicates removed: {near_duplicates_removed}")
        else:
            near_duplicates_removed = 0
        
        duplicates_after = df.duplicated().sum()
        total_duplicates_removed = duplicates_before - duplicates_after + near_duplicates_removed
        
        logger.info(f"Total duplicates removed: {total_duplicates_removed}")
        logger.info(f"Remaining duplicates: {duplicates_after}")
        
        self.cleaning_stats['duplicates'] = {
            'exact_before': int(duplicates_before),
            'exact_after': int(duplicates_after),
            'near_removed': int(near_duplicates_removed),
            'total_removed': int(total_duplicates_removed)
        }
        
        return df, self.cleaning_stats['duplicates']
    
    def _validate_cleaning_credit(self, df: pd.DataFrame) -> pd.DataFrame:
        """VALIDATE CLEANING RESULTS FOR CREDIT CARD DATA"""
        logger.info("Validating cleaning results...")
        
        # Check for remaining missing values
        missing_vals = df.isnull().sum().sum()
        if missing_vals > 0:
            logger.warning(f"  ⚠ {missing_vals} missing values remain")
        else:
            logger.info("  ✓ No missing values remain")
        
        # Check numeric types
        v_columns = [col for col in df.columns if col.startswith('V')]
        for col in v_columns[:3]:  # Check first 3
            if np.issubdtype(df[col].dtype, np.number):
                logger.info(f"  ✓ {col} is correctly numeric")
            else:
                logger.error(f"  ✗ {col} is NOT numeric: {df[col].dtype}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            logger.info("  ✓ No exact duplicates remain")
        else:
            logger.warning(f"  ⚠ {duplicates} exact duplicates remain")
        
        logger.info("Validation complete")
        return df
    
    def _generate_cleaning_report(self, original_shape: Tuple, final_shape: Tuple) -> Dict:
        """Generate comprehensive cleaning report"""
        report = {
            'dataset_type': self.dataset_type,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'rows_removed': original_shape[0] - final_shape[0],
            'columns_changed': original_shape[1] - final_shape[1],
            'cleaning_stats': self.cleaning_stats,
            'timestamp': datetime.now().isoformat(),
            'config_used': self.config['data_cleaning']
        }
        
        self.cleaning_report = report
        return report
    
    def get_cleaning_report(self) -> Dict:
        """Get the cleaning report"""
        return getattr(self, 'cleaning_report', {})