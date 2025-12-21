# task1-data-preprocessing/src/Data_Anlysis_Processing/data_validator.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any
import logging
from dataclasses import dataclass
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data class to store validation results."""
    is_valid: bool
    issues: List[str]
    statistics: Dict[str, Any]
    recommendations: List[str]

class DataValidator:
    """
    Comprehensive data validator for fraud detection datasets.
    Performs data quality checks, outlier detection, and consistency validation.
    """
    
    def __init__(self, config_path: str = "config/config_data_analysis.yaml"):
        """Initialize validator with configuration."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.validation_config = self.config['data_cleaning']
    
    def validate_fraud_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate fraud dataset.
        
        Args:
            df: Fraud dataframe
            
        Returns:
            ValidationResult object
        """
        issues = []
        statistics = {}
        recommendations = []
        
        # 1. Check for missing values
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        
        if len(missing_columns) > 0:
            issues.append(f"Missing values found in columns: {missing_columns.to_dict()}")
            recommendations.append("Consider imputation or removal of rows with missing values")
        
        statistics['missing_values'] = missing_values.to_dict()
        
        # 2. Check data types
        expected_dtypes = {
            'user_id': 'int64',
            'signup_time': 'datetime64[ns]',
            'purchase_time': 'datetime64[ns]',
            'purchase_value': 'float64',
            'device_id': 'object',
            'source': 'object',
            'browser': 'object',
            'sex': 'object',
            'age': 'int64',
            'ip_address': 'int64',
            'class': 'int64'
        }
        
        dtype_issues = []
        for col, expected_type in expected_dtypes.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not actual_type.startswith(expected_type.split('[')[0]):
                    dtype_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
        
        if dtype_issues:
            issues.append(f"Data type issues: {dtype_issues}")
        
        # 3. Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")
            recommendations.append("Remove duplicate rows to avoid data leakage")
        
        statistics['duplicate_count'] = duplicate_count
        
        # 4. Validate age range (from YOUR data: ages 18-53)
        age_config = self.validation_config['fraud_data']['age_range']
        invalid_ages = df[(df['age'] < age_config[0]) | (df['age'] > age_config[1])]
        
        if len(invalid_ages) > 0:
            issues.append(f"Found {len(invalid_ages)} rows with age outside valid range {age_config}")
            recommendations.append(f"Cap ages to range {age_config}")
        
        statistics['age_stats'] = {
            'min': df['age'].min(),
            'max': df['age'].max(),
            'mean': df['age'].mean(),
            'median': df['age'].median(),
            'invalid_count': len(invalid_ages)
        }
        
        # 5. Validate purchase values
        negative_purchases = df[df['purchase_value'] <= 0]
        if len(negative_purchases) > 0:
            issues.append(f"Found {len(negative_purchases)} rows with non-positive purchase values")
            recommendations.append("Remove transactions with non-positive amounts")
        
        # 6. Check for impossible timestamps (purchase before signup)
        time_inconsistencies = df[df['purchase_time'] < df['signup_time']]
        if len(time_inconsistencies) > 0:
            issues.append(f"Found {len(time_inconsistencies)} rows with purchase before signup")
            recommendations.append("Review and correct timestamp inconsistencies")
        
        # 7. Check IP address validity
        negative_ips = df[df['ip_address'] < 0]
        if len(negative_ips) > 0:
            issues.append(f"Found {len(negative_ips)} rows with negative IP addresses")
        
        # 8. Detect outliers using IQR method
        outlier_stats = self._detect_outliers(df, ['purchase_value', 'age'])
        statistics['outliers'] = outlier_stats
        
        if outlier_stats['purchase_value']['count'] > 0:
            issues.append(f"Found {outlier_stats['purchase_value']['count']} outliers in purchase_value")
            recommendations.append("Consider winsorizing or capping extreme purchase values")
        
        # 9. Class distribution validation
        class_dist = df['class'].value_counts()
        fraud_percentage = (class_dist.get(1, 0) / len(df)) * 100
        
        statistics['class_distribution'] = class_dist.to_dict()
        statistics['fraud_percentage'] = fraud_percentage
        
        if fraud_percentage < 0.1 or fraud_percentage > 50:
            issues.append(f"Unusual fraud percentage: {fraud_percentage:.2f}%")
            recommendations.append("Consider class imbalance handling techniques")
        
        # 10. Business rule validation from YOUR data
        immediate_purchases = df[df['signup_time'] == df['purchase_time']]
        if len(immediate_purchases) > 0:
            fraud_immediate = immediate_purchases[immediate_purchases['class'] == 1]
            if len(fraud_immediate) > 0:
                logger.warning(f"🚨 BUSINESS INSIGHT: {len(fraud_immediate)} fraud cases occurred INSTANTLY after signup")
                recommendations.append("Flag transactions within 1 hour of signup for additional verification")
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def validate_creditcard_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate credit card dataset.
        
        Args:
            df: Credit card dataframe
            
        Returns:
            ValidationResult object
        """
        issues = []
        statistics = {}
        recommendations = []
        
        # 1. Check for missing values
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        
        if len(missing_columns) > 0:
            issues.append(f"Missing values in credit card data: {missing_columns.to_dict()}")
        
        statistics['missing_values'] = missing_values.to_dict()
        
        # 2. Check Amount column
        negative_amounts = df[df['Amount'] < 0]
        if len(negative_amounts) > 0:
            issues.append(f"Found {len(negative_amounts)} negative transaction amounts")
        
        # 3. Check Time column consistency
        if 'Time' in df.columns:
            time_stats = df['Time'].describe()
            statistics['time_stats'] = time_stats.to_dict()
            
            # Check if Time is monotonically increasing (should be in this dataset)
            if not df['Time'].is_monotonic_increasing:
                issues.append("Time column is not monotonically increasing")
        
        # 4. Check V features (PCA components)
        v_columns = [col for col in df.columns if col.startswith('V')]
        v_stats = {}
        
        for v_col in v_columns:
            col_stats = df[v_col].describe()
            v_stats[v_col] = {
                'mean': col_stats['mean'],
                'std': col_stats['std'],
                'min': col_stats['min'],
                'max': col_stats['max']
            }
        
        statistics['v_features_stats'] = v_stats
        
        # 5. Class distribution
        class_dist = df['Class'].value_counts()
        fraud_percentage = (class_dist.get(1, 0) / len(df)) * 100
        
        statistics['class_distribution'] = class_dist.to_dict()
        statistics['fraud_percentage'] = fraud_percentage
        
        if fraud_percentage < 0.01:
            issues.append(f"Extremely low fraud percentage: {fraud_percentage:.3f}%")
            recommendations.append("Use specialized techniques for highly imbalanced data")
        
        # 6. Detect outliers in Amount
        amount_outliers = self._detect_outliers(df, ['Amount'])
        statistics['amount_outliers'] = amount_outliers
        
        if amount_outliers['Amount']['count'] > 0:
            issues.append(f"Found {amount_outliers['Amount']['count']} outliers in Amount")
            recommendations.append("Apply log transformation to Amount column")
        
        is_valid = len(issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations
        )
    
    def _detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Dataframe
            columns: Columns to check for outliers
            
        Returns:
            Dictionary with outlier statistics
        """
        outlier_stats = {}
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outlier_stats[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min_outlier': outliers[col].min() if len(outliers) > 0 else None,
                    'max_outlier': outliers[col].max() if len(outliers) > 0 else None
                }
        
        return outlier_stats
    
    def generate_validation_report(self, fraud_result: ValidationResult,
                                 credit_result: ValidationResult) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            fraud_result: Fraud data validation result
            credit_result: Credit card data validation result
            
        Returns:
            Comprehensive validation report
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'fraud_data': {
                'is_valid': fraud_result.is_valid,
                'issue_count': len(fraud_result.issues),
                'issues': fraud_result.issues,
                'recommendations': fraud_result.recommendations,
                'statistics': fraud_result.statistics
            },
            'creditcard_data': {
                'is_valid': credit_result.is_valid,
                'issue_count': len(credit_result.issues),
                'issues': credit_result.issues,
                'recommendations': credit_result.recommendations,
                'statistics': credit_result.statistics
            },
            'summary': {
                'total_issues': len(fraud_result.issues) + len(credit_result.issues),
                'critical_issues': self._identify_critical_issues(fraud_result, credit_result),
                'overall_status': 'PASS' if (fraud_result.is_valid and credit_result.is_valid) else 'FAIL',
                'data_quality_score': self._calculate_data_quality_score(fraud_result, credit_result)
            }
        }
        
        return report
    
    def _identify_critical_issues(self, fraud_result: ValidationResult,
                                credit_result: ValidationResult) -> List[str]:
        """Identify critical issues that need immediate attention."""
        critical_issues = []
        
        # Check for immediate fraud patterns (from YOUR data insight)
        if 'immediate_purchases' in str(fraud_result.issues):
            critical_issues.append("Immediate purchases after signup - high fraud risk")
        
        # Check for extreme class imbalance
        fraud_pct = fraud_result.statistics.get('fraud_percentage', 0)
        credit_pct = credit_result.statistics.get('fraud_percentage', 0)
        
        if fraud_pct < 0.1:
            critical_issues.append(f"Extreme class imbalance in fraud data: {fraud_pct:.3f}% fraud")
        
        if credit_pct < 0.01:
            critical_issues.append(f"Extreme class imbalance in credit data: {credit_pct:.3f}% fraud")
        
        return critical_issues
    
    def _calculate_data_quality_score(self, fraud_result: ValidationResult,
                                    credit_result: ValidationResult) -> float:
        """Calculate overall data quality score (0-100)."""
        max_penalty = 100
        penalty = 0
        
        # Penalty for issues
        penalty += len(fraud_result.issues) * 5
        penalty += len(credit_result.issues) * 5
        
        # Penalty for data type issues
        if 'Data type issues' in str(fraud_result.issues):
            penalty += 10
        
        # Penalty for missing values
        fraud_missing = sum(fraud_result.statistics.get('missing_values', {}).values())
        credit_missing = sum(credit_result.statistics.get('missing_values', {}).values())
        penalty += (fraud_missing + credit_missing) * 0.1
        
        # Penalty for duplicates
        penalty += fraud_result.statistics.get('duplicate_count', 0) * 2
        
        # Ensure score is between 0 and 100
        quality_score = max(0, 100 - penalty)
        
        return round(quality_score, 2)
    
    def clean_fraud_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean fraud data based on validation results."""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Cap ages to valid range
        age_config = self.validation_config['fraud_data']['age_range']
        df_clean['age'] = df_clean['age'].clip(age_config[0], age_config[1])
        
        # Remove non-positive purchase values
        df_clean = df_clean[df_clean['purchase_value'] > 0]
        
        # Remove impossible timestamps
        df_clean = df_clean[df_clean['purchase_time'] >= df_clean['signup_time']]
        
        return df_clean
    
    def clean_creditcard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean credit card data based on validation results."""
        df_clean = df.copy()
        
        # Remove negative amounts
        df_clean = df_clean[df_clean['Amount'] >= 0]
        
        return df_clean