# src/Data_Anlysis_Processing/data_loader.py

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Comprehensive data loader for fraud detection datasets.
    Handles loading, basic validation, and data type conversion.
    """
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config: Either a string path to config file or a config dictionary
        """
        if isinstance(config, str):
            # Config is a file path
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
            self.config_path = config
        elif isinstance(config, dict):
            # Config is already a dictionary
            self.config = config
            self.config_path = None
        else:
            raise TypeError("config must be either a string path or a dictionary")
        
        self.data_paths = self.config.get('paths', {})
        self.dataset_names = self.config.get('datasets', {})
        
        # Set default paths if not provided
        if not self.data_paths:
            self.data_paths = {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed'
            }
        
        # Initialize data containers
        self.fraud_data = None
        self.creditcard_data = None
        self.ip_country_data = None
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all three datasets.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        datasets = {}
        
        # Load Fraud_Data.csv
        if 'fraud_data' in self.dataset_names:
            fraud_path = os.path.join(self.data_paths['raw_data'], 
                                     self.dataset_names['fraud_data'])
            self.fraud_data = self._load_fraud_data(fraud_path)
            datasets['fraud_data'] = self.fraud_data
        
        # Load creditcard.csv
        if 'creditcard_data' in self.dataset_names:
            credit_path = os.path.join(self.data_paths['raw_data'],
                                      self.dataset_names['creditcard_data'])
            self.creditcard_data = self._load_creditcard_data(credit_path)
            datasets['creditcard_data'] = self.creditcard_data
        
        # Load IP to Country mapping
        if 'ip_country_data' in self.dataset_names:
            ip_path = os.path.join(self.data_paths['raw_data'],
                                  self.dataset_names['ip_country_data'])
            self.ip_country_data = self._load_ip_country_data(ip_path)
            datasets['ip_country_data'] = self.ip_country_data
        
        return datasets
    
    def _load_fraud_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess fraud data.
        
        Args:
            filepath: Path to Fraud_Data.csv
            
        Returns:
            Preprocessed fraud dataframe
        """
        try:
            # Load the data
            df = pd.read_csv(filepath)
            
            # Convert timestamps - YOUR DATA FORMAT
            if 'signup_time' in df.columns:
                df['signup_time'] = pd.to_datetime(df['signup_time'], 
                                                  format='%m/%d/%Y %H:%M', errors='coerce')
            if 'purchase_time' in df.columns:
                df['purchase_time'] = pd.to_datetime(df['purchase_time'], 
                                                    format='%m/%d/%Y %H:%M', errors='coerce')
            
            # Convert IP address to integer if it exists
            if 'ip_address' in df.columns:
                try:
                    df['ip_address'] = df['ip_address'].astype(int)
                except:
                    logger.warning("Could not convert ip_address to int")
            
            # Standardize browser names if browser column exists
            if 'browser' in df.columns:
                browser_config = self.config.get('data_cleaning', {}).get('fraud_data', {}).get('browser_standardization', {})
                if browser_config:
                    df['browser'] = self._standardize_browser(df['browser'], browser_config)
            
            logger.info(f"Fraud data loaded: {len(df)} records")
            
            if 'class' in df.columns:
                fraud_count = df['class'].sum()
                logger.info(f"Fraud cases: {fraud_count}")
                
                # Log critical insight from YOUR data
                if 'signup_time' in df.columns and 'purchase_time' in df.columns:
                    immediate_fraud = df[df['signup_time'] == df['purchase_time']]
                    if len(immediate_fraud) > 0:
                        logger.warning(f"Found {len(immediate_fraud)} transactions made INSTANTLY after signup!")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading fraud data: {e}")
            return pd.DataFrame()
    
    def _load_creditcard_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and preprocess credit card data.
        
        Args:
            filepath: Path to creditcard.csv
            
        Returns:
            Preprocessed credit card dataframe
        """
        try:
            df = pd.read_csv(filepath)
            
            # Log class distribution
            if 'Class' in df.columns:
                fraud_count = df['Class'].sum()
                total_count = len(df)
                fraud_percentage = (fraud_count / total_count) * 100
                
                logger.info(f"Credit card fraud: {fraud_count}/{total_count} ({fraud_percentage:.3f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading credit card data: {e}")
            return pd.DataFrame()
    
    def _load_ip_country_data(self, filepath: str) -> pd.DataFrame:
        """
        Load IP to country mapping data.
        
        Args:
            filepath: Path to IpAddress_to_Country.csv
            
        Returns:
            IP country mapping dataframe
        """
        try:
            df = pd.read_csv(filepath)
            
            # Ensure correct data types
            if 'lower_bound_ip_address' in df.columns:
                df['lower_bound_ip_address'] = df['lower_bound_ip_address'].astype('int64')
            if 'upper_bound_ip_address' in df.columns:
                df['upper_bound_ip_address'] = df['upper_bound_ip_address'].astype('int64')
            
            # Sort for efficient searching
            df = df.sort_values('lower_bound_ip_address').reset_index(drop=True)
            
            logger.info(f"IP ranges loaded: {len(df)} country ranges")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading IP country data: {e}")
            return pd.DataFrame()
    
    def _standardize_browser(self, browser_series: pd.Series, 
                           config: Dict) -> pd.Series:
        """
        Standardize browser names.
        
        Args:
            browser_series: Raw browser column
            config: Browser standardization configuration
            
        Returns:
            Standardized browser names
        """
        # Create mapping dictionary
        browser_map = {}
        
        # Add all variants to mapping
        for browser, variants in config.items():
            for variant in variants:
                browser_map[variant.lower()] = browser.replace('_variants', '')
        
        # Apply mapping
        standardized = browser_series.copy()
        standardized = standardized.str.lower().map(browser_map).fillna(browser_series)
        
        return standardized
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics for all datasets.
        
        Returns:
            Dictionary with data summaries
        """
        summary = {
            'fraud_data': {
                'shape': self.fraud_data.shape if self.fraud_data is not None else None,
                'columns': list(self.fraud_data.columns) if self.fraud_data is not None else [],
            },
            'creditcard_data': {
                'shape': self.creditcard_data.shape if self.creditcard_data is not None else None,
                'columns': list(self.creditcard_data.columns) if self.creditcard_data is not None else [],
            },
            'ip_country_data': {
                'shape': self.ip_country_data.shape if self.ip_country_data is not None else None,
                'columns': list(self.ip_country_data.columns) if self.ip_country_data is not None else [],
            }
        }
        
        return summary