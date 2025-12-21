#!/usr/bin/env python3
"""
Working pipeline that loads and saves data without complex dependencies
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simple_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleDataLoader:
    """Simple data loader without complex dependencies"""
    
    def __init__(self, config):
        self.config = config
        self.paths = config.get('paths', {})
        self.datasets = config.get('datasets', {})
        
    def load_all_datasets(self):
        """Load all datasets"""
        datasets = {}
        
        raw_data_dir = Path(self.paths.get('raw_data', 'data/raw'))
        
        for name, filename in self.datasets.items():
            file_path = raw_data_dir / filename
            if file_path.exists():
                try:
                    logger.info(f"Loading {name} from {file_path}")
                    
                    # Special handling for fraud data
                    if name == 'fraud_data':
                        df = pd.read_csv(file_path, parse_dates=['signup_time', 'purchase_time'])
                    else:
                        df = pd.read_csv(file_path)
                    
                    datasets[name] = df
                    logger.info(f"  Loaded {name}: {df.shape}")
                    
                except Exception as e:
                    logger.error(f"  Error loading {name}: {e}")
                    datasets[name] = None
            else:
                logger.error(f"  File not found: {file_path}")
                datasets[name] = None
        
        return datasets

class SimpleDataValidator:
    """Simple data validator"""
    
    def __init__(self, config):
        self.config = config
    
    def clean_fraud_data(self, df):
        """Simple cleaning for fraud data"""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Remove negative purchase values
        if 'purchase_value' in df_clean.columns:
            df_clean = df_clean[df_clean['purchase_value'] > 0]
        
        return df_clean
    
    def clean_creditcard_data(self, df):
        """Simple cleaning for credit card data"""
        df_clean = df.copy()
        
        # Remove negative amounts
        if 'Amount' in df_clean.columns:
            df_clean = df_clean[df_clean['Amount'] >= 0]
        
        return df_clean

def run_pipeline():
    """Run the complete simple pipeline"""
    
    print("="*60)
    print("SIMPLE WORKING PIPELINE")
    print("="*60)
    
    # 1. Load config
    print("\n1. Loading configuration...")
    config_path = "config/config_data_analysis.yaml"
    
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded")
    
    # 2. Setup paths
    paths = config.get('paths', {})
    default_paths = {
        'raw_data': 'data/raw',
        'processed_data': 'data/processed',
        'visualizations': 'outputs/Data_Analysis_Processing/visualizations',
        'reports': 'outputs/Data_Analysis_Processing/reports',
        'statistics': 'outputs/Data_Analysis_Processing/statistics'
    }
    
    for key, default in default_paths.items():
        if key not in paths:
            paths[key] = default
    
    # Create directories
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # 3. Load data
    print("\n2. Loading data...")
    loader = SimpleDataLoader(config)
    datasets = loader.load_all_datasets()
    
    # 4. Clean data
    print("\n3. Cleaning data...")
    validator = SimpleDataValidator(config)
    
    cleaned_data = {}
    for name, df in datasets.items():
        if df is not None:
            if name == 'fraud_data':
                cleaned_data[name] = validator.clean_fraud_data(df)
            elif name == 'creditcard_data':
                cleaned_data[name] = validator.clean_creditcard_data(df)
            else:
                cleaned_data[name] = df
            print(f"  ✓ Cleaned {name}: {cleaned_data[name].shape}")
    
    # 5. Save processed data
    print("\n4. Saving processed data...")
    processed_dir = Path(paths['processed_data'])
    processed_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, df in cleaned_data.items():
        if df is not None:
            # Save versioned file
            versioned_file = processed_dir / f"{name}_processed_{timestamp}.csv"
            df.to_csv(versioned_file, index=False)
            
            # Save latest file
            latest_file = processed_dir / f"{name}_processed_latest.csv"
            df.to_csv(latest_file, index=False)
            
            processed_files.append(versioned_file)
            print(f"  ✓ Saved {name}: {versioned_file.name}")
    
    # 6. Generate summary
    print("\n5. Generating summary...")
    summary = {}
    for name, df in cleaned_data.items():
        if df is not None:
            summary[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'sample_size': len(df),
                'missing_values': df.isnull().sum().sum()
            }
    
    # Save summary
    stats_dir = Path(paths['statistics'])
    summary_file = stats_dir / "data_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"  ✓ Summary saved: {summary_file}")
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"\nProcessed data saved to: {processed_dir}")
    print(f"Summary saved to: {summary_file}")
    
    if processed_files:
        print("\nFiles created:")
        for file in processed_files:
            print(f"  • {file.name}")
    
    return True

if __name__ == "__main__":
    success = run_pipeline()
    
    if success:
        print("\n✅ NEXT STEPS:")
        print("  1. Check data/processed/ for your cleaned data")
        print("  2. Run EDA notebook for analysis")
        print("  3. Check outputs/Data_Analysis_Processing/statistics/data_summary.json")
    else:
        print("\n❌ Pipeline failed")
        sys.exit(1)