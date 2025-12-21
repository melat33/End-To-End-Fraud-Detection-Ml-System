#!/usr/bin/env python3
"""
Run the complete Task 1 pipeline
Usage: python scripts/run1_modules.py
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

# First, create the logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Setup logging WITHOUT emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task1_pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # Use stdout which handles Unicode better
    ]
)
logger = logging.getLogger(__name__)

# Define simple text symbols (no emojis)
SYMBOLS = {
    'check': '[OK]',
    'error': '[ERROR]',
    'warning': '[WARNING]',
    'folder': '[DIR]',
    'rocket': '[START]',
    'disk': '[SAVE]',
    'data': '[DATA]'
}

def import_modules():
    """Import all required modules"""
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        
        from Data_Anlysis_Processing.data_loader import DataLoader
        from Data_Anlysis_Processing.data_validator import DataValidator
        from Data_Anlysis_Processing.geolocation_mapper import GeolocationMapper
        from Data_Anlysis_Processing.feature_engineer import FeatureEngineer
        
        # Try to import imbalance_handler but handle the error gracefully
        try:
            from Data_Anlysis_Processing.imbalance_handler import ImbalanceHandler
            imbalance_available = True
        except ImportError as e:
            logger.warning(f"ImbalanceHandler not available: {e}")
            ImbalanceHandler = None
            imbalance_available = False
        
        # Try to import visualizer
        try:
            from Data_Anlysis_Processing.visualizer import Visualizer
            visualizer_available = True
        except ImportError as e:
            logger.warning(f"Visualizer not available: {e}")
            Visualizer = None
            visualizer_available = False
        
        logger.info(f"{SYMBOLS['check']} All modules imported successfully")
        return DataLoader, DataValidator, GeolocationMapper, FeatureEngineer, ImbalanceHandler, Visualizer
        
    except ImportError as e:
        logger.error(f"{SYMBOLS['error']} Failed to import modules: {e}")
        logger.error("Make sure your modules are in src/Data_Anlysis_Processing/")
        logger.error("Available modules in src/Data_Anlysis_Processing/:")
        src_dir = Path("src/Data_Anlysis_Processing")
        if src_dir.exists():
            for file in src_dir.glob("*.py"):
                logger.error(f"  • {file.name}")
        else:
            logger.error(f"  Directory not found: {src_dir}")
        raise

def create_main_pipeline():
    """Create a simple pipeline runner"""
    
    class DataPreprocessingPipeline:
        def __init__(self, config_path: str = "config/config_data_analysis.yaml"):
            self.config_path = config_path
            self.load_config()
            self.setup_paths()
            self.processed_files = {}
            
        def load_config(self):
            """Load configuration from YAML"""
            try:
                config_file = Path(self.config_path)
                if not config_file.exists():
                    # Try to find config
                    alt_configs = [
                        Path("config/config_task1.yaml"),
                        Path("config/config.yaml"),
                        Path("config/task1.yaml")
                    ]
                    
                    for alt in alt_configs:
                        if alt.exists():
                            self.config_path = str(alt)
                            logger.info(f"Using alternative config: {alt}")
                            break
                    else:
                        raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"{SYMBOLS['check']} Configuration loaded from {self.config_path}")
                
            except Exception as e:
                logger.error(f"{SYMBOLS['error']} Failed to load config: {e}")
                logger.error("Creating default config...")
                self.create_default_config()
                
        def create_default_config(self):
            """Create default config if file doesn't exist"""
            default_config = {
                'paths': {
                    'raw_data': 'data/raw',
                    'processed_data': 'data/processed',
                    'visualizations': 'outputs/Data_Analysis_Processing/visualizations',
                    'reports': 'outputs/Data_Analysis_Processing/reports',
                    'statistics': 'outputs/Data_Analysis_Processing/statistics'
                },
                'datasets': {
                    'fraud_data': 'Fraud_Data.csv',
                    'creditcard_data': 'creditcard.csv',
                    'ip_country_data': 'IpAddress_to_Country.csv'
                },
                'imbalance_handling': {
                    'method': 'none'  # Disable by default due to compatibility issues
                }
            }
            self.config = default_config
            logger.info(f"{SYMBOLS['check']} Using default configuration")
            
            # Save default config
            config_dir = Path("config")
            config_dir.mkdir(exist_ok=True)
            with open(config_dir / "config_data_analysis.yaml", 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"{SYMBOLS['disk']} Default config saved to {config_dir / 'config_data_analysis.yaml'}")
        
        def setup_paths(self):
            """Create output directories"""
            self.paths = self.config.get('paths', {})
            
            # Default paths if not in config
            default_paths = {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'visualizations': 'outputs/Data_Analysis_Processing/visualizations',
                'reports': 'outputs/Data_Analysis_Processing/reports',
                'statistics': 'outputs/Data_Analysis_Processing/statistics'
            }
            
            for key, default in default_paths.items():
                if key not in self.paths:
                    self.paths[key] = default
            
            # Create directories
            for key, path in self.paths.items():
                dir_path = Path(path)
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"{SYMBOLS['folder']} Directory ready: {dir_path}")
        
        def run(self):
            """Execute the complete pipeline"""
            logger.info(f"{SYMBOLS['rocket']} Starting Data Preprocessing Pipeline")
            
            try:
                DataLoader, DataValidator, GeolocationMapper, FeatureEngineer, ImbalanceHandler, Visualizer = import_modules()
                
                # Step 1: Load data
                logger.info("1. Loading data...")
                loader = DataLoader(self.config)
                datasets = loader.load_all_datasets()
                
                if not datasets:
                    logger.error(f"{SYMBOLS['error']} No datasets loaded!")
                    return None
                
                logger.info(f"{SYMBOLS['data']} Datasets loaded: {list(datasets.keys())}")
                for name, df in datasets.items():
                    if df is not None:
                        logger.info(f"   • {name}: {df.shape}")
                    else:
                        logger.warning(f"   • {name}: Failed to load")
                
                # Step 2: Clean and validate
                logger.info("2. Cleaning and validating data...")
                validator = DataValidator(self.config)
                
                cleaned_data = {}
                for name, df in datasets.items():
                    if df is not None:
                        try:
                            if name == 'fraud_data':
                                cleaned_data[name] = validator.clean_fraud_data(df)
                            elif name == 'creditcard_data':
                                cleaned_data[name] = validator.clean_creditcard_data(df)
                            else:
                                cleaned_data[name] = df
                            logger.info(f"   • Cleaned {name}: {cleaned_data[name].shape}")
                        except Exception as e:
                            logger.error(f"   {SYMBOLS['error']} Failed to clean {name}: {e}")
                            cleaned_data[name] = df
                
                # Step 3: Map geolocation (if we have the data)
                logger.info("3. Mapping geolocation...")
                if 'fraud_data' in cleaned_data and 'ip_country_data' in cleaned_data:
                    try:
                        geo_mapper = GeolocationMapper(self.config)
                        geo_data = geo_mapper.map_ip_to_country_wrapper(cleaned_data)
                        logger.info(f"   {SYMBOLS['check']} Geolocation mapping completed")
                    except Exception as e:
                        logger.error(f"   {SYMBOLS['error']} Geolocation mapping failed: {e}")
                        geo_data = cleaned_data
                else:
                    geo_data = cleaned_data
                    logger.info("   [SKIP] Skipped geolocation (missing required data)")
                
                # Step 4: Feature engineering
                logger.info("4. Engineering features...")
                if 'fraud_data' in geo_data:
                    try:
                        feature_engineer = FeatureEngineer(self.config)
                        engineered_data = feature_engineer.create_features(geo_data)
                        logger.info(f"   {SYMBOLS['check']} Features engineered: {engineered_data['fraud_data'].shape}")
                    except Exception as e:
                        logger.error(f"   {SYMBOLS['error']} Feature engineering failed: {e}")
                        engineered_data = geo_data
                else:
                    engineered_data = geo_data
                    logger.info("   [SKIP] Skipped feature engineering (no fraud data)")
                
                # Step 5: Handle imbalance (if configured and possible)
                if (self.config.get('imbalance_handling', {}).get('method') != 'none' and 
                    'fraud_data' in engineered_data and ImbalanceHandler is not None):
                    logger.info("5. Handling class imbalance...")
                    try:
                        imbalance_handler = ImbalanceHandler(self.config)
                        balanced_data = imbalance_handler.handle_imbalance(engineered_data)
                        logger.info(f"   {SYMBOLS['check']} Class imbalance handled")
                    except Exception as e:
                        logger.error(f"   {SYMBOLS['error']} Imbalance handling failed: {e}")
                        balanced_data = engineered_data
                else:
                    balanced_data = engineered_data
                    logger.info("5. [SKIP] Skipping imbalance handling")
                
                # Step 6: Save processed data
                logger.info("6. Saving processed data...")
                self.save_processed_data(balanced_data)
                
                # Step 7: Generate visualizations (if module exists)
                logger.info("7. Generating visualizations...")
                if Visualizer is not None:
                    try:
                        visualizer = Visualizer(self.config)
                        visualizer.generate_all_visualizations(balanced_data)
                        logger.info(f"   {SYMBOLS['check']} Visualizations generated")
                    except Exception as e:
                        logger.warning(f"   [SKIP] Visualizations skipped: {e}")
                else:
                    logger.info("   [SKIP] Visualizer module not available")
                
                logger.info(f"{SYMBOLS['check']} Pipeline completed successfully!")
                return self.processed_files
                
            except Exception as e:
                logger.error(f"{SYMBOLS['error']} Pipeline failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                raise
        
        def save_processed_data(self, data_dict):
            """Save all processed datasets"""
            processed_dir = Path(self.paths['processed_data'])
            
            for name, data in data_dict.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Save versioned file
                    versioned_file = processed_dir / f"{name}_processed_{timestamp}.csv"
                    try:
                        data.to_csv(versioned_file, index=False)
                        
                        # Save latest file
                        latest_file = processed_dir / f"{name}_processed_latest.csv"
                        data.to_csv(latest_file, index=False)
                        
                        self.processed_files[name] = {
                            'versioned': str(versioned_file),
                            'latest': str(latest_file),
                            'shape': data.shape,
                            'columns': list(data.columns)
                        }
                        
                        logger.info(f"{SYMBOLS['disk']} Saved {name}: {versioned_file} (Shape: {data.shape})")
                        
                    except Exception as e:
                        logger.error(f"{SYMBOLS['error']} Failed to save {name}: {e}")
                        
                elif data is not None:
                    logger.info(f"[INFO] {name}: Not a DataFrame or empty")
                else:
                    logger.info(f"[INFO] {name}: No data to save")
    
    return DataPreprocessingPipeline

def main():
    """Main function to run the pipeline"""
    
    print("="*60)
    print("FRAUD DETECTION - TASK 1 PIPELINE")
    print("="*60)
    
    try:
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("outputs/Data_Analysis_Processing").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        # Get the pipeline class
        PipelineClass = create_main_pipeline()
        
        # Initialize pipeline
        pipeline = PipelineClass("config/config_data_analysis.yaml")
        
        # Run pipeline
        print("\nStarting pipeline execution...")
        processed_files = pipeline.run()
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        if processed_files:
            print("\nPROCESSED DATA:")
            for name, info in processed_files.items():
                print(f"  • {name}:")
                print(f"    Shape: {info['shape']}")
                print(f"    Features: {len(info['columns'])}")
                print(f"    File: {info['latest']}")
        else:
            print("\n[WARNING] No processed files were saved")
        
        print("\nOUTPUT DIRECTORIES:")
        for key, path in pipeline.paths.items():
            dir_path = Path(path)
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print(f"  • {key}: {path} ({file_count} files)")
            else:
                print(f"  • {key}: {path} (not created)")
        
        print("\nNEXT STEPS:")
        print("  1. Check data/processed/ for cleaned data")
        print("  2. Check logs/task1_pipeline.log for details")
        print("  3. Run EDA notebook for analysis")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] PIPELINE FAILED: {e}")
        print("Check logs/task1_pipeline.log for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)