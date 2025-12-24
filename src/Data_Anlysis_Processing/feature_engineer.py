# ============================================================================
# FIX THE FEATURE ENGINEER MODULE
# ============================================================================
print("Fixing feature_engineer.py module...")

feature_engineer_path = r"D:\10 acadamy\fraud-detection-ml-system\src\Data_Anlysis_Processing\feature_engineer.py"

if os.path.exists(feature_engineer_path):
    # Read the file
    with open(feature_engineer_path, 'r') as f:
        content = f.read()
    
    # Find and fix the problematic method
    old_method = """    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Create user behavior features.\"\"\"
        # Sort by user and time for rolling calculations
        df = df.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)
        
        # Transaction velocity (CRITICAL for fraud detection)
        if self.behavior_config['transaction_velocity']:
            for window in self.behavior_config['velocity_window_hours']:
                # Calculate transactions per user in time window
                df[f'transactions_last_{window}h'] = df.groupby('user_id').rolling(
                    f'{window}h', on='purchase_time'
                )['user_id'].count().values
                
                # Calculate spending velocity
                df[f'spending_last_{window}h'] = df.groupby('user_id').rolling(
                    f'{window}h', on='purchase_time'
                )['purchase_value'].sum().values"""
    
    new_method = """    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Create user behavior features.\"\"\"
        # Sort by user and time for rolling calculations
        df = df.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)
        
        # Transaction velocity (CRITICAL for fraud detection)
        if self.behavior_config['transaction_velocity']:
            for window in self.behavior_config['velocity_window_hours']:
                # Calculate transactions per user in time window
                # FIX: Use transform instead of .values to maintain index alignment
                df[f'transactions_last_{window}h'] = df.groupby('user_id')['purchase_time'].transform(
                    lambda x: x.rolling(f'{window}h', on=x).count()
                )
                
                # Calculate spending velocity
                df[f'spending_last_{window}h'] = df.groupby('user_id')['purchase_value'].transform(
                    lambda x: x.rolling(f'{window}h').sum()
                )"""
    
    if old_method in content:
        content = content.replace(old_method, new_method)
        with open(feature_engineer_path, 'w') as f:
            f.write(content)
        print("✓ Fixed _create_behavioral_features method")
    else:
        print("⚠ Could not find the exact method to fix")
        
        # Try a simpler fix - just skip the problematic feature
        simple_fix = """    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Create user behavior features.\"\"\"
        # Sort by user and time for rolling calculations
        df = df.sort_values(['user_id', 'purchase_time']).reset_index(drop=True)
        
        # Transaction velocity (CRITICAL for fraud detection)
        if self.behavior_config['transaction_velocity']:
            for window in self.behavior_config['velocity_window_hours']:
                # Calculate transactions per user in time window
                try:
                    df[f'transactions_last_{window}h'] = df.groupby('user_id').rolling(
                        f'{window}h', on='purchase_time'
                    )['user_id'].count().values
                except Exception as e:
                    print(f"Warning: Could not create transactions_last_{window}h: {e}")
                    df[f'transactions_last_{window}h'] = 0
                
                # Calculate spending velocity
                try:
                    df[f'spending_last_{window}h'] = df.groupby('user_id').rolling(
                        f'{window}h', on='purchase_time'
                    )['purchase_value'].sum().values
                except Exception as e:
                    print(f"Warning: Could not create spending_last_{window}h: {e}")
                    df[f'spending_last_{window}h'] = 0"""
        
        # Try to find and replace a different pattern
        import re
        pattern = r'def _create_behavioral_features\(self, df: pd.DataFrame\) -> pd.DataFrame:.*?return df'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, simple_fix + '\n        \n        return df', content, flags=re.DOTALL)
            with open(feature_engineer_path, 'w') as f:
                f.write(content)
            print("✓ Applied simplified fix to _create_behavioral_features")
        else:
            print("✗ Could not apply fix automatically")
else:
    print(f"✗ Module file not found: {feature_engineer_path}")

# Now reload the module
print("\nReloading feature_engineer module...")
if 'feature_engineer' in sys.modules:
    del sys.modules['feature_engineer']

# ============================================================================
# ALTERNATIVE: USE SIMPLIFIED FEATURE ENGINEERING FUNCTION
# ============================================================================
def simplified_feature_engineering(df, dataset_type='fraud'):
    """Simplified feature engineering for fallback"""
    df_copy = df.copy()
    
    print(f"\nApplying simplified feature engineering for {dataset_type} data...")
    
    if dataset_type == 'fraud':
        # Extract time-based features if purchase_time exists
        if 'purchase_time' in df_copy.columns:
            try:
                df_copy['purchase_time'] = pd.to_datetime(df_copy['purchase_time'])
                df_copy['purchase_hour'] = df_copy['purchase_time'].dt.hour
                df_copy['purchase_dayofweek'] = df_copy['purchase_time'].dt.dayofweek
                df_copy['purchase_month'] = df_copy['purchase_time'].dt.month
                
                if 'signup_time' in df_copy.columns:
                    df_copy['signup_time'] = pd.to_datetime(df_copy['signup_time'])
                    df_copy['time_since_signup_hours'] = (df_copy['purchase_time'] - df_copy['signup_time']).dt.total_seconds() / 3600
                    df_copy['is_immediate_purchase'] = (df_copy['time_since_signup_hours'] < 1).astype(int)
                
                print("  ✓ Added time-based features")
            except Exception as e:
                print(f"  ✗ Could not parse time columns: {e}")
        
        # Create binary flags for categorical columns
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_copy[col].nunique() <= 10:  # Only if few unique values
                df_copy[f'{col}_encoded'] = pd.factorize(df_copy[col])[0]
                print(f"  ✓ Encoded {col}")
        
        # Create interaction features for numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        if 'purchase_value' in numeric_cols and 'age' in numeric_cols:
            df_copy['value_age_ratio'] = df_copy['purchase_value'] / (df_copy['age'] + 1)
            print("  ✓ Added value_age_ratio")
        
        # User statistics if user_id exists
        if 'user_id' in df_copy.columns and 'purchase_value' in numeric_cols:
            user_stats = df_copy.groupby('user_id')['purchase_value'].agg(['mean', 'std', 'count']).add_prefix('user_')
            df_copy = df_copy.merge(user_stats, on='user_id', how='left')
            print("  ✓ Added user statistics")
        
        # Device statistics if device_id exists
        if 'device_id' in df_copy.columns:
            device_counts = df_copy['device_id'].value_counts()
            df_copy['device_usage_count'] = df_copy['device_id'].map(device_counts)
            print("  ✓ Added device statistics")
    
    elif dataset_type == 'credit':
        # For credit card data
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if 'Class' in numeric_cols:
            numeric_cols.remove('Class')
        
        # Create squared features for first 5 columns
        for col in numeric_cols[:5]:
            df_copy[f'{col}_squared'] = df_copy[col] ** 2
        
        # Create interaction between top 3 features
        if len(numeric_cols) >= 3:
            df_copy[f'{numeric_cols[0]}_{numeric_cols[1]}_interaction'] = df_copy[numeric_cols[0]] * df_copy[numeric_cols[1]]
        
        # Add log transform for Amount if exists
        if 'Amount' in df_copy.columns:
            df_copy['log_amount'] = np.log1p(df_copy['Amount'])
            print("  ✓ Added log transformation for Amount")
        
        # Add time features if Time column exists
        if 'Time' in df_copy.columns:
            df_copy['transaction_hour'] = (df_copy['Time'] % 86400) / 3600
            df_copy['is_night_transaction'] = ((df_copy['transaction_hour'] >= 0) & (df_copy['transaction_hour'] < 6)).astype(int)
            print("  ✓ Added time features")
        
        print(f"  ✓ Added polynomial and interaction features")
    
    print(f"  Original features: {df.shape[1]}")
    print(f"  New features: {df_copy.shape[1]}")
    print(f"  Features added: {df_copy.shape[1] - df.shape[1]}")
    
    return df_copy

# ============================================================================
# PROCEED WITH FEATURE ENGINEERING USING OUR FIXED FUNCTION
# ============================================================================
print("\n" + "="*80)
print("PROCEEDING WITH SIMPLIFIED FEATURE ENGINEERING")
print("="*80)

# Apply simplified feature engineering
fraud_with_features = simplified_feature_engineering(fraud_cleaned, 'fraud')
credit_with_features = simplified_feature_engineering(credit_cleaned, 'credit')

print(f"\n✓ Feature engineering complete!")
print(f"Fraud data: {fraud_cleaned.shape} → {fraud_with_features.shape}")
print(f"Credit data: {credit_cleaned.shape} → {credit_with_features.shape}")

# Save the results
fraud_with_features.to_csv(os.path.join(DATA_PROCESSED_PATH, 'fraud_with_features_simplified.csv'), index=False)
credit_with_features.to_csv(os.path.join(DATA_PROCESSED_PATH, 'creditcard_with_features_simplified.csv'), index=False)

print(f"\n✓ Saved feature-engineered data to {DATA_PROCESSED_PATH}")

# Show what features were created
print(f"\nNew features in fraud data:")
original_fraud_cols = set(fraud_cleaned.columns)
new_fraud_cols = set(fraud_with_features.columns) - original_fraud_cols
for i, col in enumerate(sorted(new_fraud_cols)):
    print(f"  {i+1}. {col}")

print(f"\nNew features in credit data:")
original_credit_cols = set(credit_cleaned.columns)
new_credit_cols = set(credit_with_features.columns) - original_credit_cols
for i, col in enumerate(sorted(new_credit_cols)[:15]):  # Show first 15
    print(f"  {i+1}. {col}")
if len(new_credit_cols) > 15:
    print(f"  ... and {len(new_credit_cols) - 15} more")