# task1-data-preprocessing/src/Data_Anlysis_Processing/geolocation_mapper.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import bisect
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeolocationResult:
    """Data class for geolocation mapping results."""
    mapped_df: pd.DataFrame
    country_stats: Dict[str, int]
    risk_scores: Dict[str, float]
    unmapped_count: int

class GeolocationMapper:
    """
    Efficient IP to country mapper using binary search.
    Optimized for large-scale IP range lookups.
    """
    
    def __init__(self, config_path: str = "config/config_data_analysis.yaml"):
        """Initialize mapper with configuration."""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.high_risk_countries = self.config['feature_engineering']['geolocation']['high_risk_countries']
        self.ip_ranges = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.countries = None
    
    def prepare_ip_ranges(self, ip_df: pd.DataFrame) -> None:
        """
        Prepare IP ranges for efficient lookup.
        
        Args:
            ip_df: DataFrame with lower_bound_ip_address, upper_bound_ip_address, country
        """
        # Sort by lower bound for binary search
        self.ip_ranges = ip_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
        self.lower_bounds = self.ip_ranges['lower_bound_ip_address'].values
        self.upper_bounds = self.ip_ranges['upper_bound_ip_address'].values
        self.countries = self.ip_ranges['country'].values
        
        logger.info(f"Prepared {len(self.ip_ranges)} IP ranges for binary search")
    
    def map_ip_to_country(self, ip_address: int) -> Optional[str]:
        """
        Map single IP address to country using binary search.
        
        Args:
            ip_address: Integer IP address
            
        Returns:
            Country name or None if not found
        """
        if self.lower_bounds is None:
            raise ValueError("IP ranges not prepared. Call prepare_ip_ranges first.")
        
        # Binary search for the right range
        idx = bisect.bisect_right(self.lower_bounds, ip_address) - 1
        
        if idx >= 0 and self.lower_bounds[idx] <= ip_address <= self.upper_bounds[idx]:
            return self.countries[idx]
        
        return None
    
    def batch_map_ips(self, ip_series: pd.Series) -> pd.Series:
        """
        Map multiple IP addresses to countries efficiently.
        
        Args:
            ip_series: Series of integer IP addresses
            
        Returns:
            Series of country names
        """
        if self.lower_bounds is None:
            raise ValueError("IP ranges not prepared. Call prepare_ip_ranges first.")
        
        # Vectorized mapping using numpy searchsorted
        indices = np.searchsorted(self.lower_bounds, ip_series.values, side='right') - 1
        
        # Create mask for valid indices
        valid_mask = (indices >= 0) & (ip_series.values <= self.upper_bounds[indices])
        
        # Create result array
        countries = np.full(len(ip_series), 'Unknown', dtype=object)
        countries[valid_mask] = self.countries[indices[valid_mask]]
        
        return pd.Series(countries, index=ip_series.index)
    
    def map_fraud_data(self, fraud_df: pd.DataFrame, 
                      ip_df: pd.DataFrame) -> GeolocationResult:
        """
        Map IP addresses in fraud data to countries.
        
        Args:
            fraud_df: Fraud dataframe
            ip_df: IP to country mapping dataframe
            
        Returns:
            GeolocationResult object
        """
        logger.info("Starting geolocation mapping...")
        
        # Prepare IP ranges
        self.prepare_ip_ranges(ip_df)
        
        # Map IP addresses
        fraud_df_copy = fraud_df.copy()
        fraud_df_copy['country'] = self.batch_map_ips(fraud_df_copy['ip_address'])
        
        # Calculate statistics
        country_counts = fraud_df_copy['country'].value_counts()
        fraud_by_country = fraud_df_copy.groupby('country')['class'].agg(['count', 'sum'])
        fraud_by_country['fraud_rate'] = (fraud_by_country['sum'] / fraud_by_country['count']) * 100
        
        # Calculate risk scores
        risk_scores = {}
        for country in fraud_by_country.index:
            if country != 'Unknown':
                fraud_rate = fraud_by_country.loc[country, 'fraud_rate']
                base_risk = min(fraud_rate / 10, 10)  # Scale 0-10
                
                # Adjust for high-risk countries
                if country in self.high_risk_countries:
                    base_risk *= 1.5
                
                risk_scores[country] = round(base_risk, 2)
        
        # Count unmapped IPs
        unmapped_count = (fraud_df_copy['country'] == 'Unknown').sum()
        
        # Log critical findings from YOUR data
        sample_ip = fraud_df_copy.iloc[0]['ip_address'] if len(fraud_df_copy) > 0 else None
        sample_country = fraud_df_copy.iloc[0]['country'] if len(fraud_df_copy) > 0 else None
        
        if sample_ip is not None:
            logger.info(f"Sample mapping: IP {sample_ip} → {sample_country}")
        
        # Find high-risk countries
        high_risk = fraud_by_country[fraud_by_country['fraud_rate'] > 5]
        if len(high_risk) > 0:
            logger.warning(f"🚨 High-risk countries identified: {high_risk.index.tolist()}")
        
        result = GeolocationResult(
            mapped_df=fraud_df_copy,
            country_stats=country_counts.to_dict(),
            risk_scores=risk_scores,
            unmapped_count=unmapped_count
        )
        
        logger.info(f"Geolocation mapping complete. "
                   f"Mapped {len(fraud_df_copy) - unmapped_count}/{len(fraud_df_copy)} IPs. "
                   f"Unmapped: {unmapped_count}")
        
        return result
    
    def detect_vpn_proxies(self, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect potential VPN/Proxy usage based on IP patterns.
        
        Args:
            fraud_df: Fraud dataframe with country mapping
            
        Returns:
            DataFrame with VPN detection flags
        """
        df = fraud_df.copy()
        
        # Common VPN/Proxy indicators
        # 1. IP ranges from hosting providers
        hosting_asns = ['DigitalOcean', 'Linode', 'AWS', 'Google Cloud', 
                       'Microsoft Azure', 'OVH', 'Hetzner']
        
        # 2. Multiple users from same IP
        ip_user_counts = df.groupby('ip_address')['user_id'].nunique()
        df['users_per_ip'] = df['ip_address'].map(ip_user_counts)
        
        # 3. IP changes in short time
        # (This would require user session tracking - simplified here)
        
        # Create VPN suspicion score
        df['vpn_suspicion_score'] = 0
        
        # Score based on hosting providers (simplified)
        # In reality, you'd use IP intelligence database
        
        # Score based on multiple users
        df.loc[df['users_per_ip'] > 5, 'vpn_suspicion_score'] += 3
        df.loc[df['users_per_ip'] > 10, 'vpn_suspicion_score'] += 2
        df.loc[df['users_per_ip'] > 20, 'vpn_suspicion_score'] += 5
        
        # Score based on country risk
        if hasattr(self, 'risk_scores'):
            df['country_risk_score'] = df['country'].map(self.risk_scores).fillna(0)
            df['vpn_suspicion_score'] += df['country_risk_score'] * 0.5
        
        # Flag high suspicion
        df['is_vpn_suspected'] = df['vpn_suspicion_score'] > 7
        
        logger.info(f"VPN detection: {df['is_vpn_suspected'].sum()} suspected VPN/proxy transactions")
        
        return df
    
    def generate_geolocation_report(self, result: GeolocationResult) -> Dict[str, Any]:
        """
        Generate comprehensive geolocation report.
        
        Args:
            result: Geolocation mapping result
            
        Returns:
            Dictionary with geolocation insights
        """
        df = result.mapped_df
        
        # Calculate fraud rates by country
        fraud_by_country = df.groupby('country').agg(
            total_transactions=('class', 'count'),
            fraud_transactions=('class', 'sum'),
            avg_purchase=('purchase_value', 'mean')
        ).reset_index()
        
        fraud_by_country['fraud_rate'] = (fraud_by_country['fraud_transactions'] / 
                                         fraud_by_country['total_transactions']) * 100
        
        # Top 10 countries by fraud rate
        top_fraud_countries = fraud_by_country.nlargest(10, 'fraud_rate')
        
        # Geographic patterns
        country_risk = {}
        for _, row in fraud_by_country.iterrows():
            risk_level = 'LOW'
            if row['fraud_rate'] > 10:
                risk_level = 'CRITICAL'
            elif row['fraud_rate'] > 5:
                risk_level = 'HIGH'
            elif row['fraud_rate'] > 2:
                risk_level = 'MEDIUM'
            
            country_risk[row['country']] = {
                'risk_level': risk_level,
                'fraud_rate': row['fraud_rate'],
                'total_transactions': row['total_transactions']
            }
        
        report = {
            'mapping_statistics': {
                'total_transactions': len(df),
                'mapped_transactions': len(df) - result.unmapped_count,
                'unmapped_percentage': (result.unmapped_count / len(df)) * 100,
                'unique_countries': df['country'].nunique()
            },
            'top_fraud_countries': top_fraud_countries.to_dict('records'),
            'country_risk_assessment': country_risk,
            'risk_scores': result.risk_scores,
            'geographic_insights': self._generate_geographic_insights(df),
            'recommendations': [
                f"Implement additional verification for transactions from {country}" 
                for country in top_fraud_countries.head(3)['country'].tolist()
            ]
        }
        
        return report
    
    def _generate_geographic_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate geographic insights for business recommendations."""
        insights = []
        
        # Group by country and calculate statistics
        stats_by_country = df.groupby('country').agg({
            'class': ['mean', 'sum', 'count'],
            'purchase_value': ['mean', 'std'],
            'time_since_signup_hours': 'mean'
        }).round(2)
        
        # Find anomalies
        high_value_fraud = stats_by_country[
            (stats_by_country[('class', 'mean')] > 0.1) & 
            (stats_by_country[('purchase_value', 'mean')] > 100)
        ]
        
        if len(high_value_fraud) > 0:
            insights.append(f"High-value fraud detected in: {high_value_fraud.index.tolist()}")
        
        # Find countries with immediate purchases
        if 'time_since_signup_hours' in df.columns:
            immediate_by_country = df[df['time_since_signup_hours'] < 1].groupby('country').size()
            if len(immediate_by_country) > 0:
                top_immediate = immediate_by_country.nlargest(3)
                insights.append(f"Immediate purchases common in: {top_immediate.index.tolist()}")
        
        return insights
    
    def map_ip_to_country_wrapper(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Wrapper method to map IP addresses for pipeline integration.
        
        Args:
            data_dict: Dictionary containing all datasets
            
        Returns:
            Dictionary with mapped fraud data
        """
        result = {}
        
        if 'fraud_data' in data_dict and 'ip_country_data' in data_dict:
            if data_dict['fraud_data'] is not None and data_dict['ip_country_data'] is not None:
                # Check if ip_address column exists
                if 'ip_address' in data_dict['fraud_data'].columns:
                    mapped_result = self.map_fraud_data(data_dict['fraud_data'], data_dict['ip_country_data'])
                    result['fraud_data'] = mapped_result.mapped_df
                    logger.info(f"Geolocation mapping completed for fraud data")
                else:
                    result['fraud_data'] = data_dict['fraud_data']
                    logger.warning("ip_address column not found in fraud data, skipping geolocation mapping")
            else:
                result['fraud_data'] = data_dict['fraud_data']
                logger.warning("Missing required datasets for geolocation mapping")
        else:
            result['fraud_data'] = data_dict.get('fraud_data')
            logger.warning("Required datasets not available for geolocation mapping")
        
        # Copy other datasets unchanged
        for key, df in data_dict.items():
            if key != 'fraud_data':
                result[key] = df
        
        return result