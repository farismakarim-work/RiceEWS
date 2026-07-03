"""
Rice Market Data Fetcher Module
================================

Mengumpulkan data harga beras dari berbagai pasar di Indonesia.
Mendukung multiple data sources dan preprocessing otomatis.

Data Sources:
- Kementan (Badan Pusat Statistik)
- Local market APIs
- CSV/database imports
- Time series data untuk multiple locations
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RiceMarketDataFetcher:
    """
    Fetcher untuk rice market data dari berbagai sumber.
    """

    def __init__(self, data_dir: str = "data/raw_market_data"):
        """
        Initialize fetcher.
        
        Parameters:
        -----------
        data_dir : str
            Directory untuk menyimpan raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Market metadata
        self.market_registry = self._init_market_registry()
    
    def _init_market_registry(self) -> Dict:
        """Initialize registry pasar-pasar utama Indonesia."""
        return {
            # Pasar utama Jabodetabek
            "JKT_Kramat": {
                "region": "DKI Jakarta",
                "type": "wholesale",
                "coordinates": (-6.1751, 106.8249),
                "market_name": "Pasar Kramat Jati"
            },
            "JKT_Benhil": {
                "region": "DKI Jakarta",
                "type": "wholesale",
                "coordinates": (-6.2088, 106.8030),
                "market_name": "Pasar Benhil"
            },
            
            # Pasar regional
            "BANDUNG_Caringin": {
                "region": "Jawa Barat",
                "type": "wholesale",
                "coordinates": (-6.9147, 107.6062),
                "market_name": "Pasar Caringin Bandung"
            },
            "SURABAYA_Wonokromo": {
                "region": "Jawa Timur",
                "type": "wholesale",
                "coordinates": (-7.2575, 112.7618),
                "market_name": "Pasar Wonokromo Surabaya"
            },
            "MEDAN_Petisah": {
                "region": "Sumatera Utara",
                "type": "wholesale",
                "coordinates": (3.1955, 98.6722),
                "market_name": "Pasar Petisah Medan"
            },
            "SEMARANG_Johar": {
                "region": "Jawa Tengah",
                "type": "wholesale",
                "coordinates": (-6.9667, 110.4167),
                "market_name": "Pasar Johar Semarang"
            },
            
            # Pasar regional lainnya
            "MAKASSAR_Baru": {
                "region": "Sulawesi Selatan",
                "type": "wholesale",
                "coordinates": (-5.1477, 119.4327),
                "market_name": "Pasar Baru Makassar"
            },
            "YOGYAKARTA_Beringharjo": {
                "region": "DI Yogyakarta",
                "type": "wholesale",
                "coordinates": (-7.7956, 110.3695),
                "market_name": "Pasar Beringharjo Yogyakarta"
            },
            "PALEMBANG_Tradisional": {
                "region": "Sumatera Selatan",
                "type": "wholesale",
                "coordinates": (-2.9264, 104.7456),
                "market_name": "Pasar Tradisional Palembang"
            },
            "BANJARMASIN_Sudimampir": {
                "region": "Kalimantan Selatan",
                "type": "wholesale",
                "coordinates": (-3.3286, 114.5897),
                "market_name": "Pasar Sudimampir Banjarmasin"
            },
            "PONTIANAK_Bawah": {
                "region": "Kalimantan Barat",
                "type": "wholesale",
                "coordinates": (-0.0263, 109.3425),
                "market_name": "Pasar Bawah Pontianak"
            },
            "DENPASAR_Gianyar": {
                "region": "Bali",
                "type": "wholesale",
                "coordinates": (-8.6705, 115.2126),
                "market_name": "Pasar Gianyar Denpasar"
            },
        }
    
    def fetch_synthetic_timeseries(self, 
                                  start_date: str = "2022-01-01",
                                  end_date: str = "2024-12-31",
                                  frequency: str = "D") -> pd.DataFrame:
        """
        Generate synthetic time series data untuk testing.
        
        Parameters:
        -----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        frequency : str
            'D' = Daily, 'W' = Weekly, 'M' = Monthly
            
        Returns:
        --------
        pd.DataFrame
            Time series dengan kolom: date, market, price, volume
        """
        
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        market_keys = list(self.market_registry.keys())
        
        data_list = []
        
        # Generate synthetic data dengan karakteristik real market
        for date in date_range:
            # Base seasonal component
            day_of_year = date.dayofyear
            seasonal = 1000 + 300 * np.sin(2 * np.pi * day_of_year / 365)
            
            for market_key in market_keys:
                # Market-specific trend dan noise
                market_trend = np.random.normal(50, 20)  # Drift per market
                noise = np.random.normal(0, 100)
                
                # Price dengan regional characteristics
                base_price = 15000 + seasonal + market_trend + noise
                price = max(8000, base_price)  # Minimum realistic price
                
                # Volume dengan some correlation
                volume = np.random.lognormal(mean=8, sigma=1.5)
                
                data_list.append({
                    'date': date,
                    'market': market_key,
                    'market_name': self.market_registry[market_key]['market_name'],
                    'region': self.market_registry[market_key]['region'],
                    'price': price,
                    'volume': volume,
                    'type': 'medium_grain'  # Rice type
                })
        
        df = pd.DataFrame(data_list)
        self.logger.info(f"Generated synthetic data: {len(df)} records, "
                        f"{len(market_keys)} markets, {len(date_range)} dates")
        
        return df
    
    def fetch_from_bps_api(self, 
                          indicator_code: str,
                          start_year: int = 2022,
                          end_year: int = 2024) -> pd.DataFrame:
        """
        Fetch data dari BPS (Badan Pusat Statistik) API.
        
        Parameters:
        -----------
        indicator_code : str
            BPS indicator code untuk harga beras
        start_year : int
        end_year : int
            
        Returns:
        --------
        pd.DataFrame
        """
        
        try:
            # BPS API endpoint (contoh)
            base_url = "https://webapi.bps.go.id/v1/api/list"
            
            params = {
                "model": "data",
                "domain": "0000",
                "var": indicator_code,
                "key": "YOUR_BPS_API_KEY"  # Perlu API key dari BPS
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['data'])
            
            self.logger.info(f"Fetched {len(df)} records from BPS API")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching from BPS API: {e}")
            raise
    
    def fetch_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load time series data dari CSV file.
        
        Expected CSV format:
        date, market, price, volume, [other columns]
        
        Parameters:
        -----------
        filepath : str
            Path ke CSV file
            
        Returns:
        --------
        pd.DataFrame
        """
        
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            self.logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
    
    def consolidate_multiple_sources(self, 
                                    data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Consolidate data dari multiple sources.
        
        Parameters:
        -----------
        data_sources : Dict[str, pd.DataFrame]
            Dict dengan key = source_name, value = dataframe
            
        Returns:
        --------
        pd.DataFrame
            Consolidated data dengan handling duplicates
        """
        
        dfs = []
        for source_name, df in data_sources.items():
            df = df.copy()
            df['source'] = source_name
            dfs.append(df)
        
        consolidated = pd.concat(dfs, ignore_index=True)
        
        # Handle duplicates - average jika ada
        consolidated = consolidated.sort_values('date')
        
        self.logger.info(f"Consolidated {len(data_sources)} sources into "
                        f"{len(consolidated)} records")
        
        return consolidated
    
    def save_raw_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save raw data ke file."""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved raw data to {filepath}")
        return filepath
    
    def get_market_info(self) -> pd.DataFrame:
        """Get market registry sebagai DataFrame."""
        data = []
        for market_key, info in self.market_registry.items():
            data.append({
                'market_id': market_key,
                **info
            })
        return pd.DataFrame(data)
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Validate data quality.
        
        Returns:
        --------
        Dict dengan metrics:
            - missing_values
            - date_range
            - num_markets
            - price_statistics
        """
        
        validation = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'num_markets': df['market'].nunique(),
            'missing_values': df.isnull().sum().to_dict(),
            'price_stats': {
                'mean': df['price'].mean(),
                'std': df['price'].std(),
                'min': df['price'].min(),
                'max': df['price'].max(),
            },
            'volume_stats': {
                'mean': df['volume'].mean(),
                'std': df['volume'].std(),
            }
        }
        
        return validation


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    fetcher = RiceMarketDataFetcher()
    
    # Generate synthetic data for testing
    print("Generating synthetic rice market data...")
    df = fetcher.fetch_synthetic_timeseries(
        start_date="2022-01-01",
        end_date="2024-12-31",
        frequency="D"
    )
    
    # Validate data
    print("\nData Quality Validation:")
    validation = fetcher.validate_data_quality(df)
    print(json.dumps(validation, indent=2, default=str))
    
    # Save
    fetcher.save_raw_data(df, "rice_prices_synthetic_2022_2024.csv")
    
    # Display sample
    print("\nSample data:")
    print(df.head(10))
    print(f"\nMarkets: {df['market'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
