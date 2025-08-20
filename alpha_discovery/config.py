# alpha_discovery/config.py

from pydantic import BaseModel
from typing import List
from datetime import date

class GaConfig(BaseModel):
    """Genetic Algorithm Search Parameters"""
    population_size: int = 100
    generations: int = 15
    elitism_rate: float = 0.1
    mutation_rate: float = 0.2
    seed: int = 68
    setup_lengths_to_explore: List[int] = [2, 3]

class DataConfig(BaseModel):
    """Data Source and Ticker Configuration"""
    excel_file_path: str = 'data_store/raw/bb_data.xlsx'
    parquet_file_path: str = 'data_store/processed/bb_data.parquet'
    start_date: date = date(2010, 1, 1)
    end_date: date = date(2025, 8, 19)
    holdout_start_date: date = date(2023, 8, 19)

    # Finalized ticker lists
    tradable_tickers: List[str] = [
        'XLE US Equity', 'XLK US Equity', 'XLRE US Equity', 'XLC US Equity',
        'XLV US Equity', 'XLP US Equity', 'SPY US Equity', 'QQQ US Equity',
        'CRWV US Equity', 'TSLA US Equity', 'AMZN US Equity',
        'GOOGL US Equity', 'MSFT US Equity', 'AAPL US Equity', 'LLY US Equity',
        'JPM US Equity', 'C US Equity', 'PLTR US Equity', 'ARM US Equity',
        'AMD US Equity', 'BMY US Equity', 'PEPS US Equity', 'NKE US Equity',
        'WMT US Equity', 'MSTR US Equity', 'COIN US Equity', 'BTC Index'
    ]
    macro_tickers: List[str] = [
        'SPX Index', 'NDX Index', 'RTY Index', 'MXWO Index',
        'USGG10YR Index', 'USGG2YR Index', 'DXY Curncy', 'JPY Curncy',
        'EUR Curncy', 'EEM US Equity', 'CL1 Comdty', 'HG1 Comdty',
        'XAU Curncy'
    ]

class ValidationConfig(BaseModel):
    """Walk-Forward and Robustness Validation Settings"""
    min_initial_support: int = 5
    min_final_support: int = 8
    embargo_days: int = 5

class Settings(BaseModel):
    """Main container for all project settings"""
    ga: GaConfig = GaConfig()
    data: DataConfig = DataConfig()
    validation: ValidationConfig = ValidationConfig()

# Instantiate a global settings object for easy import
settings = Settings()