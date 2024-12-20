import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from portfolio_optimization.data import DataCollector

@pytest.fixture
def data_collector():
    """Create a DataCollector instance with test configuration."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "test_config.yaml")
    return DataCollector(config_path)

@pytest.fixture
def test_tickers():
    """Provide test tickers."""
    return ["AAPL", "MSFT"]

def test_initialization(data_collector):
    """Test if DataCollector initializes correctly."""
    assert isinstance(data_collector, DataCollector)
    assert data_collector.start_date == "2023-01-01"
    assert data_collector.end_date == "2023-12-31"
    assert data_collector.price_col == "Close"

def test_fetch_data(data_collector, test_tickers):
    """Test if fetch_data returns correct DataFrame."""
    df = data_collector.fetch_data(test_tickers)
    
    assert isinstance(df, pd.DataFrame)
    assert all(ticker in df.columns for ticker in test_tickers)
    assert df.index.is_monotonic_increasing  # Check if dates are ordered
    assert not df.empty
    assert df.index.tz is None  # Check that index is timezone-naive

def test_calculate_returns(data_collector):
    """Test return calculation for different frequencies."""
    # Create sample price data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
    prices = pd.DataFrame({
        'AAPL': np.random.random(len(dates)) * 100,
        'MSFT': np.random.random(len(dates)) * 100
    }, index=dates)
    
    # Test daily returns
    daily_returns = data_collector.calculate_returns(prices, 'daily')
    assert isinstance(daily_returns, pd.DataFrame)
    assert len(daily_returns) == len(prices) - 1  # One less due to pct_change
    
    # Test weekly returns
    weekly_returns = data_collector.calculate_returns(prices, 'weekly')
    assert isinstance(weekly_returns, pd.DataFrame)
    assert len(weekly_returns) < len(daily_returns)  # Should have fewer rows
    
    # Test monthly returns
    monthly_returns = data_collector.calculate_returns(prices, 'monthly')
    assert isinstance(monthly_returns, pd.DataFrame)
    assert len(monthly_returns) < len(weekly_returns)  # Should have fewer rows

def test_process_data(data_collector, test_tickers):
    """Test the complete data processing pipeline."""
    data = data_collector.process_data(test_tickers)
    
    assert isinstance(data, dict)
    assert 'prices' in data
    assert 'returns' in data
    assert all(freq in data['returns'] for freq in ['daily', 'weekly', 'monthly'])
    
    # Check prices DataFrame
    assert isinstance(data['prices'], pd.DataFrame)
    assert not data['prices'].empty
    assert all(ticker in data['prices'].columns for ticker in test_tickers)
    
    # Check returns DataFrames
    for freq in ['daily', 'weekly', 'monthly']:
        returns_df = data['returns'][freq]
        assert isinstance(returns_df, pd.DataFrame)
        assert not returns_df.empty
        assert returns_df.index.tz is None  # Check that index is timezone-naive
        assert all(ticker in returns_df.columns for ticker in test_tickers)

def test_invalid_frequency(data_collector):
    """Test if invalid frequency raises ValueError."""
    prices = pd.DataFrame({'AAPL': [100, 101, 102]})
    with pytest.raises(ValueError):
        data_collector.calculate_returns(prices, 'invalid_freq')

def test_empty_tickers(data_collector):
    """Test if empty tickers list raises ValueError."""
    with pytest.raises(ValueError):
        data_collector.fetch_data([])

def test_invalid_tickers(data_collector):
    """Test handling of invalid tickers."""
    invalid_tickers = ["INVALID1", "INVALID2"]
    with pytest.raises(ValueError):
        data_collector.fetch_data(invalid_tickers)