import os
import yaml
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime

class DataCollector:
    """Class to handle data collection and preprocessing for portfolio optimization."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataCollector with configuration."""
        # Get the project root directory (two levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_file = os.path.join(project_root, config_path)
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.price_col = self.config['data']['price_column']
        
    def get_sp500_tickers(self) -> List[str]:
        """Get list of stock tickers from config or S&P 500."""
        # First check if specific stocks are defined in config
        if 'stocks' in self.config['universe']:
            return self.config['universe']['stocks']
        
        # If no specific stocks defined, use default stocks
        default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        n_stocks = min(len(default_stocks), self.config['universe']['n_stocks'])
        return default_stocks[:n_stocks]
        
        # TODO: For future implementation
        # To get actual S&P 500 constituents, you would need to:
        # 1. Use a proper market data API (e.g., Alpha Vantage, IEX Cloud)
        # 2. Or scrape from a reliable source
        # 3. Or maintain a local database of constituents
    
    def fetch_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch historical data for given tickers."""
        data_frames = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                
                # Handle different column naming conventions in yfinance
                if 'Adj Close' in df.columns:
                    price_col = 'Adj Close'
                elif 'Adj Close' in df.columns:
                    price_col = 'Adj Close'
                elif 'Close' in df.columns:
                    price_col = 'Close'
                else:
                    raise ValueError(f"No suitable price column found for {ticker}")
                
                df = df[[price_col]]
                df.columns = [ticker]
                
                # Convert timezone-aware datetime to timezone-naive
                df.index = df.index.tz_localize(None)
                
                data_frames.append(df)
                print(f"Successfully fetched data for {ticker}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
        
        if not data_frames:
            raise ValueError("No data was successfully fetched for any ticker")
        
        return pd.concat(data_frames, axis=1)
    
    def calculate_returns(self, prices: pd.DataFrame, frequency: str = 'daily') -> pd.DataFrame:
        """Calculate returns from price data."""
        if frequency == 'daily':
            returns = prices.pct_change()
        elif frequency == 'weekly':
            returns = prices.resample('W').last().pct_change()
        elif frequency == 'monthly':
            returns = prices.resample('ME').last().pct_change()  # Using 'ME' instead of 'M'
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        return returns.dropna()
    
    def process_data(self) -> Dict[str, pd.DataFrame]:
        """Main function to collect and process all required data."""
        # Get universe of stocks
        sp500_tickers = self.get_sp500_tickers()
        etfs = self.config['universe']['etfs']
        all_tickers = sp500_tickers + etfs
        
        # Fetch price data
        prices = self.fetch_data(all_tickers)
        
        # Calculate returns at different frequencies
        returns = {
            'daily': self.calculate_returns(prices, 'daily'),
            'weekly': self.calculate_returns(prices, 'weekly'),
            'monthly': self.calculate_returns(prices, 'monthly')
        }
        
        # Save data
        self.save_data(prices, returns)
        
        return {'prices': prices, 'returns': returns}
    
    def save_data(self, prices: pd.DataFrame, returns: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to files."""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, 'data')
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Save prices
        prices.to_csv(os.path.join(data_dir, 'prices.csv'))
        
        # Save returns
        for freq, df in returns.items():
            df.to_csv(os.path.join(data_dir, f'returns_{freq}.csv'))

if __name__ == "__main__":
    collector = DataCollector()
    data = collector.process_data()
    print("Data collection and processing completed successfully.") 