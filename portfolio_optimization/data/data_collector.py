import os
import yaml
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = __name__,
    console_level: Optional[int] = None,  # None means no console output
    file_level: int = logging.INFO
) -> logging.Logger:
    """Configure logging to work in both scripts and notebooks.
    
    Args:
        name: Logger name
        console_level: Logging level for console output. None to disable console logging.
        file_level: Logging level for file output
        
    Returns:
        logging.Logger: Configured logger instance
        
    Notes:
        - Always writes to a rotating log file
        - Console output is optional and can be disabled by setting console_level to None
        - File handler rotates at 10MB with 5 backup files
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(file_level, console_level or logging.INFO))
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # File handler with rotation (always enabled)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "data_collector.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(file_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Initialize logger with default settings
# In scripts: Enable console output at INFO level
# In notebooks: Disable console output by not specifying console_level
logger = setup_logger()

class DataCollector:
    """Class to handle data collection and preprocessing for portfolio optimization.
    
    This class provides functionality to fetch historical price data from Yahoo Finance,
    calculate returns at different frequencies, and save the processed data.
    
    Attributes:
        config (dict): Configuration parameters loaded from yaml file
        start_date (str): Start date for data collection
        end_date (str): End date for data collection 
        price_col (str): Name of price column to use from data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize DataCollector with configuration.
        
        Args:
            config_path (str): Path to configuration yaml file. Defaults to "config/config.yaml"
            
        Raises:
            Exception: If configuration file cannot be loaded
        """
        # Get the project root directory (two levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_file = os.path.join(project_root, config_path)
        
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {config_file}")
            
            self.start_date = self.config['data']['start_date']
            self.end_date = self.config['data']['end_date']
            self.price_col = self.config['data']['price_column']
            
            logger.info(f"Data collection period: {self.start_date} to {self.end_date}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def fetch_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch historical price data for given tickers from Yahoo Finance.
        
        Args:
            tickers (List[str]): List of stock ticker symbols to fetch data for
            
        Returns:
            pd.DataFrame: DataFrame containing price data for all tickers
            
        Raises:
            ValueError: If no data could be fetched for any ticker
        """
        logger.info(f"Starting data collection for {len(tickers)} tickers")
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
                    logger.error(f"No suitable price column found for {ticker}")
                    raise ValueError(f"No suitable price column found for {ticker}")
                
                df = df[[price_col]]
                df.columns = [ticker]
                
                # Convert timezone-aware datetime to timezone-naive
                df.index = df.index.tz_localize(None)
                
                data_frames.append(df)
                logger.info(f"Successfully fetched data for {ticker} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
        
        if not data_frames:
            logger.error("No data was successfully fetched for any ticker")
            raise ValueError("No data was successfully fetched for any ticker")
        
        result = pd.concat(data_frames, axis=1)
        logger.info(f"Combined data shape: {result.shape}")
        return result
    
    def calculate_returns(self, prices: pd.DataFrame, frequency: str = 'daily') -> pd.DataFrame:
        """Calculate returns from price data at specified frequency.
        
        Args:
            prices (pd.DataFrame): DataFrame containing price data
            frequency (str): Frequency for returns calculation. One of 'daily', 'weekly', 'monthly'
            
        Returns:
            pd.DataFrame: DataFrame containing calculated returns
            
        Raises:
            ValueError: If unsupported frequency is specified
        """
        logger.info(f"Calculating {frequency} returns")
        
        try:
            if frequency == 'daily':
                returns = prices.pct_change()
            elif frequency == 'weekly':
                returns = prices.resample('W').last().pct_change()
            elif frequency == 'monthly':
                returns = prices.resample('ME').last().pct_change()
            else:
                logger.error(f"Unsupported frequency: {frequency}")
                raise ValueError(f"Unsupported frequency: {frequency}")
            
            returns = returns.dropna()
            logger.info(f"Generated {len(returns)} {frequency} returns")
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating {frequency} returns: {str(e)}")
            raise
    
    def process_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Main function to collect and process all required data.
        
        Fetches price data, calculates returns at different frequencies, and saves the data.
        
        Args:
            tickers (List[str]): List of stock ticker symbols to process
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing prices and returns DataFrames
            
        Raises:
            Exception: If data processing fails
        """
        logger.info("Starting data processing pipeline")
        
        try:
            # Fetch price data
            prices = self.fetch_data(tickers)
            
            # Calculate returns at different frequencies
            returns = {
                'daily': self.calculate_returns(prices, 'daily'),
                'weekly': self.calculate_returns(prices, 'weekly'),
                'monthly': self.calculate_returns(prices, 'monthly')
            }
            
            # Save data
            self.save_data(prices, returns)
            
            logger.info("Data processing completed successfully")
            return {'prices': prices, 'returns': returns}
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise
    
    def save_data(self, prices: pd.DataFrame, returns: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to CSV files.
        
        Args:
            prices (pd.DataFrame): DataFrame containing price data
            returns (Dict[str, pd.DataFrame]): Dictionary of returns DataFrames at different frequencies
            
        Raises:
            Exception: If saving data fails
        """
        try:
            # Get project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(project_root, 'data')
            
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Saving data to directory: {data_dir}")
            
            # Save prices
            price_path = os.path.join(data_dir, 'prices.csv')
            prices.to_csv(price_path)
            logger.info(f"Saved prices to {price_path}")
            
            # Save returns
            for freq, df in returns.items():
                returns_path = os.path.join(data_dir, f'returns_{freq}.csv')
                df.to_csv(returns_path)
                logger.info(f"Saved {freq} returns to {returns_path}")
                
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Example tickers
    data = collector.process_data(tickers)
    logger.info("Script execution completed successfully")