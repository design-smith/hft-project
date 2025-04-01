import os
import time
import logging
from typing import Optional, Dict, List, Any
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oanda_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OANDAFetcher')

# Constants
OANDA_API_URL = os.getenv('OANDA_API_URL')
INSTRUMENTS = os.getenv('INSTRUMENTS', 'EUR_USD,GBP_USD,USD_JPY,USD_CAD').split(',')

class OandaDataFetcher:
    """
    Class for fetching forex data from OANDA API with error handling and rate limiting.
    """
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OandaDataFetcher with API credentials."""
        self.api_key = api_key or os.getenv('OANDA_API_KEY')

        if not self.api_key:
            raise ValueError("Please provide OANDA_API_KEY either as parameter or in .env file")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept-Datetime-Format": "RFC3339"
        }

        self.max_retries = 3
        self.retry_delay = 1
        self.request_timeout = 30
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.now()
        self.rate_limit_pause = 1.0  # seconds between requests

    def _handle_rate_limit(self):
        """Implement rate limiting"""
        current_time = datetime.now()
        time_since_last_request = (current_time - self.last_request_time).total_seconds()

        if time_since_last_request < self.rate_limit_pause:
            sleep_time = self.rate_limit_pause - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = datetime.now()
        self.request_count += 1

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        retry_count: int = 0
    ) -> Optional[Dict]:
        """Make an API request with retry logic."""
        self._handle_rate_limit()
        
        try:
            response = requests.get(
                f"{OANDA_API_URL}/{endpoint}",
                headers=self.headers,
                params=params,
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Too many requests
                logger.warning("Rate limit hit, increasing pause time")
                self.rate_limit_pause *= 2
                time.sleep(self.rate_limit_pause)
                return self._make_request(endpoint, params)  # Retry
            else:
                if retry_count < self.max_retries:
                    logger.warning(f"Request failed, retrying ({retry_count + 1}/{self.max_retries}): {response.status_code} - {response.text}")
                    time.sleep(self.retry_delay * (2 ** retry_count))
                    return self._make_request(endpoint, params, retry_count + 1)
                else:
                    logger.error(f"Request failed after {self.max_retries} retries: {response.status_code} - {response.text}")
                    return None

        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                logger.warning(f"Request error, retrying ({retry_count + 1}/{self.max_retries}): {str(e)}")
                time.sleep(self.retry_delay * (2 ** retry_count))
                return self._make_request(endpoint, params, retry_count + 1)
            else:
                logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                return None

    def validate_instrument(self, instrument: str) -> None:
        """Validate that the instrument is supported."""
        if instrument not in INSTRUMENTS:
            raise ValueError(f"Invalid instrument: {instrument}. Must be one of {INSTRUMENTS}")

    def validate_timeframe(self, timeframe: str) -> str:
        """Validate that the timeframe is supported."""
        valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D", "W", "M"]
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
        return timeframe

    def get_latest_candles(
        self,
        instrument: str,
        count: int = 100,
        timeframe: str = "H1"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch the most recent candles for a given instrument.
        
        Args:
            instrument: Currency pair to fetch (e.g., "EUR_USD")
            count: Number of candles to fetch
            timeframe: Candle timeframe (e.g., "H1" for hourly)
            
        Returns:
            DataFrame with candle data or None if request failed
        """
        self.validate_instrument(instrument)
        timeframe = self.validate_timeframe(timeframe)
        
        endpoint = f"instruments/{instrument}/candles"
        params = {
            "price": "M",  # Midpoint prices
            "granularity": timeframe,
            "count": count
        }
        
        logger.info(f"Fetching {count} latest {timeframe} candles for {instrument}")
        response_data = self._make_request(endpoint, params)
        
        if response_data and 'candles' in response_data:
            return self._process_candles(response_data['candles'])
        return None

    def get_historical_candles(
        self,
        instrument: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        timeframe: str = "H1"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candles for a specific time period.
        
        Args:
            instrument: Currency pair to fetch (e.g., "EUR_USD")
            start_time: Start time for data fetch
            end_time: End time for data fetch (defaults to now)
            timeframe: Candle timeframe (e.g., "H1" for hourly)
            
        Returns:
            DataFrame with candle data or None if request failed
        """
        self.validate_instrument(instrument)
        timeframe = self.validate_timeframe(timeframe)
        
        if end_time is None:
            end_time = datetime.now()
            
        # Split into chunks to avoid hitting API limits
        date_ranges = self._calculate_date_ranges(start_time, end_time)
        all_candles = []
        
        for start, end in date_ranges:
            endpoint = f"instruments/{instrument}/candles"
            params = {
                "price": "M",  # Midpoint prices
                "granularity": timeframe,
                "from": start.isoformat() + "Z",
                "to": end.isoformat() + "Z"
            }
            
            logger.info(f"Fetching {timeframe} candles for {instrument} from {start} to {end}")
            response_data = self._make_request(endpoint, params)
            
            if response_data and 'candles' in response_data:
                candles = response_data['candles']
                all_candles.extend(candles)
                logger.info(f"Fetched {len(candles)} candles")
                time.sleep(1)  # Rate limiting
            else:
                logger.error(f"Failed to fetch data for {instrument} from {start} to {end}")
                
        if not all_candles:
            return None
            
        return self._process_candles(all_candles)
        
    def _calculate_date_ranges(
        self, 
        start_time: datetime,
        end_time: datetime
    ) -> List[tuple]:
        """Split date range into monthly chunks to avoid hitting API limits."""
        ranges = []
        current_date = start_time
        
        while current_date < end_time:
            # Move to next month or end date, whichever is sooner
            if current_date.month == 12:
                next_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                next_date = current_date.replace(month=current_date.month + 1)
                
            next_date = min(next_date, end_time)
            ranges.append((current_date, next_date))
            current_date = next_date
            
        return ranges
        
    def _process_candles(self, candles: List[Dict]) -> pd.DataFrame:
        """Convert candle data to DataFrame and process it."""
        processed_data = []
        
        for candle in candles:
            if candle.get('complete', True):  # Only use complete candles
                processed_data.append({
                    'timestamp': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
                
        df = pd.DataFrame(processed_data)
        
        if not df.empty:
            # Clean the dataframe
            df = df.sort_values('timestamp')
            df = df.drop_duplicates()
            
        return df
        
    def save_data_to_csv(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str
    ) -> str:
        """Save DataFrame to CSV file."""
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        
        filename = f"{instrument}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = output_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} candles to {filepath}")
        
        return str(filepath)


def main():
    """Example usage of the OandaDataFetcher."""
    fetcher = OandaDataFetcher()
    
    #Fetch latest candles
    instrument = "EUR_USD"
    timeframe = "H1"
    latest_df = fetcher.get_latest_candles(instrument, count=100, timeframe=timeframe)
    
    if latest_df is not None:
        print("\nLatest candles:")
        print(latest_df.head())
        fetcher.save_data_to_csv(latest_df, instrument, f"{timeframe}_latest")
    
    #Fetch historical data for a date range
    start_date = datetime.now() - timedelta(days=30)  # Last 30 days
    end_date = datetime.now()
    
    historical_df = fetcher.get_historical_candles(
        instrument, 
        start_time=start_date,
        end_time=end_date,
        timeframe=timeframe
    )
    
    if historical_df is not None:
        print("\nHistorical data sample:")
        print(historical_df.head())
        print(f"\nTotal candles fetched: {len(historical_df)}")
        fetcher.save_data_to_csv(historical_df, instrument, f"{timeframe}_historical")
        
    #Fetch data for multiple instruments
    instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
    for instrument in instruments:
        try:
            print(f"\nFetching data for {instrument}...")
            df = fetcher.get_latest_candles(instrument, count=50)
            if df is not None:
                print(f"Latest {instrument} data:")
                print(df.head(3))
        except Exception as e:
            print(f"Error fetching {instrument}: {str(e)}")


if __name__ == "__main__":
    main()