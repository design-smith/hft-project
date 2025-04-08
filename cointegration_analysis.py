"""
Module for performing cointegration analysis on forex data.
This module contains functions for testing statistical relationships
between currency pairs to identify potential trading opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('CointegrationAnalysis')

def align_price_data(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Align two price dataframes to ensure they have matching timestamps.
    """
    logger.info("Aligning price data between two series")
    
    if 'timestamp' in df1.columns:
        df1 = df1.set_index('timestamp')
    if 'timestamp' in df2.columns:
        df2 = df2.set_index('timestamp')
    
    common_index = df1.index.intersection(df2.index)
    
    if len(common_index) < 30:
        logger.warning(f"Insufficient common data points: {len(common_index)}")
        raise ValueError("Insufficient common data points for cointegration testing")
    
    price1 = df1.loc[common_index, 'close']
    price2 = df2.loc[common_index, 'close']
    
    logger.info(f"Successfully aligned {len(common_index)} data points")
    return price1, price2

def calculate_hedge_ratio(price1: pd.Series, price2: pd.Series) -> float:
    """
    Calculate the hedge ratio between two price series using OLS regression.
    """
    X = price2.values.reshape(-1, 1)
    y = price1.values
    
    model = LinearRegression().fit(X, y)
    hedge_ratio = model.coef_[0]
    
    logger.info(f"Calculated hedge ratio: {hedge_ratio}")
    return hedge_ratio

def calculate_spread(price1: pd.Series, price2: pd.Series, hedge_ratio: Optional[float] = None) -> pd.Series:
    """
    Calculate the spread between two price series.
    """
    if hedge_ratio is None:
        hedge_ratio = calculate_hedge_ratio(price1, price2)
    
    spread = price1 - hedge_ratio * price2
    return spread

def calculate_zscore(spread: pd.Series) -> pd.Series:
    """
    Calculate the z-score of a spread series.
    """
    mean = spread.mean()
    std = spread.std()
    z_score = (spread - mean) / std
    return z_score

def test_cointegration(price1: pd.Series, price2: pd.Series) -> Tuple[float, float]:
    """
    Test for cointegration between two price series.
    """
    logger.info("Testing cointegration between two price series")
    
    score, p_value, _ = coint(price1, price2)
    
    logger.info(f"Cointegration test results - Score: {score}, P-value: {p_value}")
    return score, p_value

def is_cointegrated(price1: pd.Series, price2: pd.Series, threshold: float = 0.05) -> bool:
    """
    Determine if two price series are cointegrated based on p-value threshold.
    """
    _, p_value = test_cointegration(price1, price2)
    return p_value < threshold

def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate the half-life of mean reversion for the spread.
    
    Args:
        spread: Spread series
        
    Returns:
        Half-life in periods (e.g., hours if timeframe is H1)
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    
    # Ensure lengths match
    spread_lag = spread_lag[1:]
    spread_diff = spread_diff[1:]
    
    # Add constant term for OLS regression
    spread_lag_with_const = np.vstack([spread_lag, np.ones(len(spread_lag))]).T
    
    # Fit OLS regression: spread_diff ~ spread_lag
    model = OLS(spread_diff, spread_lag_with_const).fit()
    beta = model.params[0]
    
    if beta >= 0:
        logger.warning("Non-mean-reverting spread detected (beta >= 0)")
        return np.inf  # Infinite half-life if not mean-reverting
    
    half_life = -np.log(2) / beta
    return half_life

def analyze_pair(df1: pd.DataFrame, df2: pd.DataFrame, pair1: str, pair2: str, threshold: float = 0.05) -> Optional[Dict]:
    """
    Perform full cointegration analysis on a pair of currency pairs, including half-life.
    """
    logger.info(f"Analyzing potential cointegration between {pair1} and {pair2}")
    
    try:
        # Align price data
        price1, price2 = align_price_data(df1, df2)
        
        # Test cointegration
        score, p_value = test_cointegration(price1, price2)
        
        if p_value >= threshold:
            logger.info(f"{pair1} and {pair2} are not cointegrated (p-value: {p_value})")
            return None
        
        # Calculate hedge ratio
        hedge_ratio = calculate_hedge_ratio(price1, price2)
        
        # Calculate spread and z-score
        spread = calculate_spread(price1, price2, hedge_ratio)
        z_score = calculate_zscore(spread)
        
        # Calculate half-life
        half_life = calculate_half_life(spread)
        
        # Create result dictionary
        result = {
            'pair1': pair1,
            'pair2': pair2,
            'p_value': float(p_value),
            'score': float(score),
            'hedge_ratio': float(hedge_ratio),
            'spread_mean': float(spread.mean()),
            'spread_std': float(spread.std()),
            'current_spread': float(spread.iloc[-1]),
            'current_z_score': float(z_score.iloc[-1]),
            'spread_series': spread.values.tolist(),
            'z_score_series': z_score.values.tolist(),
            'timestamps': price1.index.tolist(),
            'half_life': float(half_life),
            'analysis_time': datetime.now().isoformat()
        }
        
        logger.info(f"Found cointegration between {pair1} and {pair2} (p-value: {p_value}, half-life: {half_life:.2f})")
        return result
    
    except Exception as e:
        logger.error(f"Error analyzing {pair1} and {pair2}: {str(e)}")
        return None

def find_cointegrated_pairs(
    pair_data: Dict[str, pd.DataFrame], 
    threshold: float = 0.05
) -> List[Dict]:
    """
    Find all cointegrated pairs from a dictionary of price data.
    """
    logger.info(f"Searching for cointegrated pairs among {len(pair_data)} currency pairs")
    
    results = []
    pairs = list(pair_data.keys())
    
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            pair1 = pairs[i]
            pair2 = pairs[j]
            
            if len(pair_data[pair1]) < 30 or len(pair_data[pair2]) < 30:
                logger.warning(f"Skipping {pair1}/{pair2} due to insufficient data")
                continue
            
            result = analyze_pair(pair_data[pair1], pair_data[pair2], pair1, pair2, threshold)
            
            if result:
                results.append(result)
    
    results.sort(key=lambda x: x['p_value'])
    
    logger.info(f"Found {len(results)} cointegrated pairs")
    return results

'''class Backtest:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.equity = [initial_capital]
        self.trades = []

    def simulate(self, cointegration_results, pair_data):
        trades = []
        for result in cointegration_results:
            pair1 = result['pair1']
            pair2 = result['pair2']
            z_scores = pd.Series(result['z_score_series'], index=result['timestamps'])
            position = 0
            for t in range(1, len(z_scores)):
                if position == 0 and abs(z_scores[t]) > 1.0:  # Entry condition
                    position = 1 if z_scores[t] < -1.0 else -1
                    entry_price1 = pair_data[pair1]['close'].iloc[t]
                    entry_price2 = pair_data[pair2]['close'].iloc[t]
                elif position != 0 and abs(z_scores[t]) < 0.5:  # Exit condition
                    exit_price1 = pair_data[pair1]['close'].iloc[t]
                    exit_price2 = pair_data[pair2]['close'].iloc[t]
                    pl = (exit_price1 - entry_price1) * position - (exit_price2 - entry_price2) * position * result['hedge_ratio']
                    trades.append({'pair1': pair1, 'pair2': pair2, 'pl': pl})
                    position = 0
        return pd.DataFrame(trades)'''