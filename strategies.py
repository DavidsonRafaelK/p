"""
Advanced Trading Strategies Module
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedStrategies:
    def __init__(self):
        self.strategies = {
            'momentum': self.momentum_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy,
            'scalping': self.scalping_strategy,
            'swing': self.swing_strategy,
            'grid': self.grid_strategy
        }
    
    def momentum_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Momentum-based trading strategy
        """
        if len(df) < 50:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'insufficient_data'}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        reasons = []
        
        # RSI momentum
        if latest['rsi'] > 50 and prev['rsi'] <= 50:
            signals.append('buy')
            reasons.append('rsi_momentum_up')
        elif latest['rsi'] < 50 and prev['rsi'] >= 50:
            signals.append('sell')
            reasons.append('rsi_momentum_down')
        
        # MACD momentum
        if (latest['macd'] > latest['macd_signal'] and 
            prev['macd'] <= prev['macd_signal'] and 
            latest['macd'] > 0):
            signals.append('buy')
            reasons.append('macd_bullish_cross')
        elif (latest['macd'] < latest['macd_signal'] and 
              prev['macd'] >= prev['macd_signal'] and 
              latest['macd'] < 0):
            signals.append('sell')
            reasons.append('macd_bearish_cross')
        
        # Price momentum
        price_change = (latest['close'] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        if price_change > 0.02:  # 2% price increase
            signals.append('buy')
            reasons.append('strong_price_momentum')
        elif price_change < -0.02:  # 2% price decrease
            signals.append('sell')
            reasons.append('weak_price_momentum')
        
        # Volume confirmation
        volume_ratio = latest['volume'] / df['volume'].rolling(20).mean().iloc[-1]
        if volume_ratio > 1.5:  # High volume confirmation
            confidence_boost = 0.2
        else:
            confidence_boost = 0
        
        # Determine final signal
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        
        if buy_count > sell_count:
            signal = 'buy'
            confidence = (buy_count / len(signals)) * 0.7 + confidence_boost
        elif sell_count > buy_count:
            signal = 'sell'
            confidence = (sell_count / len(signals)) * 0.7 + confidence_boost
        else:
            signal = 'hold'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': min(confidence, 1.0),
            'reason': '; '.join(reasons),
            'details': {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'price_change': price_change,
                'volume_ratio': volume_ratio
            }
        }
    
    def mean_reversion_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Mean reversion strategy using Bollinger Bands and RSI
        """
        if len(df) < 50:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'insufficient_data'}
        
        latest = df.iloc[-1]
        
        signals = []
        reasons = []
        
        # Bollinger Bands mean reversion
        bb_position = ((latest['close'] - latest['bb_lower']) / 
                      (latest['bb_upper'] - latest['bb_lower']))
        
        if bb_position < 0.1 and latest['rsi'] < 30:  # Near lower band + oversold
            signals.append('buy')
            reasons.append('bb_oversold')
        elif bb_position > 0.9 and latest['rsi'] > 70:  # Near upper band + overbought
            signals.append('sell')
            reasons.append('bb_overbought')
        
        # RSI mean reversion
        if latest['rsi'] < 25:  # Extremely oversold
            signals.append('buy')
            reasons.append('rsi_extreme_oversold')
        elif latest['rsi'] > 75:  # Extremely overbought
            signals.append('sell')
            reasons.append('rsi_extreme_overbought')
        
        # Stochastic mean reversion
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            signals.append('buy')
            reasons.append('stoch_oversold')
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            signals.append('sell')
            reasons.append('stoch_overbought')
        
        # Price distance from moving average
        sma_distance = (latest['close'] - latest['sma_20']) / latest['sma_20']
        if sma_distance < -0.03:  # 3% below SMA
            signals.append('buy')
            reasons.append('far_below_sma')
        elif sma_distance > 0.03:  # 3% above SMA
            signals.append('sell')
            reasons.append('far_above_sma')
        
        # Determine final signal
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        
        if buy_count >= 2:
            signal = 'buy'
            confidence = min(buy_count / 4, 0.8)
        elif sell_count >= 2:
            signal = 'sell'
            confidence = min(sell_count / 4, 0.8)
        else:
            signal = 'hold'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': '; '.join(reasons),
            'details': {
                'bb_position': bb_position,
                'rsi': latest['rsi'],
                'sma_distance': sma_distance,
                'stoch_k': latest['stoch_k']
            }
        }
    
    def breakout_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Breakout strategy using volume and volatility
        """
        if len(df) < 50:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'insufficient_data'}
        
        latest = df.iloc[-1]
        
        signals = []
        reasons = []
        
        # Bollinger Bands breakout
        if latest['close'] > latest['bb_upper']:
            signals.append('buy')
            reasons.append('bb_upper_breakout')
        elif latest['close'] < latest['bb_lower']:
            signals.append('sell')
            reasons.append('bb_lower_breakout')
        
        # Volume breakout
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        if latest['volume'] > avg_volume * 2:  # Volume spike
            price_change = (latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            if price_change > 0.005:  # 0.5% price increase with volume
                signals.append('buy')
                reasons.append('volume_price_breakout_up')
            elif price_change < -0.005:  # 0.5% price decrease with volume
                signals.append('sell')
                reasons.append('volume_price_breakout_down')
        
        # Volatility breakout
        volatility = df['close'].rolling(20).std().iloc[-1]
        avg_volatility = df['close'].rolling(50).std().mean()
        
        if volatility > avg_volatility * 1.5:  # High volatility
            # Check price direction
            recent_high = df['high'].rolling(5).max().iloc[-1]
            recent_low = df['low'].rolling(5).min().iloc[-1]
            
            if latest['close'] > recent_high * 0.999:  # Near recent high
                signals.append('buy')
                reasons.append('volatility_breakout_up')
            elif latest['close'] < recent_low * 1.001:  # Near recent low
                signals.append('sell')
                reasons.append('volatility_breakout_down')
        
        # Support/Resistance breakout
        support = df['low'].rolling(20).min().iloc[-1]
        resistance = df['high'].rolling(20).max().iloc[-1]
        
        if latest['close'] > resistance * 1.001:  # Break above resistance
            signals.append('buy')
            reasons.append('resistance_breakout')
        elif latest['close'] < support * 0.999:  # Break below support
            signals.append('sell')
            reasons.append('support_breakout')
        
        # Determine final signal
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        
        if buy_count > sell_count and buy_count >= 2:
            signal = 'buy'
            confidence = min(buy_count / 4, 0.9)
        elif sell_count > buy_count and sell_count >= 2:
            signal = 'sell'
            confidence = min(sell_count / 4, 0.9)
        else:
            signal = 'hold'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': '; '.join(reasons),
            'details': {
                'volume_ratio': latest['volume'] / avg_volume,
                'volatility_ratio': volatility / avg_volatility,
                'support': support,
                'resistance': resistance
            }
        }
    
    def scalping_strategy(self, df: pd.DataFrame) -> Dict:
        """
        High-frequency scalping strategy
        """
        if len(df) < 20:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'insufficient_data'}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        reasons = []
        
        # Quick RSI scalping
        if latest['rsi'] < 40 and prev['rsi'] >= 40:
            signals.append('buy')
            reasons.append('rsi_quick_dip')
        elif latest['rsi'] > 60 and prev['rsi'] <= 60:
            signals.append('sell')
            reasons.append('rsi_quick_rise')
        
        # Price action scalping
        price_change = (latest['close'] - prev['close']) / prev['close']
        if price_change > 0.002:  # 0.2% quick rise
            signals.append('buy')
            reasons.append('quick_price_rise')
        elif price_change < -0.002:  # 0.2% quick drop
            signals.append('sell')
            reasons.append('quick_price_drop')
        
        # Volume confirmation for scalping
        volume_ratio = latest['volume'] / df['volume'].rolling(5).mean().iloc[-1]
        if volume_ratio > 1.3:  # Volume confirmation
            confidence_boost = 0.3
        else:
            confidence_boost = 0
        
        # EMA scalping
        ema_5 = ta.trend.ema_indicator(df['close'], window=5)
        ema_10 = ta.trend.ema_indicator(df['close'], window=10)
        
        if ema_5.iloc[-1] > ema_10.iloc[-1] and ema_5.iloc[-2] <= ema_10.iloc[-2]:
            signals.append('buy')
            reasons.append('ema_cross_up')
        elif ema_5.iloc[-1] < ema_10.iloc[-1] and ema_5.iloc[-2] >= ema_10.iloc[-2]:
            signals.append('sell')
            reasons.append('ema_cross_down')
        
        # Determine final signal
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        
        if buy_count > sell_count:
            signal = 'buy'
            confidence = min((buy_count / 3) * 0.6 + confidence_boost, 0.8)
        elif sell_count > buy_count:
            signal = 'sell'
            confidence = min((sell_count / 3) * 0.6 + confidence_boost, 0.8)
        else:
            signal = 'hold'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': '; '.join(reasons),
            'details': {
                'price_change': price_change,
                'volume_ratio': volume_ratio,
                'rsi': latest['rsi']
            }
        }
    
    def swing_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Swing trading strategy for longer-term positions
        """
        if len(df) < 100:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'insufficient_data'}
        
        latest = df.iloc[-1]
        
        signals = []
        reasons = []
        
        # Trend following
        if latest['sma_20'] > latest['sma_50']:
            if latest['close'] > latest['sma_20'] and latest['rsi'] > 45:
                signals.append('buy')
                reasons.append('uptrend_continuation')
        else:
            if latest['close'] < latest['sma_20'] and latest['rsi'] < 55:
                signals.append('sell')
                reasons.append('downtrend_continuation')
        
        # Swing high/low detection
        swing_high = df['high'].rolling(10).max().iloc[-5]
        swing_low = df['low'].rolling(10).min().iloc[-5]
        
        if latest['close'] > swing_high:
            signals.append('buy')
            reasons.append('swing_high_break')
        elif latest['close'] < swing_low:
            signals.append('sell')
            reasons.append('swing_low_break')
        
        # Weekly momentum (assuming 1-minute data)
        weekly_change = (latest['close'] - df['close'].iloc[-1440]) / df['close'].iloc[-1440]  # 1440 minutes = 1 day
        if weekly_change > 0.05:  # 5% weekly gain
            signals.append('buy')
            reasons.append('weekly_momentum_up')
        elif weekly_change < -0.05:  # 5% weekly loss
            signals.append('sell')
            reasons.append('weekly_momentum_down')
        
        # Determine final signal
        buy_count = signals.count('buy')
        sell_count = signals.count('sell')
        
        if buy_count >= 2:
            signal = 'buy'
            confidence = min(buy_count / 3, 0.8)
        elif sell_count >= 2:
            signal = 'sell'
            confidence = min(sell_count / 3, 0.8)
        else:
            signal = 'hold'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': '; '.join(reasons),
            'details': {
                'weekly_change': weekly_change,
                'swing_high': swing_high,
                'swing_low': swing_low
            }
        }
    
    def grid_strategy(self, df: pd.DataFrame, grid_size: float = 0.01) -> Dict:
        """
        Grid trading strategy
        """
        if len(df) < 50:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'insufficient_data'}
        
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Calculate grid levels
        base_price = df['close'].rolling(50).mean().iloc[-1]
        
        # Create grid levels
        grid_levels = []
        for i in range(-5, 6):  # 11 grid levels
            level = base_price * (1 + i * grid_size)
            grid_levels.append(level)
        
        # Find current grid position
        current_grid = None
        for i, level in enumerate(grid_levels[:-1]):
            if grid_levels[i] <= current_price < grid_levels[i+1]:
                current_grid = i
                break
        
        if current_grid is None:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'price_outside_grid'}
        
        # Grid trading logic
        if current_grid < 5:  # Below middle, buy
            signal = 'buy'
            confidence = (5 - current_grid) / 5 * 0.7
            reason = f'grid_buy_level_{current_grid}'
        elif current_grid > 5:  # Above middle, sell
            signal = 'sell'
            confidence = (current_grid - 5) / 5 * 0.7
            reason = f'grid_sell_level_{current_grid}'
        else:  # At middle
            signal = 'hold'
            confidence = 0
            reason = 'grid_middle'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'details': {
                'current_grid': current_grid,
                'base_price': base_price,
                'grid_levels': grid_levels
            }
        }
    
    def get_strategy_signals(self, df: pd.DataFrame, strategy_names: List[str]) -> Dict:
        """
        Get signals from multiple strategies
        """
        results = {}
        
        for strategy_name in strategy_names:
            if strategy_name in self.strategies:
                try:
                    result = self.strategies[strategy_name](df)
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"Error in {strategy_name}: {e}")
                    results[strategy_name] = {
                        'signal': 'hold',
                        'confidence': 0,
                        'reason': f'error: {e}'
                    }
        
        return results
    
    def aggregate_signals(self, strategy_results: Dict, weights: Dict = None) -> Dict:
        """
        Aggregate signals from multiple strategies
        """
        if not strategy_results:
            return {'signal': 'hold', 'confidence': 0, 'reason': 'no_strategies'}
        
        if weights is None:
            weights = {name: 1.0 for name in strategy_results.keys()}
        
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        reasons = []
        
        for strategy_name, result in strategy_results.items():
            weight = weights.get(strategy_name, 1.0)
            confidence = result['confidence']
            
            if result['signal'] == 'buy':
                buy_score += weight * confidence
                reasons.append(f"{strategy_name}:buy({confidence:.2f})")
            elif result['signal'] == 'sell':
                sell_score += weight * confidence
                reasons.append(f"{strategy_name}:sell({confidence:.2f})")
            
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final signal
        if buy_score > sell_score and buy_score > 0.3:
            signal = 'buy'
            confidence = buy_score
        elif sell_score > buy_score and sell_score > 0.3:
            signal = 'sell'
            confidence = sell_score
        else:
            signal = 'hold'
            confidence = 0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': '; '.join(reasons),
            'details': {
                'buy_score': buy_score,
                'sell_score': sell_score,
                'strategies': strategy_results
            }
        }

# Example usage
if __name__ == "__main__":
    # This would be used within the main trading bot
    strategies = AdvancedStrategies()
    
    # Example: Get signals from multiple strategies
    # df = get_historical_data()  # Your data
    # results = strategies.get_strategy_signals(df, ['momentum', 'mean_reversion', 'breakout'])
    # final_signal = strategies.aggregate_signals(results)
    
    print("Advanced trading strategies module loaded successfully")
