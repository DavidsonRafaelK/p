import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv
import time
import schedule
import json
from typing import Dict, List, Optional, Tuple

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.exchange = self._init_exchange()
        self.symbol = os.getenv('SYMBOL', 'BTC/USDT')
        self.timeframe = os.getenv('TIMEFRAME', '1m')
        self.trade_amount = float(os.getenv('TRADE_AMOUNT', '0.001'))
        self.stop_loss_percent = float(os.getenv('STOP_LOSS_PERCENT', '2.0'))
        self.take_profit_percent = float(os.getenv('TAKE_PROFIT_PERCENT', '3.0'))
        
        # Trading state
        self.position = None  # 'long', 'short', None
        self.entry_price = 0
        self.last_signal = None
        self.balance = {}
        
        # Strategy parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_signal_threshold = 0.001
        self.bb_squeeze_threshold = 0.02
        
        logger.info(f"Trading bot initialized for {self.symbol}")
    
    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection"""
        exchange_name = os.getenv('EXCHANGE', 'binance').lower()
        
        if exchange_name == 'binance':
            exchange = ccxt.binance({
                'apiKey': os.getenv('API_KEY'),
                'secret': os.getenv('API_SECRET'),
                'sandbox': True,  # Set to False for live trading
                'enableRateLimit': True,
            })
        else:
            raise ValueError(f"Exchange {exchange_name} not supported")
        
        return exchange
    
    def get_historical_data(self, limit: int = 100) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                limit=limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if df.empty:
            return df
        
        # Moving Averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd(df['close'])
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Stochastic
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> str:
        """Generate trading signals based on multiple strategies"""
        if len(df) < 50:
            return 'hold'
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        
        # Strategy 1: RSI + MACD
        if (latest['rsi'] < self.rsi_oversold and 
            latest['macd'] > latest['macd_signal'] and 
            prev['macd'] <= prev['macd_signal']):
            signals.append('buy')
        
        if (latest['rsi'] > self.rsi_overbought and 
            latest['macd'] < latest['macd_signal'] and 
            prev['macd'] >= prev['macd_signal']):
            signals.append('sell')
        
        # Strategy 2: Moving Average Crossover
        if (latest['sma_20'] > latest['sma_50'] and 
            prev['sma_20'] <= prev['sma_50'] and
            latest['rsi'] < 70):
            signals.append('buy')
        
        if (latest['sma_20'] < latest['sma_50'] and 
            prev['sma_20'] >= prev['sma_50'] and
            latest['rsi'] > 30):
            signals.append('sell')
        
        # Strategy 3: Bollinger Bands Mean Reversion
        if (latest['close'] < latest['bb_lower'] and 
            latest['rsi'] < 35 and
            latest['volume'] > latest['volume_sma']):
            signals.append('buy')
        
        if (latest['close'] > latest['bb_upper'] and 
            latest['rsi'] > 65 and
            latest['volume'] > latest['volume_sma']):
            signals.append('sell')
        
        # Strategy 4: Stochastic + Price Action
        if (latest['stoch_k'] < 20 and latest['stoch_d'] < 20 and
            latest['stoch_k'] > latest['stoch_d'] and
            latest['close'] > latest['sma_20']):
            signals.append('buy')
        
        if (latest['stoch_k'] > 80 and latest['stoch_d'] > 80 and
            latest['stoch_k'] < latest['stoch_d'] and
            latest['close'] < latest['sma_20']):
            signals.append('sell')
        
        # Aggregate signals
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        
        if buy_signals >= 2 and buy_signals > sell_signals:
            return 'buy'
        elif sell_signals >= 2 and sell_signals > buy_signals:
            return 'sell'
        else:
            return 'hold'
    
    def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    def place_order(self, side: str, amount: float, price: Optional[float] = None) -> Dict:
        """Place a market or limit order"""
        try:
            if price is None:
                # Market order
                order = self.exchange.create_market_order(
                    self.symbol, side, amount
                )
            else:
                # Limit order
                order = self.exchange.create_limit_order(
                    self.symbol, side, amount, price
                )
            
            logger.info(f"Order placed: {side} {amount} {self.symbol} at {price or 'market'}")
            return order
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {}
    
    def calculate_position_size(self, signal: str, current_price: float) -> float:
        """Calculate position size based on risk management"""
        balance = self.get_balance()
        
        if signal == 'buy':
            base_currency = self.symbol.split('/')[1]  # USDT in BTC/USDT
            available_balance = balance.get(base_currency, {}).get('free', 0)
            
            # Risk management: use only a percentage of available balance
            risk_amount = available_balance * 0.05  # 5% of balance
            position_size = min(risk_amount / current_price, self.trade_amount)
            
        elif signal == 'sell':
            base_currency = self.symbol.split('/')[0]  # BTC in BTC/USDT
            available_balance = balance.get(base_currency, {}).get('free', 0)
            position_size = min(available_balance, self.trade_amount)
        
        else:
            position_size = 0
        
        return position_size
    
    def check_stop_loss_take_profit(self, current_price: float) -> Optional[str]:
        """Check if stop loss or take profit should be triggered"""
        if self.position is None or self.entry_price == 0:
            return None
        
        if self.position == 'long':
            # Calculate stop loss and take profit for long position
            stop_loss_price = self.entry_price * (1 - self.stop_loss_percent / 100)
            take_profit_price = self.entry_price * (1 + self.take_profit_percent / 100)
            
            if current_price <= stop_loss_price:
                logger.warning(f"Stop loss triggered at {current_price}")
                return 'sell'
            elif current_price >= take_profit_price:
                logger.info(f"Take profit triggered at {current_price}")
                return 'sell'
        
        elif self.position == 'short':
            # Calculate stop loss and take profit for short position
            stop_loss_price = self.entry_price * (1 + self.stop_loss_percent / 100)
            take_profit_price = self.entry_price * (1 - self.take_profit_percent / 100)
            
            if current_price >= stop_loss_price:
                logger.warning(f"Stop loss triggered at {current_price}")
                return 'buy'
            elif current_price <= take_profit_price:
                logger.info(f"Take profit triggered at {current_price}")
                return 'buy'
        
        return None
    
    def execute_trade(self, signal: str, current_price: float) -> bool:
        """Execute trade based on signal"""
        try:
            if signal == 'buy' and self.position != 'long':
                # Close short position if exists
                if self.position == 'short':
                    position_size = self.calculate_position_size('buy', current_price)
                    if position_size > 0:
                        self.place_order('buy', position_size)
                
                # Open long position
                position_size = self.calculate_position_size('buy', current_price)
                if position_size > 0:
                    order = self.place_order('buy', position_size)
                    if order:
                        self.position = 'long'
                        self.entry_price = current_price
                        logger.info(f"Long position opened at {current_price}")
                        return True
            
            elif signal == 'sell' and self.position != 'short':
                # Close long position if exists
                if self.position == 'long':
                    position_size = self.calculate_position_size('sell', current_price)
                    if position_size > 0:
                        self.place_order('sell', position_size)
                
                # Open short position (if exchange supports it)
                position_size = self.calculate_position_size('sell', current_price)
                if position_size > 0:
                    order = self.place_order('sell', position_size)
                    if order:
                        self.position = 'short'
                        self.entry_price = current_price
                        logger.info(f"Short position opened at {current_price}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def run_strategy(self):
        """Main trading strategy execution"""
        try:
            # Get historical data
            df = self.get_historical_data()
            if df.empty:
                logger.warning("No data available")
                return
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Check stop loss / take profit first
            sl_tp_signal = self.check_stop_loss_take_profit(current_price)
            if sl_tp_signal:
                self.execute_trade(sl_tp_signal, current_price)
                self.position = None
                self.entry_price = 0
                return
            
            # Generate trading signals
            signal = self.generate_signals(df)
            
            # Execute trade if signal is different from last signal
            if signal != self.last_signal and signal != 'hold':
                logger.info(f"New signal: {signal} at price: {current_price}")
                
                # Print current indicators for debugging
                latest = df.iloc[-1]
                logger.info(f"RSI: {latest['rsi']:.2f}, MACD: {latest['macd']:.6f}, "
                           f"BB Position: {((current_price - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])):.2f}")
                
                self.execute_trade(signal, current_price)
                self.last_signal = signal
            
            # Log current status
            logger.info(f"Current position: {self.position}, Entry: {self.entry_price}, "
                       f"Current price: {current_price:.2f}, Signal: {signal}")
            
        except Exception as e:
            logger.error(f"Error in strategy execution: {e}")
    
    def start_trading(self):
        """Start the trading bot"""
        logger.info("Starting trading bot...")
        
        # Schedule the strategy to run every minute
        schedule.every(1).minutes.do(self.run_strategy)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Trading bot error: {e}")

def main():
    """Main function"""
    bot = TradingBot()
    
    # Test mode - run strategy once
    print("Testing strategy...")
    bot.run_strategy()
    
    # Uncomment to start live trading
    # bot.start_trading()

if __name__ == "__main__":
    main()