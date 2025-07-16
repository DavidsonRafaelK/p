"""
Trading Bot Configuration and Strategy Settings
"""

class TradingConfig:
    # Exchange settings
    EXCHANGE = 'binance'
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '1m'
    
    # Risk management
    TRADE_AMOUNT = 0.001  # Base trade amount
    STOP_LOSS_PERCENT = 2.0  # Stop loss percentage
    TAKE_PROFIT_PERCENT = 3.0  # Take profit percentage
    MAX_RISK_PERCENT = 5.0  # Maximum risk per trade (% of balance)
    
    # Technical indicators settings
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    RSI_WINDOW = 14
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    BB_WINDOW = 20
    BB_STD = 2
    
    SMA_SHORT = 20
    SMA_LONG = 50
    
    STOCH_K = 14
    STOCH_D = 3
    
    # Strategy weights (for signal aggregation)
    STRATEGY_WEIGHTS = {
        'rsi_macd': 0.3,
        'ma_crossover': 0.25,
        'bollinger_bands': 0.25,
        'stochastic': 0.2
    }
    
    # Minimum signals required for trade
    MIN_SIGNALS_FOR_TRADE = 2
    
    # Trading session settings
    TRADING_HOURS = {
        'start': 0,  # 24-hour format
        'end': 23
    }
    
    # Backtesting settings
    BACKTEST_DAYS = 30
    INITIAL_BALANCE = 1000  # USDT
