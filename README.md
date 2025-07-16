# Trading Bot Setup and Usage Guide

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Trading account** (Binance, Coinbase, etc.)
3. **API credentials** from your exchange

## 🚀 Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure environment variables:**
   Edit `.env` file and add your API credentials:

```
API_KEY=your_binance_api_key
API_SECRET=your_binance_api_secret
EXCHANGE=binance
SYMBOL=BTC/USDT
```

## 🎯 Features

### 1. **Multiple Trading Strategies**

- **Momentum Strategy**: RSI + MACD + Price momentum
- **Mean Reversion**: Bollinger Bands + RSI oversold/overbought
- **Breakout Strategy**: Volume + Volatility breakouts
- **Scalping**: High-frequency small profits
- **Swing Trading**: Longer-term trend following
- **Grid Trading**: Buy low, sell high in ranges

### 2. **Risk Management**

- **Position sizing** based on Kelly Criterion
- **Stop loss** and **Take profit** automation
- **Maximum drawdown** protection
- **Daily loss limits**
- **Portfolio correlation** checks

### 3. **Technical Indicators**

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Stochastic Oscillator
- Volume indicators

### 4. **Backtesting**

- Historical performance analysis
- Performance metrics calculation
- Equity curve visualization
- Risk-adjusted returns

## 📊 Usage Examples

### Basic Usage

```python
from main import TradingBot

# Initialize bot
bot = TradingBot()

# Run strategy once (testing)
bot.run_strategy()

# Start live trading
bot.start_trading()
```

### Advanced Strategy Usage

```python
from strategies import AdvancedStrategies
from main import TradingBot

bot = TradingBot()
strategies = AdvancedStrategies()

# Get data
df = bot.get_historical_data()

# Get signals from multiple strategies
results = strategies.get_strategy_signals(df, [
    'momentum',
    'mean_reversion',
    'breakout'
])

# Aggregate signals
final_signal = strategies.aggregate_signals(results)
print(f"Final signal: {final_signal}")
```

### Backtesting

```python
from backtest import run_backtest

# Run 30-day backtest
metrics = run_backtest()

# Results will show:
# - Total return
# - Win rate
# - Sharpe ratio
# - Maximum drawdown
# - Profit factor
```

### Risk Management

```python
from risk_manager import RiskManager

# Initialize risk manager
risk_manager = RiskManager(initial_balance=1000)

# Calculate position size
position_size = risk_manager.calculate_position_size(
    entry_price=50000,
    stop_loss_price=49000,
    confidence=0.8
)

# Check if can open position
can_trade = risk_manager.can_open_position('long', position_size)
```

## ⚙️ Configuration

### Environment Variables (.env)

```
API_KEY=your_api_key
API_SECRET=your_api_secret
EXCHANGE=binance
SYMBOL=BTC/USDT
TIMEFRAME=1m
TRADE_AMOUNT=0.001
STOP_LOSS_PERCENT=2.0
TAKE_PROFIT_PERCENT=3.0
```

### Strategy Parameters (config.py)

```python
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
STOP_LOSS_PERCENT = 2.0
TAKE_PROFIT_PERCENT = 3.0
MAX_RISK_PERCENT = 5.0
```

## 🔧 Bot Commands

### Testing Mode

```bash
python main.py  # Run once for testing
```

### Live Trading Mode

```python
# In main.py, uncomment:
bot.start_trading()
```

### Backtest Mode

```bash
python backtest.py
```

## 📈 Performance Metrics

The bot tracks:

- **Total Return**: Overall profit/loss percentage
- **Win Rate**: Percentage of winning trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Average profit per winning/losing trade

## 🛡️ Risk Management Features

1. **Position Sizing**: Uses Kelly Criterion for optimal position size
2. **Stop Loss**: Automatic stop loss at configurable percentage
3. **Take Profit**: Automatic profit taking at target levels
4. **Daily Loss Limit**: Stops trading if daily loss exceeds limit
5. **Maximum Drawdown**: Emergency stop if drawdown too high
6. **Correlation Check**: Prevents overexposure to correlated positions

## 📊 Supported Exchanges

- **Binance** (Primary)
- **Coinbase Pro**
- **Kraken**
- **Bitfinex**
- **Huobi**

(Add your exchange API credentials in the `_init_exchange` method)

## 🚨 Important Notes

1. **Paper Trading**: Set `sandbox=True` in exchange config for testing
2. **Live Trading**: Only use with funds you can afford to lose
3. **API Security**: Never share your API keys
4. **Risk Management**: Always use stop losses and position sizing
5. **Backtesting**: Test strategies thoroughly before live trading

## 📝 Logging

The bot logs all activities to:

- Console output
- `trading_bot.log` file

Log levels:

- INFO: General information
- WARNING: Risk warnings
- ERROR: Errors and exceptions
- CRITICAL: Emergency stops

## 🔍 Monitoring

Monitor your bot through:

- Log files
- Exchange account balance
- Position tracking
- Performance metrics
- Risk status updates

## 🛠️ Troubleshooting

### Common Issues:

1. **API Connection**: Check API keys and permissions
2. **Insufficient Balance**: Ensure adequate funds
3. **Rate Limits**: Bot includes rate limiting
4. **Data Issues**: Check internet connection and exchange status

### Debug Mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

1. **Strategy Combination**: Use multiple strategies for better signals
2. **Parameter Tuning**: Adjust RSI, MACD, and other parameters
3. **Timeframe Optimization**: Test different timeframes
4. **Risk Adjustment**: Optimize position sizing and stop losses

## 🔄 Updates and Maintenance

- Monitor bot performance regularly
- Update strategies based on market conditions
- Adjust risk parameters as needed
- Keep dependencies updated

## 📞 Support

For issues or questions:

1. Check logs for error messages
2. Review configuration settings
3. Test in sandbox mode first
4. Verify API permissions

---

**⚠️ DISCLAIMER**: Trading cryptocurrencies involves significant risk. This bot is for educational purposes. Always do your own research and never risk more than you can afford to lose.
