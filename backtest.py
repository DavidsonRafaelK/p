"""
Backtesting module for the trading bot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
from main import TradingBot

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, bot: TradingBot, days: int = 30) -> Dict:
        """Run backtest on historical data"""
        logger.info(f"Starting backtest for {days} days")
        
        # Get historical data
        df = bot.get_historical_data(limit=days * 24 * 60)  # minutes in days
        if df.empty:
            logger.error("No historical data available")
            return {}
        
        # Calculate indicators
        df = bot.calculate_indicators(df)
        
        # Initialize variables
        position = None
        entry_price = 0
        position_size = 0
        current_balance = self.initial_balance
        
        # Simulate trading
        for i in range(50, len(df)):  # Start after indicators are calculated
            current_data = df.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            
            # Generate signal
            signal = bot.generate_signals(current_data)
            
            # Execute trade logic
            if signal == 'buy' and position != 'long':
                if position == 'short':
                    # Close short position
                    pnl = (entry_price - current_price) * position_size
                    current_balance += pnl
                    self.trades.append({
                        'type': 'close_short',
                        'price': current_price,
                        'size': position_size,
                        'pnl': pnl,
                        'balance': current_balance,
                        'timestamp': current_data.index[-1]
                    })
                
                # Open long position
                position_size = (current_balance * 0.05) / current_price  # 5% of balance
                position = 'long'
                entry_price = current_price
                
                self.trades.append({
                    'type': 'open_long',
                    'price': current_price,
                    'size': position_size,
                    'pnl': 0,
                    'balance': current_balance,
                    'timestamp': current_data.index[-1]
                })
            
            elif signal == 'sell' and position != 'short':
                if position == 'long':
                    # Close long position
                    pnl = (current_price - entry_price) * position_size
                    current_balance += pnl
                    self.trades.append({
                        'type': 'close_long',
                        'price': current_price,
                        'size': position_size,
                        'pnl': pnl,
                        'balance': current_balance,
                        'timestamp': current_data.index[-1]
                    })
                
                # Open short position
                position_size = (current_balance * 0.05) / current_price  # 5% of balance
                position = 'short'
                entry_price = current_price
                
                self.trades.append({
                    'type': 'open_short',
                    'price': current_price,
                    'size': position_size,
                    'pnl': 0,
                    'balance': current_balance,
                    'timestamp': current_data.index[-1]
                })
            
            # Check stop loss / take profit
            if position == 'long':
                stop_loss = entry_price * 0.98  # 2% stop loss
                take_profit = entry_price * 1.03  # 3% take profit
                
                if current_price <= stop_loss or current_price >= take_profit:
                    pnl = (current_price - entry_price) * position_size
                    current_balance += pnl
                    self.trades.append({
                        'type': 'close_long_sl_tp',
                        'price': current_price,
                        'size': position_size,
                        'pnl': pnl,
                        'balance': current_balance,
                        'timestamp': current_data.index[-1]
                    })
                    position = None
                    entry_price = 0
                    position_size = 0
            
            elif position == 'short':
                stop_loss = entry_price * 1.02  # 2% stop loss
                take_profit = entry_price * 0.97  # 3% take profit
                
                if current_price >= stop_loss or current_price <= take_profit:
                    pnl = (entry_price - current_price) * position_size
                    current_balance += pnl
                    self.trades.append({
                        'type': 'close_short_sl_tp',
                        'price': current_price,
                        'size': position_size,
                        'pnl': pnl,
                        'balance': current_balance,
                        'timestamp': current_data.index[-1]
                    })
                    position = None
                    entry_price = 0
                    position_size = 0
            
            # Record equity curve
            self.equity_curve.append({
                'timestamp': current_data.index[-1],
                'balance': current_balance,
                'price': current_price
            })
        
        # Calculate performance metrics
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_return = ((equity_df['balance'].iloc[-1] - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate Sharpe ratio
        returns = equity_df['balance'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Maximum drawdown
        running_max = equity_df['balance'].expanding().max()
        drawdown = (equity_df['balance'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Average win/loss
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_balance': equity_df['balance'].iloc[-1]
        }
    
    def plot_results(self):
        """Plot backtest results"""
        if not self.equity_curve:
            print("No data to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot equity curve
        ax1.plot(equity_df['timestamp'], equity_df['balance'], label='Portfolio Value', color='blue')
        ax1.axhline(y=self.initial_balance, color='red', linestyle='--', label='Initial Balance')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Balance (USDT)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot price
        ax2.plot(equity_df['timestamp'], equity_df['price'], label='Price', color='green')
        ax2.set_title('Price Movement')
        ax2.set_ylabel('Price (USDT)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, metrics: Dict):
        """Print backtest summary"""
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${metrics['final_balance']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Average Win: ${metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print("="*50)

def run_backtest():
    """Run backtest"""
    bot = TradingBot()
    backtester = Backtester(initial_balance=1000)
    
    # Run backtest
    metrics = backtester.run_backtest(bot, days=30)
    
    if metrics:
        # Print results
        backtester.print_summary(metrics)
        
        # Plot results
        backtester.plot_results()
    
    return metrics

if __name__ == "__main__":
    run_backtest()
