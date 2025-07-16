"""
Risk Management Module for Trading Bot
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, initial_balance: float = 1000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_risk_per_trade = 0.02  # 2% of balance
        self.max_drawdown_limit = 0.15  # 15% maximum drawdown
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.position_size_limit = 0.1  # 10% of balance per position
        
        # Risk tracking
        self.daily_pnl = 0
        self.peak_balance = initial_balance
        self.current_drawdown = 0
        self.open_positions = {}
        
    def calculate_position_size(self, 
                              entry_price: float, 
                              stop_loss_price: float, 
                              confidence: float = 1.0) -> float:
        """
        Calculate position size based on Kelly Criterion and risk management
        """
        # Calculate risk amount
        risk_amount = self.current_balance * self.max_risk_per_trade
        
        # Calculate position size based on stop loss distance
        stop_loss_distance = abs(entry_price - stop_loss_price)
        
        if stop_loss_distance == 0:
            return 0
        
        # Base position size
        base_position_size = risk_amount / stop_loss_distance
        
        # Adjust for confidence level
        adjusted_position_size = base_position_size * confidence
        
        # Apply position size limit
        max_position_value = self.current_balance * self.position_size_limit
        max_position_size = max_position_value / entry_price
        
        final_position_size = min(adjusted_position_size, max_position_size)
        
        logger.info(f"Position size calculated: {final_position_size:.6f}")
        return final_position_size
    
    def kelly_criterion(self, 
                       win_rate: float, 
                       avg_win: float, 
                       avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        """
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly (more conservative)
        conservative_kelly = kelly_fraction * 0.25  # Use 25% of Kelly
        
        # Ensure it's within reasonable bounds
        return max(0, min(conservative_kelly, 0.1))  # Max 10% of balance
    
    def can_open_position(self, position_type: str, size: float) -> bool:
        """
        Check if we can open a new position based on risk limits
        """
        # Check daily loss limit
        if self.daily_pnl < -self.current_balance * self.daily_loss_limit:
            logger.warning("Daily loss limit reached")
            return False
        
        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_limit:
            logger.warning("Maximum drawdown limit reached")
            return False
        
        # Check if we have enough balance
        if size <= 0:
            logger.warning("Invalid position size")
            return False
        
        # Check correlation with existing positions
        if self.check_correlation_risk(position_type):
            logger.warning("Correlation risk too high")
            return False
        
        return True
    
    def check_correlation_risk(self, position_type: str) -> bool:
        """
        Check if opening new position would create too much correlation risk
        """
        # Simple correlation check - limit number of positions in same direction
        same_direction_positions = sum(1 for pos in self.open_positions.values() 
                                     if pos['type'] == position_type)
        
        # Maximum 3 positions in same direction
        return same_direction_positions >= 3
    
    def update_balance(self, new_balance: float):
        """Update current balance and risk metrics"""
        self.current_balance = new_balance
        
        # Update peak balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        # Calculate current drawdown
        self.current_drawdown = (self.peak_balance - new_balance) / self.peak_balance
        
        # Update daily PnL (simplified - in real implementation, track by day)
        self.daily_pnl = new_balance - self.initial_balance
    
    def add_position(self, position_id: str, position_info: Dict):
        """Add position to tracking"""
        self.open_positions[position_id] = position_info
        logger.info(f"Position {position_id} added to tracking")
    
    def remove_position(self, position_id: str):
        """Remove position from tracking"""
        if position_id in self.open_positions:
            del self.open_positions[position_id]
            logger.info(f"Position {position_id} removed from tracking")
    
    def calculate_portfolio_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio
        """
        # Simplified VaR calculation
        # In real implementation, use historical returns
        returns_std = 0.02  # Assume 2% daily volatility
        
        # Z-score for confidence level
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z_score = z_scores.get(confidence_level, 1.65)
        
        # VaR calculation
        var = self.current_balance * returns_std * z_score
        
        return var
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': self.current_drawdown,
            'daily_pnl': self.daily_pnl,
            'open_positions': len(self.open_positions),
            'risk_per_trade': self.max_risk_per_trade,
            'var_95': self.calculate_portfolio_var(0.95),
            'available_risk': self.current_balance * self.max_risk_per_trade
        }
    
    def emergency_stop(self) -> bool:
        """
        Check if emergency stop should be triggered
        """
        # Trigger emergency stop if drawdown exceeds limit
        if self.current_drawdown > self.max_drawdown_limit:
            logger.critical("EMERGENCY STOP: Maximum drawdown exceeded")
            return True
        
        # Trigger if daily loss exceeds limit
        if self.daily_pnl < -self.current_balance * self.daily_loss_limit:
            logger.critical("EMERGENCY STOP: Daily loss limit exceeded")
            return True
        
        return False
    
    def adjust_risk_parameters(self, performance_metrics: Dict):
        """
        Dynamically adjust risk parameters based on performance
        """
        win_rate = performance_metrics.get('win_rate', 0.5)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        
        # Increase risk if performing well
        if win_rate > 0.6 and sharpe_ratio > 1.0:
            self.max_risk_per_trade = min(0.03, self.max_risk_per_trade * 1.1)
            logger.info("Risk increased due to good performance")
        
        # Decrease risk if performing poorly
        elif win_rate < 0.4 or sharpe_ratio < 0:
            self.max_risk_per_trade = max(0.01, self.max_risk_per_trade * 0.9)
            logger.info("Risk decreased due to poor performance")
    
    def log_risk_status(self):
        """Log current risk status"""
        metrics = self.get_risk_metrics()
        
        logger.info("=== RISK STATUS ===")
        logger.info(f"Balance: ${metrics['current_balance']:,.2f}")
        logger.info(f"Drawdown: {metrics['current_drawdown']:.2%}")
        logger.info(f"Daily PnL: ${metrics['daily_pnl']:,.2f}")
        logger.info(f"Open Positions: {metrics['open_positions']}")
        logger.info(f"VaR (95%): ${metrics['var_95']:,.2f}")
        logger.info(f"Risk per Trade: {metrics['risk_per_trade']:.2%}")
        logger.info("==================")

# Example usage
if __name__ == "__main__":
    risk_manager = RiskManager(initial_balance=1000)
    
    # Example position size calculation
    position_size = risk_manager.calculate_position_size(
        entry_price=50000,
        stop_loss_price=49000,
        confidence=0.8
    )
    
    print(f"Calculated position size: {position_size}")
    
    # Example risk metrics
    risk_manager.log_risk_status()
