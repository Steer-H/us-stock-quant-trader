#!/usr/bin/env python3
"""
⚠️ 离线演示模式 - 持续观察脚本（所有数据为合成数据）

⚠️ 警告：此脚本生成的全部价格、预测和交易信号均为随机合成数据，
不可用于真实交易决策！仅供系统演示和UI调试使用。

每分钟刷新一次，模拟价格小幅波动，
供用户观察系统运行和UI表现。

用法:
    python live_trading/run_watch.py
"""

import sys
import time
import random
random.seed(42)  # 确定性mock数据
import signal
from pathlib import Path

# 确保项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_trading import LiveSimulator, TradingDashboard
from live_trading.portfolio import PortfolioManager

# 标记是否退出
_running = True

def on_signal(sig, frame):
    global _running
    _running = False
    print("\n正在优雅退出...")

signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)


def main():
    """持续观察主循环"""
    print(f"\033[91m{'='*60}\033[0m")
    print(f"\033[91m  ⚠️  警告：所有数据为合成数据，不可用于真实交易！\033[0m")
    print(f"\033[91m  ⚠️  仅供系统演示和UI调试使用\033[0m")
    print(f"\033[96m{'='*60}\033[0m")
    print(f"\033[1m  美股量化交易系统 - 离线演示模式\033[0m")
    print(f"\033[2m  每分钟自动刷新，Ctrl+C 退出\033[0m")
    print(f"\033[96m{'='*60}\033[0m\n")
    
    # 创建模拟器
    sim = LiveSimulator(
        tickers=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX'],
        initial_capital=100_000.0
    )
    
    # 初始化模拟持仓
    mock_positions = {
        'AAPL':  {'qty': 50, 'cost': 175.0},
        'MSFT':  {'qty': 30, 'cost': 380.0},
        'NVDA':  {'qty': 40, 'cost': 850.0},
        'GOOGL': {'qty': 20, 'cost': 140.0},
        'TSLA':  {'qty': 25, 'cost': 220.0},
        'META':  {'qty': 15, 'cost': 480.0},
        'NFLX':  {'qty': 10, 'cost': 620.0},
    }
    
    # 建仓
    for ticker, info in mock_positions.items():
        sim.portfolio.execute_buy(
            ticker, info['qty'], info['cost'],
            commission=1.0, reason='初始建仓'
        )
        sim._current_prices[ticker] = info['cost']
    
    # 纳指基准价
    sim._current_prices['^IXIC'] = 18200.0
    sim.benchmark.nasdaq_start_price = 17800.0
    sim.benchmark.nasdaq_shares = 100000.0 / 17800.0
    sim.benchmark.nasdaq_prices = None  # 重置
    
    # 生成一些历史预测
    for _ in range(40):
        ticker = random.choice(list(mock_positions.keys()))
        actual_dir = 1 if random.random() < 0.52 else 0
        pred_id = sim.accuracy_tracker.record_prediction(
            ticker,
            random.uniform(-0.02, 0.03),
            1 if random.random() < 0.55 else 0,
            random.uniform(0.5, 0.9)
        )
        sim.accuracy_tracker.confirm_prediction(
            pred_id,
            random.uniform(-0.02, 0.03),
            actual_dir
        )
    
    # 创建仪表盘
    dash = TradingDashboard(sim)
    
    iteration = 0
    
    while _running:
        iteration += 1
        
        # 模拟价格波动（±0.5%以内）
        for ticker in sim._current_prices:
            prev = sim._current_prices[ticker]
            change = random.uniform(-0.005, 0.005)
            sim._previous_prices[ticker] = prev
            sim._current_prices[ticker] = prev * (1 + change)
        
        # 更新纳指
        nasdaq_change = random.uniform(-0.003, 0.004)
        sim._current_prices['^IXIC'] *= (1 + nasdaq_change)
        
        # 更新持仓
        sim.portfolio.update_prices({
            t: p for t, p in sim._current_prices.items() if t != '^IXIC'
        })
        
        # 更新基准
        sim.benchmark.update(
            sim._current_prices['^IXIC'],
            sim.portfolio.get_total_equity()
        )
        
        # 每5次迭代做一笔模拟交易
        if iteration % 5 == 0 and sim.portfolio.cash > 5000:
            tickers_with_pnl = []
            for t, pos in sim.portfolio.positions.items():
                if pos.unrealized_pnl_pct > 0.08:
                    tickers_with_pnl.append(('sell', t, pos))
                elif pos.unrealized_pnl_pct < -0.06:
                    tickers_with_pnl.append(('stop', t, pos))
            
            if tickers_with_pnl:
                action, ticker, pos = random.choice(tickers_with_pnl)
                price = sim._current_prices[ticker]
                sim.portfolio.execute_sell(
                    ticker, pos.quantity, price,
                    commission=1.0,
                    reason=f'止盈' if action == 'sell' else '止损'
                )
                
                # 买入新股票替换
                available = [t for t in mock_positions if t not in sim.portfolio.positions]
                if not available:
                    available = list(mock_positions.keys())
                new_ticker = random.choice(available)
                new_price = sim._current_prices[new_ticker]
                qty = min(int(sim.portfolio.cash * 0.1 / new_price), 100)
                if qty > 0:
                    sim.portfolio.execute_buy(
                        new_ticker, qty, new_price,
                        commission=1.0,
                        reason='SYNTHETIC_DEMO'
                    )
        
        # 每3次迭代生成一条新预测
        if iteration % 3 == 0:
            ticker = random.choice(list(mock_positions.keys()))
            pred_id = sim.accuracy_tracker.record_prediction(
                ticker,
                random.uniform(-0.02, 0.03),
                1 if random.random() < 0.55 else 0,
                random.uniform(0.5, 0.85)
            )
            # 有时确认之前的预测
            sim.accuracy_tracker.confirm_prediction(
                pred_id,
                random.uniform(-0.02, 0.03),
                1 if random.random() < 0.53 else 0
            )
        
        # 渲染面板
        dash.render_once()
        
        # 等待60秒
        for _ in range(60):
            if not _running:
                break
            time.sleep(1)
    
    print(f"\n\033[92m观察结束，共刷新 {iteration} 次\033[0m")


if __name__ == '__main__':
    main()
