"""
在线模拟交易系统 - 实时交易仪表盘

提供清晰、专业的终端实时展示面板，包含：

┌─────────────────────────────────────────────────────────────┐
│          美股量化交易系统 - 在线模拟交易                       │
│  市场状态 │ 倒计时 │ 当前时间                                 │
├─────────────────────────────────────────────────────────────┤
│  💰 账户概览                                                 │
│  初始资金 │ 总资产 │ 现金 │ 持仓市值                          │
├─────────────────────────────────────────────────────────────┤
│  📊 盈亏统计                                                 │
│  总盈亏 │ 盈亏% │ 当日盈亏 │ 已实现/未实现                     │
├─────────────────────────────────────────────────────────────┤
│  📈 持仓明细（股票/数量/成本价/现价/盈亏金额/盈亏%/权重）      │
├─────────────────────────────────────────────────────────────┤
│  🎯 模型准确率                                                │
│  方向准确率 │ RMSE │ 最近50次 │ 趋势                          │
├─────────────────────────────────────────────────────────────┤
│  🏦 纳指基准对比                                              │
│  策略收益 vs 纳指收益 │ Alpha │ Beta │ Sharpe                  │
├─────────────────────────────────────────────────────────────┤
│  📋 最近交易                                                  │
└─────────────────────────────────────────────────────────────┘

刷新频率：每分钟自动刷新
"""

import os
import random
import sys
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from collections import deque

from live_trading.market_clock import MarketClock, MarketStatus
from live_trading.portfolio import PortfolioSnapshot, PortfolioManager
from live_trading.benchmark import BenchmarkSnapshot
from live_trading.accuracy_tracker import AccuracySnapshot
from live_trading.live_simulator import LiveSimulator
from utils.helpers import format_currency, format_pct

logger = logging.getLogger(__name__)


# ============================================================================
# 颜色常量（终端ANSI）
# ============================================================================
class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # 前景色
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # 背景色
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'
    
    @staticmethod
    def pnl_color(value: float) -> str:
        """根据盈亏返回颜色"""
        if value > 0:
            return Color.GREEN
        elif value < 0:
            return Color.RED
        return Color.WHITE
    
    @staticmethod
    def pnl_str(value: float, as_pct: bool = False) -> str:
        """带颜色的盈亏字符串"""
        sign = '+' if value > 0 else ''
        if as_pct:
            s = f"{sign}{value:.2%}"
        else:
            s = f"{sign}{value:,.2f}"
        color = Color.pnl_color(value)
        return f"{color}{s}{Color.RESET}"


# ============================================================================
# 仪表盘渲染器
# ============================================================================
class TradingDashboard:
    """
    实时交易仪表盘
    
    以固定的终端区域刷新显示所有关键信息。
    支持自动刷新和手动刷新两种模式。
    
    使用:
        dash = TradingDashboard(simulator)
        dash.run_auto_refresh()  # 自动刷新模式
        # 或
        dash.render_once()       # 单次渲染
    """
    
    WIDTH = 85  # 显示宽度
    
    def __init__(self, simulator: LiveSimulator):
        """
        参数:
            simulator: LiveSimulator实例
        """
        self.simulator = simulator
        self.clock = MarketClock()
        self._render_count: int = 0
        
        # 注册为模拟器的刷新回调
        self.simulator.add_refresh_callback(lambda sim: self.render_once())
    
    def clear_screen(self) -> None:
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def render_once(self) -> None:
        """渲染一次面板"""
        self._render_count += 1
        
        # 获取完整快照
        snapshot = self.simulator.get_full_snapshot()
        
        market = snapshot['market']
        portfolio = snapshot['portfolio']
        benchmark = snapshot['benchmark']
        accuracy = snapshot['accuracy']
        
        self.clear_screen()
        
        # 逐段渲染
        self._render_header(market)
        print()
        self._render_account_overview(portfolio)
        print()
        self._render_pnl_summary(portfolio, benchmark)
        print()
        self._render_positions(portfolio)
        print()
        self._render_model_accuracy(accuracy)
        print()
        self._render_benchmark_comparison(benchmark, portfolio)
        print()
        self._render_recent_trades()
        print()
        self._render_footer()
    
    def _render_header(self, market: Dict) -> None:
        """渲染顶部标题和状态栏"""
        status = market.get('status', MarketStatus.CLOSED)
        desc = market.get('description', '闭市')
        current_et = market.get('current_et', '')
        is_open = market.get('is_open', False)
        
        # 状态图标
        status_icons = {
            MarketStatus.REGULAR_HOURS: '🟢',
            MarketStatus.PRE_MARKET: '🟡',
            MarketStatus.AFTER_HOURS: '🔵',
            MarketStatus.CLOSED: '⚫',
            MarketStatus.WEEKEND: '🔴',
            MarketStatus.HOLIDAY: '🔴',
        }
        icon = status_icons.get(status, '⚫')
        
        # 顶栏
        print(f"{Color.BOLD}{Color.CYAN}{'═' * self.WIDTH}{Color.RESET}")
        print(f"{Color.BOLD}  美股量化交易系统 - 在线模拟交易  {Color.DIM}v1.0{Color.RESET}")
        print(f"{Color.CYAN}{'═' * self.WIDTH}{Color.RESET}")
        
        # 状态行
        status_color = Color.GREEN if is_open else Color.YELLOW
        
        line = f"  {icon} 市场状态: {status_color}{desc}{Color.RESET}"
        
        # 倒计时
        if not market.get('is_active', False):
            cd = market.get('countdown_to_open')
            if cd:
                h, m, s = cd
                line += f"  │  ⏳ 距离开市还有: {Color.BOLD}{h}h {m:02d}m {s:02d}s{Color.RESET}"
        elif status == MarketStatus.REGULAR_HOURS:
            cd = market.get('countdown_to_close')
            if cd:
                h, m, s = cd
                line += f"  │  ⏳ 距离闭市还有: {Color.YELLOW}{h}h {m:02d}m {s:02d}s{Color.RESET}"
        
        line += f"  │  🕐 {current_et}"
        print(line)
        
        # 活跃交易时段提示
        if is_open:
            print(f"  {Color.GREEN}▶ 交易进行中... (每分钟自动刷新){Color.RESET}")
    
    def _render_account_overview(self, portfolio: PortfolioSnapshot) -> None:
        """渲染账户概览"""
        print(f"{Color.BOLD}💰 账户概览{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        cols = [
            ('初始资金', f"${portfolio.initial_capital:,.2f}"),
            ('总净资产', f"{Color.BOLD}${portfolio.total_equity:,.2f}{Color.RESET}"),
            ('现金余额', f"${portfolio.cash:,.2f}"),
            ('持仓市值', f"${portfolio.total_market_value:,.2f}"),
        ]
        
        line = '  '
        for label, value in cols:
            line += f"{label}: {value}  "
        
        print(line)
        print(f"  {'持仓数量':<10}: {portfolio.position_count} 只  │  "
              f"{'杠杆':<6}: {portfolio.leverage:.2f}x")
    
    def _render_pnl_summary(self, portfolio: PortfolioSnapshot, 
                             benchmark: BenchmarkSnapshot) -> None:
        """渲染盈亏统计"""
        print(f"{Color.BOLD}📊 盈亏统计{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        total_pnl = portfolio.total_pnl
        total_pnl_pct = portfolio.total_pnl_pct
        day_pnl = portfolio.day_pnl
        day_pnl_pct = portfolio.day_pnl_pct
        
        # 总盈亏
        line = f"  {'总盈亏:':<10} {Color.pnl_str(total_pnl)} ({Color.pnl_str(total_pnl_pct, True)})"
        line += f"  │  {'当日盈亏:':<10} {Color.pnl_str(day_pnl)} ({Color.pnl_str(day_pnl_pct, True)})"
        print(line)
        
        # 已实现/未实现
        line = f"  {'已实现:':<10} {Color.pnl_str(portfolio.realized_pnl)}"
        line += f"  │  {'未实现:':<10} {Color.pnl_str(portfolio.unrealized_pnl)}"
        
        # 最大回撤
        dd = portfolio.max_drawdown_pct
        dd_color = Color.YELLOW if dd < -0.1 else (Color.RED if dd < -0.2 else Color.WHITE)
        line += f"  │  {'最大回撤:':<10} {dd_color}{dd:+.2%}{Color.RESET}"
        print(line)
    
    def _render_positions(self, portfolio: PortfolioSnapshot) -> None:
        """渲染持仓明细表格"""
        positions = portfolio.positions
        
        print(f"{Color.BOLD}📈 持仓明细{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        if not positions:
            print(f"  {Color.GRAY}暂无持仓{Color.RESET}")
            return
        
        # 表头
        header = (f"  {'股票':<7} {'数量':>8} {'成本价':>10} {'现价':>10} "
                  f"{'市值':>12} {'盈亏金额':>10} {'盈亏%':>8} {'权重':>6}")
        print(f"{Color.DIM}{header}{Color.RESET}")
        print(f"  {Color.DIM}{'─' * 72}{Color.RESET}")
        
        # 按权重排序
        sorted_positions = sorted(
            positions.values(), 
            key=lambda p: abs(p.market_value), 
            reverse=True
        )
        
        for pos in sorted_positions[:20]:  # 最多显示20只
            pnl_str = Color.pnl_str(pos.unrealized_pnl)
            pnl_pct_str = Color.pnl_str(pos.unrealized_pnl_pct, True)
            
            line = (f"  {pos.ticker:<7} {pos.quantity:>8} "
                    f"${pos.avg_cost:>9.2f} ${pos.current_price:>9.2f} "
                    f"${pos.market_value:>11,.0f} "
                    f"{pnl_str} {pnl_pct_str} "
                    f"{pos.weight:>5.1%}")
            print(line)
        
        if len(sorted_positions) > 20:
            print(f"  {Color.GRAY}... 还有 {len(sorted_positions) - 20} 只持仓{Color.RESET}")
    
    def _render_model_accuracy(self, accuracy: AccuracySnapshot) -> None:
        """渲染模型准确率"""
        print(f"{Color.BOLD}🎯 模型预测准确率{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        dir_acc = accuracy.direction_accuracy
        recent_acc = accuracy.recent_accuracy_50
        
        # 准确率状态
        acc_color = Color.GREEN if dir_acc >= 0.55 else (Color.YELLOW if dir_acc >= 0.50 else Color.RED)
        recent_color = Color.GREEN if recent_acc >= 0.55 else Color.YELLOW
        
        line = f"  {'方向准确率:':<12} {acc_color}{dir_acc:.1%}{Color.RESET}"
        line += f"  │  {'最近50次:':<10} {recent_color}{recent_acc:.1%}{Color.RESET}"
        line += f"  │  {'RMSE:':<8} {accuracy.rmse:.6f}"
        line += f"  │  {'MAE:':<8} {accuracy.mae:.6f}"
        print(line)
        
        # 细分统计
        total = accuracy.total_long + accuracy.total_short
        long_acc = accuracy.correct_long / max(accuracy.total_long, 1)
        short_acc = accuracy.correct_short / max(accuracy.total_short, 1)
        
        line = f"  {'总预测:':<10} {accuracy.total_predictions}"
        line += f"  │  {'已确认:':<10} {accuracy.confirmed_predictions}"
        line += f"  │  {'涨正确:':<10} {accuracy.correct_long}/{accuracy.total_long} ({long_acc:.0%})"
        line += f"  │  {'跌正确:':<10} {accuracy.correct_short}/{accuracy.total_short} ({short_acc:.0%})"
        print(line)
        
        # 趋势
        trend = accuracy.accuracy_trend
        trend_icon = {'stable': '➡️ 稳定', 'degrading': '⚠️ 下降', 'improving': '📈 上升'}.get(trend, '➡️ 稳定')
        status_icon = '✅ 可接受' if accuracy.is_acceptable else '⚠️ 需调优'
        print(f"  {'模型状态:':<10} {status_icon}  │  {'趋势:':<8} {trend_icon}")
    
    def _render_benchmark_comparison(self, benchmark: BenchmarkSnapshot, 
                                      portfolio: PortfolioSnapshot) -> None:
        """渲染纳指基准对比"""
        print(f"{Color.BOLD}🏦 纳指(^IXIC)基准对比{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        # 对比表格
        header = (f"  {'':<12} {'策略':>16} {'纳指':>16} {'差异':>16}")
        print(f"{Color.DIM}{header}{Color.RESET}")
        print(f"  {Color.DIM}{'─' * 62}{Color.RESET}")
        
        # 累计收益
        strat_ret = benchmark.strategy_return_pct
        nasdaq_ret = benchmark.nasdaq_return_pct
        diff_ret = benchmark.excess_return
        
        def compare_row(label, strat_val, bench_val, diff_val, fmt='pct'):
            if fmt == 'pct':
                s = f"{strat_val:+.2%}" if isinstance(strat_val, float) else str(strat_val)
                b = f"{bench_val:+.2%}" if isinstance(bench_val, float) else str(bench_val)
                d = f"{diff_val:+.2%}" if isinstance(diff_val, float) else str(diff_val)
            elif fmt == 'money':
                s = f"${strat_val:,.0f}" if isinstance(strat_val, (int, float)) else str(strat_val)
                b = f"${bench_val:,.0f}" if isinstance(bench_val, (int, float)) else str(bench_val)
                d = f"${diff_val:+,.0f}" if isinstance(diff_val, (int, float)) else str(diff_val)
            else:
                s = f"{strat_val:.3f}" if isinstance(strat_val, float) else str(strat_val)
                b = f"{bench_val:.3f}" if isinstance(bench_val, float) else str(bench_val)
                d = f"{diff_val:+.3f}" if isinstance(diff_val, float) else str(diff_val)
            
            diff_color = Color.GREEN if (isinstance(diff_val, (int, float)) and diff_val > 0) else Color.RED
            return f"  {label:<12} {s:>16} {b:>16} {diff_color}{d:>16}{Color.RESET}"
        
        print(compare_row('累计收益', strat_ret, nasdaq_ret, diff_ret))
        print(compare_row('年化收益', benchmark.strategy_annual_return, 
                          benchmark.nasdaq_annual_return,
                          benchmark.strategy_annual_return - benchmark.nasdaq_annual_return))
        print(compare_row('夏普比率', benchmark.strategy_sharpe, benchmark.nasdaq_sharpe,
                          benchmark.strategy_sharpe - benchmark.nasdaq_sharpe, 'num'))
        print(compare_row('最大回撤', benchmark.strategy_max_drawdown, 
                          benchmark.nasdaq_max_drawdown,
                          benchmark.strategy_max_drawdown - benchmark.nasdaq_max_drawdown))
        
        # Alpha/Beta
        print(f"  {'Alpha(年化):':<12} {benchmark.alpha:>+16.4f}  │  "
              f"{'Beta:':<8} {benchmark.beta:>8.3f}  │  "
              f"{'信息比率:':<10} {benchmark.information_ratio:>8.3f}")
        
        # 超额收益评价
        if benchmark.outperformance_pct > 0.02:
            comment = f"{Color.GREEN}✅ 策略显著跑赢纳指{Color.RESET}"
        elif benchmark.outperformance_pct > 0:
            comment = f"{Color.GREEN}👍 策略略胜纳指{Color.RESET}"
        elif benchmark.outperformance_pct > -0.02:
            comment = f"{Color.YELLOW}➡️ 策略与纳指持平{Color.RESET}"
        else:
            comment = f"{Color.RED}📉 策略跑输纳指{Color.RESET}"
        
        print(f"  {'评价:':<12} {comment}")
    
    def _render_recent_trades(self) -> None:
        """渲染最近交易"""
        recent = self.simulator.portfolio.get_trade_summary(last_n=5)
        
        print(f"{Color.BOLD}📋 最近交易 (最新5笔){Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        if recent.empty:
            print(f"  {Color.GRAY}暂无交易记录{Color.RESET}")
            return
        
        for _, row in recent.iterrows():
            side_icon = '🟢买入' if row['Side'] == 'BUY' else '🔴卖出'
            pnl_str = f"盈亏{Color.pnl_str(row['PnL'])}" if row['PnL'] else ''
            line = (f"  {row['ID']} {side_icon} {row['Ticker']:<6} "
                    f"{row['Qty']:>4}股 @ ${row['Price']:>.2f} "
                    f"${row['Value']:>10,.0f} {pnl_str}")
            print(line)
    
    def _render_footer(self) -> None:
        """渲染页脚"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        count = self._render_count
        
        print(f"{Color.CYAN}{'═' * self.WIDTH}{Color.RESET}")
        print(f"  {Color.DIM}刷新 #{count}  │  当前时间: {now}  │  "
              f"按 Ctrl+C 退出{Color.RESET}")
    
    def run_auto_refresh(self) -> None:
        """
        启动自动刷新模式
        
        启动模拟器后台线程，面板自动每分钟刷新。
        主线程阻塞，按Ctrl+C退出。
        """
        # 启动模拟器（后台线程）
        self.simulator.start()
        
        try:
            # 主线程保持运行，等待键盘中断
            while self.simulator.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n{Color.YELLOW}正在停止...{Color.RESET}")
        finally:
            self.simulator.stop()
            print(f"{Color.GREEN}已退出在线模拟交易系统{Color.RESET}")
    
    def run_single_render(self) -> None:
        """
        单次渲染模式（不启动后台线程）
        
        用于手动刷新场景。
        """
        # 手动执行一次tick，然后渲染
        prices = self.simulator._fetch_realtime_prices()
        if prices:
            self.simulator._current_prices = prices
            self.simulator.portfolio.update_prices(prices)
        
        self.render_once()
    
    def run_offline_demo(self) -> None:
        """
        离线演示模式
        
        使用模拟数据显示完整功能，无需网络连接。
        适合演示和开发调试。
        """
        
        print(f"{Color.RED}{'='*60}{Color.RESET}")
        print(f"{Color.RED}  ⚠️  离线演示模式：所有数据为随机合成数据{Color.RESET}")
        print(f"{Color.RED}  ⚠️  不可用于真实交易决策！仅供演示和调试{Color.RESET}")
        print(f"{Color.RED}{'='*60}{Color.RESET}\n")
        
        # 设置模拟持仓
        mock_positions = {
            'AAPL': {'qty': 50, 'cost': 175.0},
            'MSFT': {'qty': 30, 'cost': 380.0},
            'NVDA': {'qty': 40, 'cost': 850.0},
            'GOOGL': {'qty': 20, 'cost': 140.0},
            'TSLA':  {'qty': 25, 'cost': 220.0},
        }
        
        for ticker, info in mock_positions.items():
            random.seed(42); current_price = info['cost'] * (1 + random.uniform(-0.15, 0.25))
            self.simulator.portfolio.execute_buy(
                ticker, info['qty'], info['cost'],
                commission=1.0, reason='模拟建仓'
            )
            self.simulator._current_prices[ticker] = current_price
        
        # 更新价格
        self.simulator.portfolio.update_prices(self.simulator._current_prices)
        
        # 模拟一些已确认的预测
        for i in range(30):
            ticker = random.choice(['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA'])
            actual_dir = 1 if random.random() < 0.55 else 0
            pred_id = self.simulator.accuracy_tracker.record_prediction(
                ticker, 
                random.uniform(-0.02, 0.03),
                1 if random.random() < 0.58 else 0,
                random.uniform(0.5, 0.9)
            )
            self.simulator.accuracy_tracker.confirm_prediction(
                pred_id,
                random.uniform(-0.02, 0.03),
                actual_dir
            )
        
        # 设置纳指基准价格
        if '^IXIC' not in self.simulator._current_prices:
            self.simulator._current_prices['^IXIC'] = 18500.0
        
        # 渲染
        self.render_once()


# ============================================================================
# 便捷入口
# ============================================================================
def launch_dashboard(
    tickers: Optional[list] = None,
    initial_capital: float = 100_000.0,
    mode: str = 'live'  # 'live' 或 'demo'
) -> None:
    """
    一键启动在线模拟交易面板
    
    参数:
        tickers: 追踪的股票代码（None使用默认8只）
        initial_capital: 初始资金
        mode: 'live'(实盘数据) 或 'demo'(演示数据)
    
    使用示例:
        from live_trading import launch_dashboard
        launch_dashboard(tickers=['AAPL','MSFT','NVDA'], mode='demo')
    """
    from live_trading.live_simulator import LiveSimulator
    
    sim = LiveSimulator(
        tickers=tickers,
        initial_capital=initial_capital
    )
    
    dash = TradingDashboard(sim)
    
    if mode == 'demo':
        print(f"{Color.CYAN}启动离线演示模式...{Color.RESET}")
        dash.run_offline_demo()
        print(f"\n{Color.GRAY}提示: 使用 mode='live' 启动实时模式{Color.RESET}")
    else:
        print(f"{Color.CYAN}启动在线模拟交易系统...{Color.RESET}")
        dash.run_auto_refresh()


# 支持直接运行
if __name__ == '__main__':
    launch_dashboard(mode='demo')
