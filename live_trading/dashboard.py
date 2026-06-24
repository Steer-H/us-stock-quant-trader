"""
在線模擬交易系統 - 實時交易儀錶盤

提供清晰、專業的終端實時展示面板，包含：

┌─────────────────────────────────────────────────────────────┐
│          美股量化交易系統 - 在線模擬交易                       │
│  市場狀態 │ 倒計時 │ 當前時間                                 │
├─────────────────────────────────────────────────────────────┤
│  💰 帳戶概覽                                                 │
│  初始資金 │ 總資產 │ 現金 │ 持倉市值                          │
├─────────────────────────────────────────────────────────────┤
│  📊 盈虧統計                                                 │
│  總盈虧 │ 盈虧% │ 當日盈虧 │ 已實現/未實現                     │
├─────────────────────────────────────────────────────────────┤
│  📈 持倉明細（股票/數量/成本價/現價/盈虧金額/盈虧%/權重）      │
├─────────────────────────────────────────────────────────────┤
│  🎯 模型準確率                                                │
│  方向準確率 │ RMSE │ 最近50次 │ 趨勢                          │
├─────────────────────────────────────────────────────────────┤
│  🏦 納指基準對比                                              │
│  策略收益 vs 納指收益 │ Alpha │ Beta │ Sharpe                  │
├─────────────────────────────────────────────────────────────┤
│  📋 最近交易                                                  │
└─────────────────────────────────────────────────────────────┘

刷新頻率：每分鐘自動刷新
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
# 顏色常量（終端ANSI）
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
        """根據盈虧返回顏色"""
        if value > 0:
            return Color.GREEN
        elif value < 0:
            return Color.RED
        return Color.WHITE
    
    @staticmethod
    def pnl_str(value: float, as_pct: bool = False) -> str:
        """帶顏色的盈虧字符串"""
        sign = '+' if value > 0 else ''
        if as_pct:
            s = f"{sign}{value:.2%}"
        else:
            s = f"{sign}{value:,.2f}"
        color = Color.pnl_color(value)
        return f"{color}{s}{Color.RESET}"


# ============================================================================
# 儀錶盤渲染器
# ============================================================================
class TradingDashboard:
    """
    實時交易儀錶盤
    
    以固定的終端區域刷新顯示所有關鍵信息。
    支持自動刷新和手動刷新兩種模式。
    
    使用:
        dash = TradingDashboard(simulator)
        dash.run_auto_refresh()  # 自動刷新模式
        # 或
        dash.render_once()       # 單次渲染
    """
    
    WIDTH = 85  # 顯示寬度
    
    def __init__(self, simulator: LiveSimulator):
        """
        參數:
            simulator: LiveSimulator實例
        """
        self.simulator = simulator
        self.clock = MarketClock()
        self._render_count: int = 0
        
        # 註冊為模擬器的刷新回調
        self.simulator.add_refresh_callback(lambda sim: self.render_once())
    
    def clear_screen(self) -> None:
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def render_once(self) -> None:
        """渲染一次面板"""
        self._render_count += 1
        
        # 獲取完整快照
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
        """渲染頂部標題和狀態欄"""
        status = market.get('status', MarketStatus.CLOSED)
        desc = market.get('description', '閉市')
        current_et = market.get('current_et', '')
        is_open = market.get('is_open', False)
        
        # 狀態圖標
        status_icons = {
            MarketStatus.REGULAR_HOURS: '🟢',
            MarketStatus.PRE_MARKET: '🟡',
            MarketStatus.AFTER_HOURS: '🔵',
            MarketStatus.CLOSED: '⚫',
            MarketStatus.WEEKEND: '🔴',
            MarketStatus.HOLIDAY: '🔴',
        }
        icon = status_icons.get(status, '⚫')
        
        # 頂欄
        print(f"{Color.BOLD}{Color.CYAN}{'═' * self.WIDTH}{Color.RESET}")
        print(f"{Color.BOLD}  美股量化交易系統 - 在線模擬交易  {Color.DIM}v1.0{Color.RESET}")
        print(f"{Color.CYAN}{'═' * self.WIDTH}{Color.RESET}")
        
        # 狀態行
        status_color = Color.GREEN if is_open else Color.YELLOW
        
        line = f"  {icon} 市場狀態: {status_color}{desc}{Color.RESET}"
        
        # 倒計時
        if not market.get('is_active', False):
            cd = market.get('countdown_to_open')
            if cd:
                h, m, s = cd
                line += f"  │  ⏳ 距離開市還有: {Color.BOLD}{h}h {m:02d}m {s:02d}s{Color.RESET}"
        elif status == MarketStatus.REGULAR_HOURS:
            cd = market.get('countdown_to_close')
            if cd:
                h, m, s = cd
                line += f"  │  ⏳ 距離閉市還有: {Color.YELLOW}{h}h {m:02d}m {s:02d}s{Color.RESET}"
        
        line += f"  │  🕐 {current_et}"
        print(line)
        
        # 活躍交易時段提示
        if is_open:
            print(f"  {Color.GREEN}▶ 交易進行中... (每分鐘自動刷新){Color.RESET}")
    
    def _render_account_overview(self, portfolio: PortfolioSnapshot) -> None:
        """渲染帳戶概覽"""
        print(f"{Color.BOLD}💰 帳戶概覽{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        cols = [
            ('初始資金', f"${portfolio.initial_capital:,.2f}"),
            ('總淨資產', f"{Color.BOLD}${portfolio.total_equity:,.2f}{Color.RESET}"),
            ('現金餘額', f"${portfolio.cash:,.2f}"),
            ('持倉市值', f"${portfolio.total_market_value:,.2f}"),
        ]
        
        line = '  '
        for label, value in cols:
            line += f"{label}: {value}  "
        
        print(line)
        print(f"  {'持倉數量':<10}: {portfolio.position_count} 只  │  "
              f"{'槓桿':<6}: {portfolio.leverage:.2f}x")
    
    def _render_pnl_summary(self, portfolio: PortfolioSnapshot, 
                             benchmark: BenchmarkSnapshot) -> None:
        """渲染盈虧統計"""
        print(f"{Color.BOLD}📊 盈虧統計{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        total_pnl = portfolio.total_pnl
        total_pnl_pct = portfolio.total_pnl_pct
        day_pnl = portfolio.day_pnl
        day_pnl_pct = portfolio.day_pnl_pct
        
        # 總盈虧
        line = f"  {'總盈虧:':<10} {Color.pnl_str(total_pnl)} ({Color.pnl_str(total_pnl_pct, True)})"
        line += f"  │  {'當日盈虧:':<10} {Color.pnl_str(day_pnl)} ({Color.pnl_str(day_pnl_pct, True)})"
        print(line)
        
        # 已實現/未實現
        line = f"  {'已實現:':<10} {Color.pnl_str(portfolio.realized_pnl)}"
        line += f"  │  {'未實現:':<10} {Color.pnl_str(portfolio.unrealized_pnl)}"
        
        # 最大回撤
        dd = portfolio.max_drawdown_pct
        dd_color = Color.YELLOW if dd < -0.1 else (Color.RED if dd < -0.2 else Color.WHITE)
        line += f"  │  {'最大回撤:':<10} {dd_color}{dd:+.2%}{Color.RESET}"
        print(line)
    
    def _render_positions(self, portfolio: PortfolioSnapshot) -> None:
        """渲染持倉明細表格"""
        positions = portfolio.positions
        
        print(f"{Color.BOLD}📈 持倉明細{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        if not positions:
            print(f"  {Color.GRAY}暫無持倉{Color.RESET}")
            return
        
        # 表頭
        header = (f"  {'股票':<7} {'數量':>8} {'成本價':>10} {'現價':>10} "
                  f"{'市值':>12} {'盈虧金額':>10} {'盈虧%':>8} {'權重':>6}")
        print(f"{Color.DIM}{header}{Color.RESET}")
        print(f"  {Color.DIM}{'─' * 72}{Color.RESET}")
        
        # 按權重排序
        sorted_positions = sorted(
            positions.values(), 
            key=lambda p: abs(p.market_value), 
            reverse=True
        )
        
        for pos in sorted_positions[:20]:  # 最多顯示20隻
            pnl_str = Color.pnl_str(pos.unrealized_pnl)
            pnl_pct_str = Color.pnl_str(pos.unrealized_pnl_pct, True)
            
            line = (f"  {pos.ticker:<7} {pos.quantity:>8} "
                    f"${pos.avg_cost:>9.2f} ${pos.current_price:>9.2f} "
                    f"${pos.market_value:>11,.0f} "
                    f"{pnl_str} {pnl_pct_str} "
                    f"{pos.weight:>5.1%}")
            print(line)
        
        if len(sorted_positions) > 20:
            print(f"  {Color.GRAY}... 還有 {len(sorted_positions) - 20} 只持倉{Color.RESET}")
    
    def _render_model_accuracy(self, accuracy: AccuracySnapshot) -> None:
        """渲染模型準確率"""
        print(f"{Color.BOLD}🎯 模型預測準確率{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        dir_acc = accuracy.direction_accuracy
        recent_acc = accuracy.recent_accuracy_50
        
        # 準確率狀態
        acc_color = Color.GREEN if dir_acc >= 0.55 else (Color.YELLOW if dir_acc >= 0.50 else Color.RED)
        recent_color = Color.GREEN if recent_acc >= 0.55 else Color.YELLOW
        
        line = f"  {'方向準確率:':<12} {acc_color}{dir_acc:.1%}{Color.RESET}"
        line += f"  │  {'最近50次:':<10} {recent_color}{recent_acc:.1%}{Color.RESET}"
        line += f"  │  {'RMSE:':<8} {accuracy.rmse:.6f}"
        line += f"  │  {'MAE:':<8} {accuracy.mae:.6f}"
        print(line)
        
        # 細分統計
        total = accuracy.total_long + accuracy.total_short
        long_acc = accuracy.correct_long / max(accuracy.total_long, 1)
        short_acc = accuracy.correct_short / max(accuracy.total_short, 1)
        
        line = f"  {'總預測:':<10} {accuracy.total_predictions}"
        line += f"  │  {'已確認:':<10} {accuracy.confirmed_predictions}"
        line += f"  │  {'漲正確:':<10} {accuracy.correct_long}/{accuracy.total_long} ({long_acc:.0%})"
        line += f"  │  {'跌正確:':<10} {accuracy.correct_short}/{accuracy.total_short} ({short_acc:.0%})"
        print(line)
        
        # 趨勢
        trend = accuracy.accuracy_trend
        trend_icon = {'stable': '➡️ 穩定', 'degrading': '⚠️ 下降', 'improving': '📈 上升'}.get(trend, '➡️ 穩定')
        status_icon = '✅ 可接受' if accuracy.is_acceptable else '⚠️ 需調優'
        print(f"  {'模型狀態:':<10} {status_icon}  │  {'趨勢:':<8} {trend_icon}")
    
    def _render_benchmark_comparison(self, benchmark: BenchmarkSnapshot, 
                                      portfolio: PortfolioSnapshot) -> None:
        """渲染納指基準對比"""
        print(f"{Color.BOLD}🏦 納指(^IXIC)基準對比{Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        # 對比表格
        header = (f"  {'':<12} {'策略':>16} {'納指':>16} {'差異':>16}")
        print(f"{Color.DIM}{header}{Color.RESET}")
        print(f"  {Color.DIM}{'─' * 62}{Color.RESET}")
        
        # 累計收益
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
        
        print(compare_row('累計收益', strat_ret, nasdaq_ret, diff_ret))
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
        
        # 超額收益評價
        if benchmark.outperformance_pct > 0.02:
            comment = f"{Color.GREEN}✅ 策略顯著跑贏納指{Color.RESET}"
        elif benchmark.outperformance_pct > 0:
            comment = f"{Color.GREEN}👍 策略略勝納指{Color.RESET}"
        elif benchmark.outperformance_pct > -0.02:
            comment = f"{Color.YELLOW}➡️ 策略與納指持平{Color.RESET}"
        else:
            comment = f"{Color.RED}📉 策略跑輸納指{Color.RESET}"
        
        print(f"  {'評價:':<12} {comment}")
    
    def _render_recent_trades(self) -> None:
        """渲染最近交易"""
        recent = self.simulator.portfolio.get_trade_summary(last_n=5)
        
        print(f"{Color.BOLD}📋 最近交易 (最新5筆){Color.RESET}")
        print(f"{Color.DIM}{'─' * self.WIDTH}{Color.RESET}")
        
        if recent.empty:
            print(f"  {Color.GRAY}暫無交易記錄{Color.RESET}")
            return
        
        for _, row in recent.iterrows():
            side_icon = '🟢買入' if row['Side'] == 'BUY' else '🔴賣出'
            pnl_str = f"盈虧{Color.pnl_str(row['PnL'])}" if row['PnL'] else ''
            line = (f"  {row['ID']} {side_icon} {row['Ticker']:<6} "
                    f"{row['Qty']:>4}股 @ ${row['Price']:>.2f} "
                    f"${row['Value']:>10,.0f} {pnl_str}")
            print(line)
    
    def _render_footer(self) -> None:
        """渲染頁腳"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        count = self._render_count
        
        print(f"{Color.CYAN}{'═' * self.WIDTH}{Color.RESET}")
        print(f"  {Color.DIM}刷新 #{count}  │  當前時間: {now}  │  "
              f"按 Ctrl+C 退出{Color.RESET}")
    
    def run_auto_refresh(self) -> None:
        """
        啟動自動刷新模式
        
        啟動模擬器後臺線程，面板自動每分鐘刷新。
        主線程阻塞，按Ctrl+C退出。
        """
        # 啟動模擬器（後臺線程）
        self.simulator.start()
        
        try:
            # 主線程保持運行，等待鍵盤中斷
            while self.simulator.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n{Color.YELLOW}正在停止...{Color.RESET}")
        finally:
            self.simulator.stop()
            print(f"{Color.GREEN}已退出在線模擬交易系統{Color.RESET}")
    
    def run_single_render(self) -> None:
        """
        單次渲染模式（不啟動後臺線程）
        
        用於手動刷新場景。
        """
        # 手動執行一次tick，然後渲染
        prices = self.simulator._fetch_realtime_prices()
        if prices:
            self.simulator._current_prices = prices
            self.simulator.portfolio.update_prices(prices)
        
        self.render_once()
    
    def run_offline_demo(self) -> None:
        """
        離線演示模式
        
        使用模擬數據顯示完整功能，無需網絡連接。
        適合演示和開發調試。
        """
        
        print(f"{Color.RED}{'='*60}{Color.RESET}")
        print(f"{Color.RED}  ⚠️  離線演示模式：所有數據為隨機合成數據{Color.RESET}")
        print(f"{Color.RED}  ⚠️  不可用於真實交易決策！僅供演示和調試{Color.RESET}")
        print(f"{Color.RED}{'='*60}{Color.RESET}\n")
        
        # 設置模擬持倉
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
                commission=1.0, reason='模擬建倉'
            )
            self.simulator._current_prices[ticker] = current_price
        
        # 更新價格
        self.simulator.portfolio.update_prices(self.simulator._current_prices)
        
        # 模擬一些已確認的預測
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
        
        # 設置納指基準價格
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
    一鍵啟動在線模擬交易面板
    
    參數:
        tickers: 追蹤的股票代碼（None使用默認8隻）
        initial_capital: 初始資金
        mode: 'live'(實盤數據) 或 'demo'(演示數據)
    
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
        print(f"{Color.CYAN}啟動離線演示模式...{Color.RESET}")
        dash.run_offline_demo()
        print(f"\n{Color.GRAY}提示: 使用 mode='live' 啟動實時模式{Color.RESET}")
    else:
        print(f"{Color.CYAN}啟動在線模擬交易系統...{Color.RESET}")
        dash.run_auto_refresh()


# 支持直接運行
if __name__ == '__main__':
    launch_dashboard(mode='demo')
