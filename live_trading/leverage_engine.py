"""
动态杠杆引擎 — Dynamic Leverage Engine

基于多因子模型动态计算最优杠杆倍数，替代静态的 LEVERAGE_TIERS 映射。

核心公式:
    leverage = kelly_base × vol_multiplier × perf_multiplier × heat_multiplier
    leverage = clamp(leverage, 0.25, MAX_LEVERAGE)
    leverage = min(leverage, drawdown_cap)

四因子说明:
  1. 凯利公式 (Kelly Criterion): f* = (p × b - q) / b, 取 half-Kelly 保守
  2. 波动率调节 (Volatility): 市场波动越大 → 杠杆越低
  3. 绩效反馈 (Performance): 近期胜率反馈, 连胜加码/连败减仓
  4. 组合热度 (Portfolio Heat): 持仓越多 → 新开仓杠杆越低

风险硬约束 (Circuit Breakers):
  - 回撤 > 5%  → 杠杆上限 1.5x
  - 回撤 > 10% → 杠杆上限 1.0x
  - 回撤 > 15% → 杠杆上限 0.5x (强制去杠杆)
  - 保证金 > 80% → 杠杆上限 1.0x
  - 保证金 > 90% → 杠杆上限 0.0x (停止开仓, margin call)

使用方式:
    engine = LeverageEngine()
    lev = engine.calculate(
        confidence=0.72,        # ML预测置信度
        ticker='AAPL',          # 目标股票
        current_prices={...},   # 所有股票当前价格
        prev_prices={...},      # 前次价格(算波动率)
        portfolio=portfolio,    # PortfolioManager实例
        accuracy=accuracy,      # AccuracyTracker实例
    )
"""

import math
import logging
from typing import Dict, Optional, Tuple
from collections import deque
import numpy as np
import time
from config.settings import trading_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 配置常量
# ═══════════════════════════════════════════════════════════════
MAX_LEVERAGE = 2.0                # 全局最大杠杆上限
MIN_LEVERAGE = 0.25               # 全局最小杠杆(低于此不开仓)
HALF_KELLY_FRACTION = 0.5         # Half-Kelly 保守系数 (0.5=half, 1.0=full)
DEFAULT_WIN_LOSS_RATIO = 1.8      # 默认盈亏比 (无历史数据时)
MIN_WIN_LOSS_RATIO = 1.2          # 盈亏比下限
MAX_WIN_LOSS_RATIO = 4.0          # 盈亏比上限

# 波动率分位阈值 (基于日收益率标准差 annualized)
VOL_LOW_THRESHOLD = 0.15          # 年化15%以下 → 低波动
VOL_HIGH_THRESHOLD = 0.35         # 年化35%以上 → 高波动
VOL_EXTREME_THRESHOLD = 0.55      # 年化55%以上 → 极端波动

# 绩效反馈窗口
PERF_WINDOW = 20                  # 最近N笔交易
PERF_BOOST_WINRATE = 0.60         # 胜率>60% → 加码
PERF_REDUCE_WINRATE = 0.45        # 胜率<45% → 减仓

# 组合热度分位
HEAT_LOW = 0.30                   # 持仓<30%容量 → 全额杠杆
HEAT_MEDIUM = 0.60                # 30-60% → 打折
# >60% → 大幅打折

# 回撤硬约束 (drawdown → max leverage cap)
DRAWDOWN_CAPS = [
    (0.05, 1.5),                   # 回撤5% → 上限1.5x
    (0.10, 1.0),                   # 回撤10% → 上限1.0x
    (0.15, 0.5),                   # 回撤15% → 上限0.5x (强平)
    (0.20, 0.0),                   # 回撤20% → 停止开仓
]


class LeverageEngine:
    """
    动态杠杆引擎
    
    追踪交易绩效和波动率状态，实时计算最优杠杆倍数。
    """

    def __init__(self):
        # 绩效追踪
        self.recent_trades: deque = deque(maxlen=PERF_WINDOW)
        # 每个交易记录: {'win': bool, 'pnl_pct': float}
        
        # 波动率缓存
        self._volatility_cache: Dict[str, float] = {}    # 全局波动率估计
        self._volatility_updated_at: float = 0.0
        self._volatility_cache_ttl: float = 300.0         # 5分钟缓存
        
        # 统计数据(跨会话持久化)
        self.total_calculations: int = 0
        self.avg_leverage: float = 1.0
        
        logger.info("LeverageEngine 初始化: Kelly+Vol+Perf+Heat 四因子模型, max=%.1fx", MAX_LEVERAGE)

    # ── 因子1: 凯利公式 ─────────────────────────────────────
    def _kelly_fraction(self, confidence: float, win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO) -> float:
        """
        凯利公式: f* = (p*b - q) / b
        
        参数:
            confidence: 预测胜率 p (0.5-1.0)
            win_loss_ratio: 盈亏比 b (avg_win / avg_loss)
        
        返回:
            最优仓位比例 (0-1)
        
        推导:
            p=0.55, b=1.5 → f* = (0.55*1.5 - 0.45)/1.5 = 0.25 → 1.25x
            p=0.70, b=2.0 → f* = (0.70*2.0 - 0.30)/2.0 = 0.55 → 1.55x
            p=0.80, b=2.5 → f* = (0.80*2.5 - 0.20)/2.5 = 0.72 → 1.72x
        """
        # clamp置信度到合理范围
        p = max(0.50, min(confidence, 0.95))
        q = 1.0 - p
        b = max(MIN_WIN_LOSS_RATIO, min(win_loss_ratio, MAX_WIN_LOSS_RATIO))
        
        # Kelly fraction
        f_star = (p * b - q) / b
        
        # Half-Kelly: 保守起见只用一半
        f_half = f_star * HALF_KELLY_FRACTION
        
        # Kelly为负 → 不应下注（但这里最小返回min leverage）
        if f_star <= 0:
            return 0.0
        
        # 转换为杠杆倍数: f_half=0.25 → 1.25x, f_half=0.50 → 1.50x
        leverage = 1.0 + f_half * (MAX_LEVERAGE - 1.0) / 0.5
        # f_half=0 → 1.0x, f_half=0.5 → MAX_LEVERAGE x
        
        return max(0.0, min(leverage, MAX_LEVERAGE))

    # ── 因子2: 波动率调节 ───────────────────────────────────
    def _estimate_volatility(self, current_prices: Dict[str, float],
                              prev_prices: Dict[str, float]) -> float:
        """
        从持仓股票价格变化估计市场波动率
        
        使用价格变动百分比的标准差 × sqrt(252) 年化。
        
        返回:
            年化波动率估计 (0-1)
        """
        now = time.time()
        
        # 缓存命中
        if now - self._volatility_updated_at < self._volatility_cache_ttl:
            cached = self._volatility_cache.get('_global', None)
            if cached is not None:
                return cached
        
        changes = []
        for ticker in current_prices:
            if ticker == '^IXIC':
                continue
            cur = current_prices.get(ticker, 0)
            prev = prev_prices.get(ticker, 0)
            if cur > 0 and prev > 0:
                pct = (cur / prev) - 1.0
                changes.append(pct)
        
        if len(changes) < 5:
            return 0.20  # 默认中等波动
        
        # 年化波动率 = std(returns) * sqrt(252)
        # 注意: 这是一个粗略估计，tick级变化不能直接年化
        # 我们取中位数股票的波动率作为代理
        std_pct = float(np.std(changes))
        
        # 粗略映射: 每tick std 0.01% → 年化约16%
        # 实际上应该用日收益率，这里做保守估计
        annualized = std_pct * 80  # 经验映射系数
        
        annualized = max(0.05, min(annualized, 0.80))
        
        self._volatility_cache['_global'] = annualized
        self._volatility_updated_at = now
        
        return annualized

    def _volatility_multiplier(self, annualized_vol: float) -> float:
        """
        波动率 → 杠杆乘数
        
        低波动 → 可以加杠杆
        高波动 → 必须降杠杆
        """
        if annualized_vol <= VOL_LOW_THRESHOLD:
            return 1.0          # 低波动: 全额杠杆
        elif annualized_vol <= VOL_HIGH_THRESHOLD:
            # 线性插值: 15%→1.0, 35%→0.6
            t = (annualized_vol - VOL_LOW_THRESHOLD) / (VOL_HIGH_THRESHOLD - VOL_LOW_THRESHOLD)
            return 1.0 - t * 0.4
        elif annualized_vol <= VOL_EXTREME_THRESHOLD:
            # 线性插值: 35%→0.6, 55%→0.25
            t = (annualized_vol - VOL_HIGH_THRESHOLD) / (VOL_EXTREME_THRESHOLD - VOL_HIGH_THRESHOLD)
            return 0.6 - t * 0.35
        else:
            return 0.25         # 极端波动: 最小杠杆

    # ── 因子3: 绩效反馈 ─────────────────────────────────────
    def record_trade(self, win: bool, pnl_pct: float):
        """记录交易结果"""
        self.recent_trades.append({'win': win, 'pnl_pct': pnl_pct})

    def _performance_multiplier(self) -> float:
        """
        近期胜率 → 杠杆乘数
        
        连胜 → 适度加码
        连败 → 强制减仓 (反马丁格尔)
        """
        if len(self.recent_trades) < 5:
            return 1.0  # 数据不足, 中性
        
        wins = sum(1 for t in self.recent_trades if t["win"])
        total = len(self.recent_trades)
        win_rate = wins / total
        
        # 基础乘数：基于胜率
        if win_rate >= PERF_BOOST_WINRATE:
            # 胜率高: 加码但设上限
            multiplier = min(1.0 + (win_rate - PERF_BOOST_WINRATE) * 2.0, 1.3)
        elif win_rate >= PERF_REDUCE_WINRATE:
            # 中等: 中性
            multiplier = 1.0
        else:
            # 胜率低: 减仓
            multiplier = max(1.0 - (PERF_REDUCE_WINRATE - win_rate) * 2.5, 0.4)
        
        # 额外: 连败检测 (最近3笔全输 → 强制上限0.5x)
        if len(self.recent_trades) >= 3:
            last_3 = list(self.recent_trades)[-3:]
            if all(not t["win"] for t in last_3):
                multiplier = min(multiplier, 0.5)
        
        return multiplier

    def _performance_multiplier_base(self) -> float:
        """基础胜率乘数(不含连败惩罚)"""
        if len(self.recent_trades) < 5:
            return 1.0
        wins = sum(1 for t in self.recent_trades if t['win'])
        win_rate = wins / len(self.recent_trades)
        if win_rate >= 0.60:
            return min(1.0 + (win_rate - 0.60) * 2.0, 1.3)
        elif win_rate >= 0.45:
            return 1.0
        else:
            return max(1.0 - (0.45 - win_rate) * 2.5, 0.4)

    # ── 因子4: 组合热度 ─────────────────────────────────────
    def _portfolio_heat_multiplier(self, portfolio) -> float:
        """
        当前持仓占比 → 杠杆乘数
        
        持仓越多 → 风险越集中 → 杠杆越低
        """
        max_positions = getattr(trading_config, 'max_positions', 40)
        current_positions = len(portfolio.positions)
        
        if max_positions <= 0:
            return 1.0
        
        fill_ratio = current_positions / max_positions
        
        if fill_ratio <= HEAT_LOW:
            return 1.0
        elif fill_ratio <= HEAT_MEDIUM:
            t = (fill_ratio - HEAT_LOW) / (HEAT_MEDIUM - HEAT_LOW)
            return 1.0 - t * 0.25   # 最多打75折
        else:
            t = min((fill_ratio - HEAT_MEDIUM) / (1.0 - HEAT_MEDIUM), 1.0)
            return 0.75 - t * 0.35  # 最多打4折

    # ── 风险硬约束 ──────────────────────────────────────────
    def _drawdown_cap(self, portfolio) -> float:
        """回撤 → 杠杆上限"""
        dd = abs(portfolio.get_max_drawdown_pct())
        cap = MAX_LEVERAGE
        for threshold, max_lev in DRAWDOWN_CAPS:
            if dd >= threshold:
                cap = min(cap, max_lev)
        return cap

    # ── 主计算方法 ──────────────────────────────────────────
    def calculate(
        self,
        confidence: float,
        ticker: str = '',
        current_prices: Optional[Dict[str, float]] = None,
        prev_prices: Optional[Dict[str, float]] = None,
        portfolio=None,
        accuracy=None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        计算动态杠杆倍数
        
        参数:
            confidence: ML预测置信度 (0.5-1.0)
            ticker: 目标股票代码
            current_prices: 所有股票当前价格
            prev_prices: 前次价格(用于波动率估算)
            portfolio: PortfolioManager 实例
            accuracy: AccuracyTracker 实例
        
        返回:
            (leverage, detail_dict)
        """
        self.total_calculations += 1
        
        # 0. 如果没有portfolio, 返回保守默认值
        if portfolio is None:
            return 1.0, {'kelly': 1.0, 'vol': 1.0, 'perf': 1.0, 'heat': 1.0, 
                        'dd_cap': MAX_LEVERAGE, 'margin': 1.0, 'final': 1.0}

        # 1. 凯利基础杠杆
        # 从accuracy tracker获取近期盈亏比
        win_loss_ratio = DEFAULT_WIN_LOSS_RATIO
        if accuracy is not None:
            # 用近期方向准确率推断盈亏比
            acc_snap = accuracy.get_snapshot()
            recent_acc = acc_snap.recent_accuracy_50
            if recent_acc > 0.5:
                # 准确率越高, 推断盈亏比越好
                win_loss_ratio = 1.2 + (recent_acc - 0.5) * 4.0
                win_loss_ratio = min(win_loss_ratio, MAX_WIN_LOSS_RATIO)
        
        kelly_lev = self._kelly_fraction(confidence, win_loss_ratio)

        # 2. 波动率乘数
        vol_mult = 1.0
        if current_prices and prev_prices:
            annual_vol = self._estimate_volatility(current_prices, prev_prices)
            vol_mult = self._volatility_multiplier(annual_vol)
        else:
            annual_vol = 0.0

        # 3. 绩效反馈乘数
        perf_mult = self._performance_multiplier()

        # 4. 组合热度乘数
        heat_mult = self._portfolio_heat_multiplier(portfolio)

        # 综合计算
        leverage = kelly_lev * vol_mult * perf_mult * heat_mult

        # 回撤硬约束
        dd_cap = self._drawdown_cap(portfolio)
        leverage = min(leverage, dd_cap)

        # 保证金硬约束
        margin_ratio = portfolio.get_margin_ratio()
        margin_cap = MAX_LEVERAGE
        if margin_ratio > 0.90:
            margin_cap = 0.0  # 停止开仓
        elif margin_ratio > 0.80:
            margin_cap = 1.0
        elif margin_ratio > 0.60:
            margin_cap = 1.5
        leverage = min(leverage, margin_cap)

        # 全局 clamp (但尊重 Kelly=0 的"不下注"信号)
        if kelly_lev <= 0:
            leverage = 0.0  # Kelly 判定无优势，不下注
        else:
            leverage = max(MIN_LEVERAGE, min(leverage, MAX_LEVERAGE))

        # 四舍五入到0.05
        leverage = round(leverage * 20) / 20

        # 更新统计
        self.avg_leverage = (self.avg_leverage * (self.total_calculations - 1) + leverage) / self.total_calculations

        detail = {
            'kelly': round(kelly_lev, 3),
            'vol': round(vol_mult, 3),
            'perf': round(perf_mult, 3),
            'heat': round(heat_mult, 3),
            'dd_cap': round(dd_cap, 2),
            'margin': round(margin_cap, 2),
            'vol_est': round(annual_vol, 3),
            'final': round(leverage, 2),
        }

        return leverage, detail

    def to_dict(self) -> Dict:
        """序列化为字典(用于状态持久化)"""
        return {
            'recent_trades': list(self.recent_trades),
            'total_calculations': self.total_calculations,
            'avg_leverage': self.avg_leverage,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LeverageEngine':
        """从字典恢复"""
        engine = cls()
        engine.recent_trades = deque(
            data.get('recent_trades', []), maxlen=PERF_WINDOW
        )
        engine.total_calculations = data.get('total_calculations', 0)
        engine.avg_leverage = data.get('avg_leverage', 1.0)
        return engine
