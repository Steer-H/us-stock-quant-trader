"""Dynamic leverage based on Kelly criterion and volatility."""

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
MAX_LEVERAGE = 2.0                # 全局最大槓桿上限
MIN_LEVERAGE = 0.25               # 全局最小槓桿(低於此不開倉)
HALF_KELLY_FRACTION = 0.5         # Half-Kelly 保守係數 (0.5=half, 1.0=full)
DEFAULT_WIN_LOSS_RATIO = 1.8      # 默認盈虧比 (無歷史數據時)
MIN_WIN_LOSS_RATIO = 1.2          # 盈虧比下限
MAX_WIN_LOSS_RATIO = 4.0          # 盈虧比上限

# 波動率分位閾值 (基於日收益率標準差 annualized)
VOL_LOW_THRESHOLD = 0.15          # 年化15%以下 → 低波動
VOL_HIGH_THRESHOLD = 0.35         # 年化35%以上 → 高波動
VOL_EXTREME_THRESHOLD = 0.55      # 年化55%以上 → 極端波動

# 績效反饋窗口
PERF_WINDOW = 20                  # 最近N筆交易
PERF_BOOST_WINRATE = 0.60         # 勝率>60% → 加碼
PERF_REDUCE_WINRATE = 0.45        # 勝率<45% → 減倉

# 組合熱度分位
HEAT_LOW = 0.30                   # 持倉<30%容量 → 全額槓桿
HEAT_MEDIUM = 0.60                # 30-60% → 打折
# >60% → 大幅打折

# 回撤硬約束 (drawdown → max leverage cap)
DRAWDOWN_CAPS = [
    (0.05, 1.5),                   # 回撤5% → 上限1.5x
    (0.10, 1.0),                   # 回撤10% → 上限1.0x
    (0.15, 0.5),                   # 回撤15% → 上限0.5x (強平)
    (0.20, 0.0),                   # 回撤20% → 停止開倉
]


class LeverageEngine:
    """
    動態槓桿引擎
    
    追蹤交易績效和波動率狀態，實時計算最優槓桿倍數。
    """

    def __init__(self):
        # 績效追蹤
        self.recent_trades: deque = deque(maxlen=PERF_WINDOW)
        # 每個交易記錄: {'win': bool, 'pnl_pct': float}
        
        # 波動率緩存
        self._volatility_cache: Dict[str, float] = {}    # 全局波動率估計
        self._volatility_updated_at: float = 0.0
        self._volatility_cache_ttl: float = 300.0         # 5分鐘緩存
        
        # 統計數據(跨會話持久化)
        self.total_calculations: int = 0
        self.avg_leverage: float = 1.0
        
        logger.info("LeverageEngine 初始化: Kelly+Vol+Perf+Heat 四因子模型, max=%.1fx", MAX_LEVERAGE)

    # ── 因子1: 凱利公式 ─────────────────────────────────────
    def _kelly_fraction(self, confidence: float, win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO) -> float:
        """
        凱利公式: f* = (p*b - q) / b
        
        參數:
            confidence: 預測勝率 p (0.5-1.0)
            win_loss_ratio: 盈虧比 b (avg_win / avg_loss)
        
        返回:
            最優倉位比例 (0-1)
        
        推導:
            p=0.55, b=1.5 → f* = (0.55*1.5 - 0.45)/1.5 = 0.25 → 1.25x
            p=0.70, b=2.0 → f* = (0.70*2.0 - 0.30)/2.0 = 0.55 → 1.55x
            p=0.80, b=2.5 → f* = (0.80*2.5 - 0.20)/2.5 = 0.72 → 1.72x
        """
        # clamp置信度到合理範圍
        p = max(0.50, min(confidence, 0.95))
        q = 1.0 - p
        b = max(MIN_WIN_LOSS_RATIO, min(win_loss_ratio, MAX_WIN_LOSS_RATIO))
        
        # Kelly fraction
        f_star = (p * b - q) / b
        
        # Half-Kelly: 保守起見只用一半
        f_half = f_star * HALF_KELLY_FRACTION
        
        # Kelly為負 → 不應下注（但這裡最小返回min leverage）
        if f_star <= 0:
            return 0.0
        
        # 轉換為槓桿倍數: f_half=0.25 → 1.25x, f_half=0.50 → 1.50x
        leverage = 1.0 + f_half * (MAX_LEVERAGE - 1.0) / 0.5
        # f_half=0 → 1.0x, f_half=0.5 → MAX_LEVERAGE x
        
        return max(0.0, min(leverage, MAX_LEVERAGE))

    # ── 因子2: 波動率調節 ───────────────────────────────────
    def _estimate_volatility(self, current_prices: Dict[str, float],
                              prev_prices: Dict[str, float]) -> float:
        """
        從持倉股票價格變化估計市場波動率
        
        使用價格變動百分比的標準差 × sqrt(252) 年化。
        
        返回:
            年化波動率估計 (0-1)
        """
        now = time.time()
        
        # 緩存命中
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
            return 0.20  # 默認中等波動
        
        # 年化波動率 = std(returns) * sqrt(252)
        # 注意: 這是一個粗略估計，tick級變化不能直接年化
        # 我們取中位數股票的波動率作為代理
        std_pct = float(np.std(changes))
        
        # 粗略映射: 每tick std 0.01% → 年化約16%
        # 實際上應該用日收益率，這裡做保守估計
        annualized = std_pct * 80  # 經驗映射係數
        
        annualized = max(0.05, min(annualized, 0.80))
        
        self._volatility_cache['_global'] = annualized
        self._volatility_updated_at = now
        
        return annualized

    def _volatility_multiplier(self, annualized_vol: float) -> float:
        """
        波動率 → 槓桿乘數
        
        低波動 → 可以加槓桿
        高波動 → 必須降槓桿
        """
        if annualized_vol <= VOL_LOW_THRESHOLD:
            return 1.0          # 低波動: 全額槓桿
        elif annualized_vol <= VOL_HIGH_THRESHOLD:
            # 線性插值: 15%→1.0, 35%→0.6
            t = (annualized_vol - VOL_LOW_THRESHOLD) / (VOL_HIGH_THRESHOLD - VOL_LOW_THRESHOLD)
            return 1.0 - t * 0.4
        elif annualized_vol <= VOL_EXTREME_THRESHOLD:
            # 線性插值: 35%→0.6, 55%→0.25
            t = (annualized_vol - VOL_HIGH_THRESHOLD) / (VOL_EXTREME_THRESHOLD - VOL_HIGH_THRESHOLD)
            return 0.6 - t * 0.35
        else:
            return 0.25         # 極端波動: 最小槓桿

    # ── 因子3: 績效反饋 ─────────────────────────────────────
    def record_trade(self, win: bool, pnl_pct: float):
        """記錄交易結果"""
        self.recent_trades.append({'win': win, 'pnl_pct': pnl_pct})

    def _performance_multiplier(self) -> float:
        """
        近期勝率 → 槓桿乘數
        
        連勝 → 適度加碼
        連敗 → 強制減倉 (反馬丁格爾)
        """
        if len(self.recent_trades) < 5:
            return 1.0  # 數據不足, 中性
        
        wins = sum(1 for t in self.recent_trades if t["win"])
        total = len(self.recent_trades)
        win_rate = wins / total
        
        # 基礎乘數：基於勝率
        if win_rate >= PERF_BOOST_WINRATE:
            # 勝率高: 加碼但設上限
            multiplier = min(1.0 + (win_rate - PERF_BOOST_WINRATE) * 2.0, 1.3)
        elif win_rate >= PERF_REDUCE_WINRATE:
            # 中等: 中性
            multiplier = 1.0
        else:
            # 勝率低: 減倉
            multiplier = max(1.0 - (PERF_REDUCE_WINRATE - win_rate) * 2.5, 0.4)
        
        # 額外: 連敗檢測 (最近3筆全輸 → 強制上限0.5x)
        if len(self.recent_trades) >= 3:
            last_3 = list(self.recent_trades)[-3:]
            if all(not t["win"] for t in last_3):
                multiplier = min(multiplier, 0.5)
        
        return multiplier

    def _performance_multiplier_base(self) -> float:
        """基礎勝率乘數(不含連敗懲罰)"""
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

    # ── 因子4: 組合熱度 ─────────────────────────────────────
    def _portfolio_heat_multiplier(self, portfolio) -> float:
        """
        當前持倉佔比 → 槓桿乘數
        
        持倉越多 → 風險越集中 → 槓桿越低
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

    # ── 風險硬約束 ──────────────────────────────────────────
    def _drawdown_cap(self, portfolio) -> float:
        """回撤 → 槓桿上限"""
        dd = abs(portfolio.get_max_drawdown_pct())
        cap = MAX_LEVERAGE
        for threshold, max_lev in DRAWDOWN_CAPS:
            if dd >= threshold:
                cap = min(cap, max_lev)
        return cap

    # ── 主計算方法 ──────────────────────────────────────────
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
        計算動態槓桿倍數
        
        參數:
            confidence: ML預測置信度 (0.5-1.0)
            ticker: 目標股票代碼
            current_prices: 所有股票當前價格
            prev_prices: 前次價格(用于波動率估算)
            portfolio: PortfolioManager 實例
            accuracy: AccuracyTracker 實例
        
        返回:
            (leverage, detail_dict)
        """
        self.total_calculations += 1
        
        # 0. 如果沒有portfolio, 返回保守默認值
        if portfolio is None:
            return 1.0, {'kelly': 1.0, 'vol': 1.0, 'perf': 1.0, 'heat': 1.0, 
                        'dd_cap': MAX_LEVERAGE, 'margin': 1.0, 'final': 1.0}

        # 1. 凱利基礎槓桿
        # 從accuracy tracker獲取近期盈虧比
        win_loss_ratio = DEFAULT_WIN_LOSS_RATIO
        if accuracy is not None:
            # 用近期方向準確率推斷盈虧比
            acc_snap = accuracy.get_snapshot()
            recent_acc = acc_snap.recent_accuracy_50
            if recent_acc > 0.5:
                # 準確率越高, 推斷盈虧比越好
                win_loss_ratio = 1.2 + (recent_acc - 0.5) * 4.0
                win_loss_ratio = min(win_loss_ratio, MAX_WIN_LOSS_RATIO)
        
        kelly_lev = self._kelly_fraction(confidence, win_loss_ratio)

        # 2. 波動率乘數
        vol_mult = 1.0
        if current_prices and prev_prices:
            annual_vol = self._estimate_volatility(current_prices, prev_prices)
            vol_mult = self._volatility_multiplier(annual_vol)
        else:
            annual_vol = 0.0

        # 3. 績效反饋乘數
        perf_mult = self._performance_multiplier()

        # 4. 組合熱度乘數
        heat_mult = self._portfolio_heat_multiplier(portfolio)

        # 綜合計算
        leverage = kelly_lev * vol_mult * perf_mult * heat_mult

        # 回撤硬約束
        dd_cap = self._drawdown_cap(portfolio)
        leverage = min(leverage, dd_cap)

        # 保證金硬約束
        margin_ratio = portfolio.get_margin_ratio()
        margin_cap = MAX_LEVERAGE
        if margin_ratio > 0.90:
            margin_cap = 0.0  # 停止開倉
        elif margin_ratio > 0.80:
            margin_cap = 1.0
        elif margin_ratio > 0.60:
            margin_cap = 1.5
        leverage = min(leverage, margin_cap)

        # 全局 clamp (但尊重 Kelly=0 的"不下注"信號)
        if kelly_lev <= 0:
            leverage = 0.0  # Kelly 判定無優勢，不下注
        else:
            leverage = max(MIN_LEVERAGE, min(leverage, MAX_LEVERAGE))

        # 四捨五入到0.05
        leverage = round(leverage * 20) / 20

        # 更新統計
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
        """序列化為字典(用於狀態持久化)"""
        return {
            'recent_trades': list(self.recent_trades),
            'total_calculations': self.total_calculations,
            'avg_leverage': self.avg_leverage,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LeverageEngine':
        """從字典恢復"""
        engine = cls()
        engine.recent_trades = deque(
            data.get('recent_trades', []), maxlen=PERF_WINDOW
        )
        engine.total_calculations = data.get('total_calculations', 0)
        engine.avg_leverage = data.get('avg_leverage', 1.0)
        return engine
