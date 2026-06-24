"""Risk manager: stop-loss, take-profit, drawdown constraints."""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.settings import TradingConfig
from utils.constants import LULD, PDT_RULES, RegSHO
from utils.exceptions import (
    RiskError, RiskLimitExceededError, CircuitBreakerError
)
from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


# ============================================================================
# 風控級別
# ============================================================================
class RiskLevel(Enum):
    """風控警報級別"""
    INFO = 'INFO'        # 信息通知
    WARNING = 'WARNING'  # 警告（可能觸發軟限制）
    CRITICAL = 'CRITICAL'  # 嚴重（觸發硬限制，暫停交易）
    CIRCUIT_BREAKER = 'CIRCUIT_BREAKER'  # 熔斷


class RiskLimitType(Enum):
    """風控限制類型"""
    HARD = 'HARD'  # 硬限制：不可繞過
    SOFT = 'SOFT'  # 軟限制：可配置


@dataclass
class RiskCheckResult:
    """單項風控檢查結果"""
    passed: bool
    level: RiskLevel = RiskLevel.INFO
    rule_name: str = ''
    message: str = ''
    limit_type: RiskLimitType = RiskLimitType.SOFT
    current_value: Optional[float] = None
    limit_value: Optional[float] = None


# ============================================================================
# 帳戶狀態快照
# ============================================================================
@dataclass
class AccountSnapshot:
    """用於風控計算的帳戶狀態快照"""
    equity: float = 0.0
    cash: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    day_trade_count_5d: int = 0
    is_pdt: bool = False
    timestamp: str = ''


# ============================================================================
# 事前風控
# ============================================================================
class PreTradeRisk:
    """
    事前風控 (Pre-Trade Risk Controls)
    
    在下單前執行，阻止不符合風控規則的訂單進入市場。
    
    檢查項：
    1. 單筆訂單金額限制
    2. 單股持倉比例限制
    3. 總持倉數量限制
    4. 槓桿上限
    5. 流動性檢查（仙股過濾）
    6. PDT規則
    7. 禁投清單
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.restricted_tickers: set = set()  # 禁投清單
    
    def add_restricted_ticker(self, ticker: str) -> None:
        """添加禁投股票"""
        self.restricted_tickers.add(ticker.upper())
    
    def check_all(
        self,
        ticker: str,
        side: str,
        quantity: int,
        estimated_price: float,
        account: 'AccountSnapshot',
        current_positions: Dict[str, float]
    ) -> List[RiskCheckResult]:
        """
        執行全部事前風控檢查
        
        返回所有檢查結果，只要有一個CRITICAL失敗就應拒絕訂單。
        
        時間複雜度: O(1) - 固定數量檢查
        空間複雜度: O(1)
        
        參數:
            ticker: 股票代碼
            side: 買賣方向
            quantity: 訂單數量
            estimated_price: 預估價格
            account: 帳戶快照
            current_positions: 當前持倉 {ticker: market_value}
        
        返回:
            風險檢查結果列表
        """
        results = []
        
        # 1. 禁投清單檢查
        if ticker.upper() in self.restricted_tickers:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CRITICAL,
                rule_name='禁投清單',
                message=f"{ticker} 在禁投清單中",
                limit_type=RiskLimitType.HARD
            ))
        
        # 2. 單筆訂單金額限制
        order_value = quantity * estimated_price
        max_order_value = account.equity * self.config.max_order_amount_pct
        results.append(RiskCheckResult(
            passed=order_value <= max_order_value,
            level=RiskLevel.CRITICAL if order_value > max_order_value * 1.2 else RiskLevel.WARNING,
            rule_name='單筆訂單金額限制',
            message=f"訂單金額 ${order_value:,.0f} vs 限制 ${max_order_value:,.0f}",
            limit_type=RiskLimitType.HARD,
            current_value=order_value,
            limit_value=max_order_value
        ))
        
        # 3. 單股持倉比例限制
        if ticker.upper() in current_positions:
            current_pos_value = current_positions[ticker.upper()]
            new_pos_value = current_pos_value + (order_value if side == 'BUY' else -order_value)
            pos_pct = safe_divide(abs(new_pos_value), account.equity, 0)
            results.append(RiskCheckResult(
                passed=pos_pct <= self.config.max_position_pct,
                level=RiskLevel.CRITICAL if pos_pct > self.config.max_position_pct * 1.5 else RiskLevel.WARNING,
                rule_name='單股持倉比例',
                message=f"{ticker} 持倉比例 {pos_pct:.1%} vs 限制 {self.config.max_position_pct:.1%}",
                limit_type=RiskLimitType.SOFT,
                current_value=pos_pct,
                limit_value=self.config.max_position_pct
            ))
        
        # 4. 槓桿上限
        new_leverage = safe_divide(
            account.gross_exposure + order_value, account.equity, 0
        )
        results.append(RiskCheckResult(
            passed=new_leverage <= self.config.max_leverage,
            level=RiskLevel.CRITICAL,
            rule_name='槓桿上限',
            message=f"槓桿 {new_leverage:.2f}x vs 限制 {self.config.max_leverage:.1f}x",
            limit_type=RiskLimitType.HARD,
            current_value=new_leverage,
            limit_value=self.config.max_leverage
        ))
        
        # 5. 流動性檢查（仙股過濾）
        if estimated_price < 5.0:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CRITICAL,
                rule_name='仙股過濾',
                message=f"{ticker} 價格 ${estimated_price:.2f} < $5.00 (低流動性)",
                limit_type=RiskLimitType.HARD
            ))
        
        # 6. PDT規則檢查
        if account.is_pdt and account.day_trade_count_5d >= 3:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CRITICAL,
                rule_name='PDT限制',
                message=f"PDT帳戶日內交易 {account.day_trade_count_5d}/3次",
                limit_type=RiskLimitType.HARD
            ))
        
        # 7. 最大回撤檢查
        if account.max_drawdown_pct >= self.config.max_drawdown_pct:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='最大回撤觸發',
                message=f"回撤 {account.max_drawdown_pct:.1%} >= {self.config.max_drawdown_pct:.1%}",
                limit_type=RiskLimitType.HARD,
                current_value=account.max_drawdown_pct,
                limit_value=self.config.max_drawdown_pct
            ))
        
        return results
    
    def is_order_allowed(self, results: List[RiskCheckResult]) -> bool:
        """
        判斷訂單是否可通過風控
        
        規則：所有HARD限制必須通過，SOFT限制至少不觸發CRITICAL。
        
        參數:
            results: 風控檢查結果列表
        
        返回:
            是否允許
        """
        for r in results:
            if not r.passed:
                if r.limit_type == RiskLimitType.HARD:
                    return False
                if r.level == RiskLevel.CRITICAL:
                    return False
        return True


# ============================================================================
# 事中風控
# ============================================================================
class InTradeRisk:
    """
    事中風控 (In-Trade Risk Controls)
    
    在交易過程中持續監控，即時響應異常情況。
    
    監控項：
    1. 實時盈虧監控
    2. 回撤監控（觸及硬限制自動暫停交易）
    3. LULD熔斷應對
    4. 異常成交檢測
    5. 連續拒單檢測
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.consecutive_rejects: int = 0
        self.max_consecutive_rejects: int = 5
        self.is_paused: bool = False
        self.pause_reason: str = ''
        self.pause_until: Optional[datetime] = None
        
        # LULD 追蹤 {ticker: {reference_price, tier, ...}}
        self.luld_state: Dict[str, Dict] = {}
    
    def check_luld(self, ticker: str, price: float, 
                   previous_close: float) -> RiskCheckResult:
        """
        檢查個股是否觸發LULD熔斷
        
        當價格觸及漲停/跌停帶時，訂單無法在該價格執行。
        
        參數:
            ticker: 股票代碼
            price: 當前價格
            previous_close: 前收盤價
        
        返回:
            RiskCheckResult
        """
        # 簡化LULD檢查：假設Tier 1（±10%）
        band = LULD.TIER1_BANDS.get('regular', 0.10)
        
        upper_limit = previous_close * (1 + band)
        lower_limit = previous_close * (1 - band)
        
        if price > upper_limit:
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name=f'LULD-{ticker}',
                message=f"{ticker} 觸及漲停 ${upper_limit:.2f} (當前 ${price:.2f})",
                limit_type=RiskLimitType.HARD
            )
        
        if price < lower_limit:
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name=f'LULD-{ticker}',
                message=f"{ticker} 觸及跌停 ${lower_limit:.2f} (當前 ${price:.2f})",
                limit_type=RiskLimitType.HARD
            )
        
        return RiskCheckResult(passed=True, rule_name=f'LULD-{ticker}')
    
    def check_drawdown(self, current_drawdown_pct: float) -> RiskCheckResult:
        """
        檢查回撤是否觸發暫停
        
        當回撤超過配置的硬限制時，自動暫停所有交易。
        
        參數:
            current_drawdown_pct: 當前回撤百分比
        
        返回:
            RiskCheckResult
        """
        if current_drawdown_pct >= self.config.max_drawdown_pct:
            self.pause(reason=f'回撤 {current_drawdown_pct:.1%} >= {self.config.max_drawdown_pct:.1%}')
            
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='回撤暫停',
                message=f'回撤 {current_drawdown_pct:.1%} 觸發交易暫停',
                limit_type=RiskLimitType.HARD,
                current_value=current_drawdown_pct,
                limit_value=self.config.max_drawdown_pct
            )
        
        return RiskCheckResult(passed=True, rule_name='回撤正常')
    
    def on_order_rejected(self) -> RiskCheckResult:
        """
        處理訂單被拒事件
        
        連續拒單超過閾值時自動暫停交易。
        """
        self.consecutive_rejects += 1
        
        if self.consecutive_rejects >= self.max_consecutive_rejects:
            self.pause(reason=f'連續 {self.consecutive_rejects} 次拒單')
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='連續拒單',
                message=f'連續 {self.consecutive_rejects} 次訂單被拒，暫停交易',
                limit_type=RiskLimitType.HARD
            )
        
        return RiskCheckResult(
            passed=True,
            level=RiskLevel.WARNING,
            rule_name='拒單警告',
            message=f'訂單被拒 ({self.consecutive_rejects}/{self.max_consecutive_rejects})'
        )
    
    def on_order_accepted(self) -> None:
        """訂單被接受後重置拒單計數"""
        self.consecutive_rejects = 0
    
    def pause(self, reason: str, duration_minutes: int = 30) -> None:
        """
        暫停所有交易
        
        參數:
            reason: 暫停原因
            duration_minutes: 暫停時長（分鐘）
        """
        self.is_paused = True
        self.pause_reason = reason
        self.pause_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        # 記錄風控日誌
        from config.logging_config import LogManager
        risk_logger = LogManager.get_risk_logger()
        risk_logger.warning(f"交易暫停: {reason}, 恢復時間: {self.pause_until}")
    
    def resume(self) -> None:
        """恢復交易（手動或超時後）"""
        if self.is_paused and self.pause_until:
            if datetime.now() >= self.pause_until:
                self.is_paused = False
                self.consecutive_rejects = 0
                logger.info("交易已恢復")
    
    def check_slippage(self, expected_price: float, 
                       actual_price: float) -> RiskCheckResult:
        """
        檢查滑點是否異常
        
        參數:
            expected_price: 預期成交價
            actual_price: 實際成交價
        
        返回:
            RiskCheckResult
        """
        if expected_price <= 0:
            return RiskCheckResult(
                passed=True,
                level=RiskLevel.OK,
                rule_name='異常滑點',
                message=f'跳過滑點檢查（預期價格無效: {expected_price}）',
                current_value=0,
                limit_value=0.05
            )
        
        slippage_pct = abs(actual_price - expected_price) / expected_price
        
        if slippage_pct > 0.05:  # 5%滑點
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.WARNING,
                rule_name='異常滑點',
                message=f'滑點 {slippage_pct:.2%} 超過5%閾值',
                current_value=slippage_pct,
                limit_value=0.05
            )
        
        return RiskCheckResult(passed=True, rule_name='滑點正常')


# ============================================================================
# 事後風控
# ============================================================================
class PostTradeRisk:
    """
    事後風控 (Post-Trade Risk Analysis)
    
    盤後或定期執行的風險分析和報告。
    
    分析內容：
    1. VaR (Value at Risk) - 在險價值
    2. CVaR (Conditional VaR) - 條件在險價值
    3. 壓力測試 - 極端市場情景模擬
    4. 盈虧歸因 - 分解收益來源
    """
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        計算歷史模擬法 VaR
        
        VaR_α: 在置信水平(1-α)下，預計可能遭受的最大損失。
        
        例如：95%日VaR = -2% 意味著有95%的把握，明日虧損不超過2%。
        
        時間複雜度: O(n log n) - 排序
        空間複雜度: O(n)
        
        參數:
            returns: 日收益率數組
            confidence: 置信水平
        
        返回:
            VaR值（負值表示損失）
        """
        if len(returns) == 0:
            return 0.0
        
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        計算 CVaR (Expected Shortfall)
        
        CVaR是超出VaR閾值的平均損失，比VaR更保守。
        
        時間複雜度: O(n log n)
        空間複雜度: O(n)
        
        參數:
            returns: 日收益率數組
            confidence: 置信水平
        
        返回:
            CVaR值
        """
        var_threshold = PostTradeRisk.calculate_var(returns, confidence)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return var_threshold
        
        return float(tail_losses.mean())
    
    @staticmethod
    def stress_test(
        positions: Dict[str, float],
        scenarios: Dict[str, float]
    ) -> Dict[str, float]:
        """
        壓力測試
        
        模擬極端市場情景下的組合損益。
        
        預設情景：
        - market_crash_20: 市場暴跌20%
        - tech_selloff: 科技股暴跌15%
        - rate_hike: 加息衝擊（金融+3%，成長股-10%）
        - vix_spike: VIX暴漲（全市場波動率翻倍）
        
        參數:
            positions: {ticker: market_value}
            scenarios: {scenario_name: pct_change}
        
        返回:
            {scenario_name: pnl_impact}
        """
        results = {}
        total_value = sum(abs(v) for v in positions.values())
        
        for name, pct_change in scenarios.items():
            impact = total_value * pct_change
            results[name] = impact
        
        return results
    
    @staticmethod
    def attribution_report(
        strategy_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        盈虧歸因報告
        
        分解策略收益為：
        - Alpha: 超額收益（策略獨有）
        - Beta: 市場收益（系統性風險補償）
        - 殘差: 無法解釋的部分
        
        使用CAPM回歸: r_strategy = α + β * r_market + ε
        
        參數:
            strategy_returns: 策略收益率
            market_returns: 市場收益率
            risk_free_rate: 無風險利率
        
        返回:
            歸因分析結果字典
        """
        # 對齊數據
        aligned = pd.concat([strategy_returns, market_returns], axis=1).dropna()
        
        if len(aligned) < 20:
            return {'error': '數據不足'}
        
        y = aligned[strategy_returns.name]
        x = aligned[market_returns.name]
        
        # OLS 回歸
        n = len(y)
        beta = np.cov(y, x)[0, 1] / np.var(x) if np.var(x) > 0 else 0
        alpha = y.mean() - beta * x.mean()
        
        # 年化
        alpha_annual = alpha * 252
        beta_annual = beta
        
        # R²
        y_pred = alpha + beta * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'alpha_daily': float(alpha),
            'alpha_annual': float(alpha_annual),
            'beta': float(beta_annual),
            'r_squared': float(r_squared),
            'market_contribution': float(beta * x.mean() * 252),
            'alpha_contribution': float(alpha_annual),
        }


# ============================================================================
# 統一風控管理器
# ============================================================================
class RiskManager:
    """
    統一風控管理器
    
    整合事前、事中、事後三重風控，提供統一入口。
    
    使用示例:
        rm = RiskManager(trading_config)
        
        # 下單前
        results = rm.pre_trade_check(ticker, side, qty, price, account)
        if rm.is_order_allowed(results):
            submit_order()
        
        # 交易中
        rm.on_fill(ticker, price, expected_price)
        
        # 盤後
        report = rm.generate_risk_report()
    """
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.pre_trade = PreTradeRisk(self.config)
        self.in_trade = InTradeRisk(self.config)
        self.post_trade = PostTradeRisk()
        
        # 收益歷史記錄
        self.daily_returns: List[float] = []
        self.equity_history: List[float] = [self.config.initial_capital]
    
    def pre_trade_check(
        self, ticker: str, side: str, quantity: int,
        estimated_price: float, account: AccountSnapshot,
        current_positions: Dict[str, float]
    ) -> List[RiskCheckResult]:
        """下單前全面風控檢查"""
        # 先檢查是否處於暫停狀態
        if self.in_trade.is_paused:
            return [RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='交易暫停',
                message=f'交易已暫停: {self.in_trade.pause_reason}',
                limit_type=RiskLimitType.HARD
            )]
        
        return self.pre_trade.check_all(
            ticker, side, quantity, estimated_price,
            account, current_positions
        )
    
    def is_order_allowed(self, results):
        """Check if all risk checks passed (delegates to PreTradeRisk)."""
        return self.pre_trade.is_order_allowed(results)
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """生成綜合風險報告"""
        returns = np.array(self.daily_returns)
        
        if len(returns) == 0:
            return {'status': '無數據'}
        
        return {
            'var_95': self.post_trade.calculate_var(returns, 0.95),
            'var_99': self.post_trade.calculate_var(returns, 0.99),
            'cvar_95': self.post_trade.calculate_cvar(returns, 0.95),
            'volatility_annual': float(np.std(returns) * np.sqrt(252)),
            'max_drawdown': self._calc_max_drawdown(),
            'sharpe_ratio': safe_divide(
                np.mean(returns) * 252,
                np.std(returns) * np.sqrt(252), 0
            ),
            'positive_days_pct': float(np.mean(returns > 0)),
            'total_return': self.equity_history[-1] / self.equity_history[0] - 1,
        }
    
    def _calc_max_drawdown(self) -> float:
        """計算最大回撤"""
        if len(self.equity_history) < 2:
            return 0.0
        
        equity = np.array(self.equity_history)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return float(drawdown.min())
