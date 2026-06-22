"""
美股量化交易系统 - 实时风险管理系统

这是量化交易能否长期存活的最重要模块，必须做到事前、事中、事后三重防护：

一、事前风控 (Pre-Trade)
- 单笔/单日/单股最大下单量限制
- 净暴露与杠杆上限
- 流动性检查（禁止交易仙股）
- PDT规则检查

二、事中风控 (In-Trade)
- 实时盈亏与回撤监控
- LULD熔断机制应对
- 异常行为检测（连续拒单、滑价过大）
- 自动减仓/暂停交易机制

三、事后风控 (Post-Trade)
- VaR / CVaR 计算
- 压力测试 / 极端情景模拟
- 盈亏归因分析
- 风险报告生成

设计原则：
- 三层防御互相独立，任一环节可独立触发熔断
- 所有风控决策记录完整审计日志
- 硬限制(Hard Limits)不可被策略覆盖，软限制(Soft Limits)可配置
"""

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
# 风控级别
# ============================================================================
class RiskLevel(Enum):
    """风控警报级别"""
    INFO = 'INFO'        # 信息通知
    WARNING = 'WARNING'  # 警告（可能触发软限制）
    CRITICAL = 'CRITICAL'  # 严重（触发硬限制，暂停交易）
    CIRCUIT_BREAKER = 'CIRCUIT_BREAKER'  # 熔断


class RiskLimitType(Enum):
    """风控限制类型"""
    HARD = 'HARD'  # 硬限制：不可绕过
    SOFT = 'SOFT'  # 软限制：可配置


@dataclass
class RiskCheckResult:
    """单项风控检查结果"""
    passed: bool
    level: RiskLevel = RiskLevel.INFO
    rule_name: str = ''
    message: str = ''
    limit_type: RiskLimitType = RiskLimitType.SOFT
    current_value: Optional[float] = None
    limit_value: Optional[float] = None


# ============================================================================
# 账户状态快照
# ============================================================================
@dataclass
class AccountSnapshot:
    """用于风控计算的账户状态快照"""
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
# 事前风控
# ============================================================================
class PreTradeRisk:
    """
    事前风控 (Pre-Trade Risk Controls)
    
    在下单前执行，阻止不符合风控规则的订单进入市场。
    
    检查项：
    1. 单笔订单金额限制
    2. 单股持仓比例限制
    3. 总持仓数量限制
    4. 杠杆上限
    5. 流动性检查（仙股过滤）
    6. PDT规则
    7. 禁投清单
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.restricted_tickers: set = set()  # 禁投清单
    
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
        执行全部事前风控检查
        
        返回所有检查结果，只要有一个CRITICAL失败就应拒绝订单。
        
        时间复杂度: O(1) - 固定数量检查
        空间复杂度: O(1)
        
        参数:
            ticker: 股票代码
            side: 买卖方向
            quantity: 订单数量
            estimated_price: 预估价格
            account: 账户快照
            current_positions: 当前持仓 {ticker: market_value}
        
        返回:
            风险检查结果列表
        """
        results = []
        
        # 1. 禁投清单检查
        if ticker.upper() in self.restricted_tickers:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CRITICAL,
                rule_name='禁投清单',
                message=f"{ticker} 在禁投清单中",
                limit_type=RiskLimitType.HARD
            ))
        
        # 2. 单笔订单金额限制
        order_value = quantity * estimated_price
        max_order_value = account.equity * self.config.max_order_amount_pct
        results.append(RiskCheckResult(
            passed=order_value <= max_order_value,
            level=RiskLevel.CRITICAL if order_value > max_order_value * 1.2 else RiskLevel.WARNING,
            rule_name='单笔订单金额限制',
            message=f"订单金额 ${order_value:,.0f} vs 限制 ${max_order_value:,.0f}",
            limit_type=RiskLimitType.HARD,
            current_value=order_value,
            limit_value=max_order_value
        ))
        
        # 3. 单股持仓比例限制
        if ticker.upper() in current_positions:
            current_pos_value = current_positions[ticker.upper()]
            new_pos_value = current_pos_value + (order_value if side == 'BUY' else -order_value)
            pos_pct = safe_divide(abs(new_pos_value), account.equity, 0)
            results.append(RiskCheckResult(
                passed=pos_pct <= self.config.max_position_pct,
                level=RiskLevel.CRITICAL if pos_pct > self.config.max_position_pct * 1.5 else RiskLevel.WARNING,
                rule_name='单股持仓比例',
                message=f"{ticker} 持仓比例 {pos_pct:.1%} vs 限制 {self.config.max_position_pct:.1%}",
                limit_type=RiskLimitType.SOFT,
                current_value=pos_pct,
                limit_value=self.config.max_position_pct
            ))
        
        # 4. 杠杆上限
        new_leverage = safe_divide(
            account.gross_exposure + order_value, account.equity, 0
        )
        results.append(RiskCheckResult(
            passed=new_leverage <= self.config.max_leverage,
            level=RiskLevel.CRITICAL,
            rule_name='杠杆上限',
            message=f"杠杆 {new_leverage:.2f}x vs 限制 {self.config.max_leverage:.1f}x",
            limit_type=RiskLimitType.HARD,
            current_value=new_leverage,
            limit_value=self.config.max_leverage
        ))
        
        # 5. 流动性检查（仙股过滤）
        if estimated_price < 5.0:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CRITICAL,
                rule_name='仙股过滤',
                message=f"{ticker} 价格 ${estimated_price:.2f} < $5.00 (低流动性)",
                limit_type=RiskLimitType.HARD
            ))
        
        # 6. PDT规则检查
        if account.is_pdt and account.day_trade_count_5d >= 3:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CRITICAL,
                rule_name='PDT限制',
                message=f"PDT账户日内交易 {account.day_trade_count_5d}/3次",
                limit_type=RiskLimitType.HARD
            ))
        
        # 7. 最大回撤检查
        if account.max_drawdown_pct >= self.config.max_drawdown_pct:
            results.append(RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='最大回撤触发',
                message=f"回撤 {account.max_drawdown_pct:.1%} >= {self.config.max_drawdown_pct:.1%}",
                limit_type=RiskLimitType.HARD,
                current_value=account.max_drawdown_pct,
                limit_value=self.config.max_drawdown_pct
            ))
        
        return results
    
    def is_order_allowed(self, results: List[RiskCheckResult]) -> bool:
        """
        判断订单是否可通过风控
        
        规则：所有HARD限制必须通过，SOFT限制至少不触发CRITICAL。
        
        参数:
            results: 风控检查结果列表
        
        返回:
            是否允许
        """
        for r in results:
            if not r.passed:
                if r.limit_type == RiskLimitType.HARD:
                    return False
                if r.level == RiskLevel.CRITICAL:
                    return False
        return True


# ============================================================================
# 事中风控
# ============================================================================
class InTradeRisk:
    """
    事中风控 (In-Trade Risk Controls)
    
    在交易过程中持续监控，即时响应异常情况。
    
    监控项：
    1. 实时盈亏监控
    2. 回撤监控（触及硬限制自动暂停交易）
    3. LULD熔断应对
    4. 异常成交检测
    5. 连续拒单检测
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.consecutive_rejects: int = 0
        self.max_consecutive_rejects: int = 5
        self.is_paused: bool = False
        self.pause_reason: str = ''
        self.pause_until: Optional[datetime] = None
        
        # LULD 追踪 {ticker: {reference_price, tier, ...}}
        self.luld_state: Dict[str, Dict] = {}
    
    def check_luld(self, ticker: str, price: float, 
                   previous_close: float) -> RiskCheckResult:
        """
        检查个股是否触发LULD熔断
        
        当价格触及涨停/跌停带时，订单无法在该价格执行。
        
        参数:
            ticker: 股票代码
            price: 当前价格
            previous_close: 前收盘价
        
        返回:
            RiskCheckResult
        """
        # 简化LULD检查：假设Tier 1（±10%）
        band = LULD.TIER1_BANDS.get('regular', 0.10)
        
        upper_limit = previous_close * (1 + band)
        lower_limit = previous_close * (1 - band)
        
        if price > upper_limit:
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name=f'LULD-{ticker}',
                message=f"{ticker} 触及涨停 ${upper_limit:.2f} (当前 ${price:.2f})",
                limit_type=RiskLimitType.HARD
            )
        
        if price < lower_limit:
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name=f'LULD-{ticker}',
                message=f"{ticker} 触及跌停 ${lower_limit:.2f} (当前 ${price:.2f})",
                limit_type=RiskLimitType.HARD
            )
        
        return RiskCheckResult(passed=True, rule_name=f'LULD-{ticker}')
    
    def check_drawdown(self, current_drawdown_pct: float) -> RiskCheckResult:
        """
        检查回撤是否触发暂停
        
        当回撤超过配置的硬限制时，自动暂停所有交易。
        
        参数:
            current_drawdown_pct: 当前回撤百分比
        
        返回:
            RiskCheckResult
        """
        if current_drawdown_pct >= self.config.max_drawdown_pct:
            self.pause(reason=f'回撤 {current_drawdown_pct:.1%} >= {self.config.max_drawdown_pct:.1%}')
            
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='回撤暂停',
                message=f'回撤 {current_drawdown_pct:.1%} 触发交易暂停',
                limit_type=RiskLimitType.HARD,
                current_value=current_drawdown_pct,
                limit_value=self.config.max_drawdown_pct
            )
        
        return RiskCheckResult(passed=True, rule_name='回撤正常')
    
    def on_order_rejected(self) -> RiskCheckResult:
        """
        处理订单被拒事件
        
        连续拒单超过阈值时自动暂停交易。
        """
        self.consecutive_rejects += 1
        
        if self.consecutive_rejects >= self.max_consecutive_rejects:
            self.pause(reason=f'连续 {self.consecutive_rejects} 次拒单')
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='连续拒单',
                message=f'连续 {self.consecutive_rejects} 次订单被拒，暂停交易',
                limit_type=RiskLimitType.HARD
            )
        
        return RiskCheckResult(
            passed=True,
            level=RiskLevel.WARNING,
            rule_name='拒单警告',
            message=f'订单被拒 ({self.consecutive_rejects}/{self.max_consecutive_rejects})'
        )
    
    def on_order_accepted(self) -> None:
        """订单被接受后重置拒单计数"""
        self.consecutive_rejects = 0
    
    def pause(self, reason: str, duration_minutes: int = 30) -> None:
        """
        暂停所有交易
        
        参数:
            reason: 暂停原因
            duration_minutes: 暂停时长（分钟）
        """
        self.is_paused = True
        self.pause_reason = reason
        self.pause_until = datetime.now() + timedelta(minutes=duration_minutes)
        
        # 记录风控日志
        from config.logging_config import LogManager
        risk_logger = LogManager.get_risk_logger()
        risk_logger.warning(f"交易暂停: {reason}, 恢复时间: {self.pause_until}")
    
    def resume(self) -> None:
        """恢复交易（手动或超时后）"""
        if self.is_paused and self.pause_until:
            if datetime.now() >= self.pause_until:
                self.is_paused = False
                self.consecutive_rejects = 0
                logger.info("交易已恢复")
    
    def check_slippage(self, expected_price: float, 
                       actual_price: float) -> RiskCheckResult:
        """
        检查滑点是否异常
        
        参数:
            expected_price: 预期成交价
            actual_price: 实际成交价
        
        返回:
            RiskCheckResult
        """
        if expected_price <= 0:
            return RiskCheckResult(
                passed=True,
                level=RiskLevel.OK,
                rule_name='异常滑点',
                message=f'跳过滑点检查（预期价格无效: {expected_price}）',
                current_value=0,
                limit_value=0.05
            )
        
        slippage_pct = abs(actual_price - expected_price) / expected_price
        
        if slippage_pct > 0.05:  # 5%滑点
            return RiskCheckResult(
                passed=False,
                level=RiskLevel.WARNING,
                rule_name='异常滑点',
                message=f'滑点 {slippage_pct:.2%} 超过5%阈值',
                current_value=slippage_pct,
                limit_value=0.05
            )
        
        return RiskCheckResult(passed=True, rule_name='滑点正常')


# ============================================================================
# 事后风控
# ============================================================================
class PostTradeRisk:
    """
    事后风控 (Post-Trade Risk Analysis)
    
    盘后或定期执行的风险分析和报告。
    
    分析内容：
    1. VaR (Value at Risk) - 在险价值
    2. CVaR (Conditional VaR) - 条件在险价值
    3. 压力测试 - 极端市场情景模拟
    4. 盈亏归因 - 分解收益来源
    """
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        计算历史模拟法 VaR
        
        VaR_α: 在置信水平(1-α)下，预计可能遭受的最大损失。
        
        例如：95%日VaR = -2% 意味着有95%的把握，明日亏损不超过2%。
        
        时间复杂度: O(n log n) - 排序
        空间复杂度: O(n)
        
        参数:
            returns: 日收益率数组
            confidence: 置信水平
        
        返回:
            VaR值（负值表示损失）
        """
        if len(returns) == 0:
            return 0.0
        
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        计算 CVaR (Expected Shortfall)
        
        CVaR是超出VaR阈值的平均损失，比VaR更保守。
        
        时间复杂度: O(n log n)
        空间复杂度: O(n)
        
        参数:
            returns: 日收益率数组
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
        压力测试
        
        模拟极端市场情景下的组合损益。
        
        预设情景：
        - market_crash_20: 市场暴跌20%
        - tech_selloff: 科技股暴跌15%
        - rate_hike: 加息冲击（金融+3%，成长股-10%）
        - vix_spike: VIX暴涨（全市场波动率翻倍）
        
        参数:
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
        盈亏归因报告
        
        分解策略收益为：
        - Alpha: 超额收益（策略独有）
        - Beta: 市场收益（系统性风险补偿）
        - 残差: 无法解释的部分
        
        使用CAPM回归: r_strategy = α + β * r_market + ε
        
        参数:
            strategy_returns: 策略收益率
            market_returns: 市场收益率
            risk_free_rate: 无风险利率
        
        返回:
            归因分析结果字典
        """
        # 对齐数据
        aligned = pd.concat([strategy_returns, market_returns], axis=1).dropna()
        
        if len(aligned) < 20:
            return {'error': '数据不足'}
        
        y = aligned[strategy_returns.name]
        x = aligned[market_returns.name]
        
        # OLS 回归
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
# 统一风控管理器
# ============================================================================
class RiskManager:
    """
    统一风控管理器
    
    整合事前、事中、事后三重风控，提供统一入口。
    
    使用示例:
        rm = RiskManager(trading_config)
        
        # 下单前
        results = rm.pre_trade_check(ticker, side, qty, price, account)
        if rm.is_order_allowed(results):
            submit_order()
        
        # 交易中
        rm.on_fill(ticker, price, expected_price)
        
        # 盘后
        report = rm.generate_risk_report()
    """
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or TradingConfig()
        self.pre_trade = PreTradeRisk(self.config)
        self.in_trade = InTradeRisk(self.config)
        self.post_trade = PostTradeRisk()
        
        # 收益历史记录
        self.daily_returns: List[float] = []
        self.equity_history: List[float] = [self.config.initial_capital]
    
    def pre_trade_check(
        self, ticker: str, side: str, quantity: int,
        estimated_price: float, account: AccountSnapshot,
        current_positions: Dict[str, float]
    ) -> List[RiskCheckResult]:
        """下单前全面风控检查"""
        # 先检查是否处于暂停状态
        if self.in_trade.is_paused:
            return [RiskCheckResult(
                passed=False,
                level=RiskLevel.CIRCUIT_BREAKER,
                rule_name='交易暂停',
                message=f'交易已暂停: {self.in_trade.pause_reason}',
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
        """生成综合风险报告"""
        returns = np.array(self.daily_returns)
        
        if len(returns) == 0:
            return {'status': '无数据'}
        
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
        """计算最大回撤"""
        if len(self.equity_history) < 2:
            return 0.0
        
        equity = np.array(self.equity_history)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return float(drawdown.min())
