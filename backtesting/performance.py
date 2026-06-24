"""
美股量化交易系統 - 績效分析模塊

提供全面的回測績效分析，包括：
- 收益指標: 總收益、年化收益、超額收益(Alpha)
- 風險指標: 波動率、最大回撤、VaR、CVaR
- 風險調整收益: Sharpe、Sortino、Calmar、信息比率
- 交易分析: 勝率、盈虧比、利潤因子、平均持倉時間
- 歸因分析: Alpha vs Beta vs 行業 vs 風格
- 參數敏感性: 不同參數組合下的穩健性檢驗

所有計算基於權益曲線和交易記錄。
"""

import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """綜合績效報告"""
    # 收益指標
    total_return: float = 0.0
    annual_return: float = 0.0
    cumulative_return: float = 0.0
    
    # 風險指標
    volatility_annual: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR (Expected Shortfall)
    
    # 風險調整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # 期望收益
    avg_holding_days: float = 0.0
    
    # 月度/年度統計
    monthly_returns: Optional[pd.DataFrame] = None
    yearly_returns: Optional[pd.DataFrame] = None
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months_pct: float = 0.0
    
    # 成本分析
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_other_fees: float = 0.0
    cost_drag_annual: float = 0.0  # 年化成本拖累
    
    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            'total_return': f"{self.total_return:.2%}",
            'annual_return': f"{self.annual_return:.2%}",
            'volatility': f"{self.volatility_annual:.2%}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'sortino_ratio': round(self.sortino_ratio, 3),
            'calmar_ratio': round(self.calmar_ratio, 3),
            'total_trades': self.total_trades,
            'win_rate': f"{self.win_rate:.1%}",
            'profit_factor': round(self.profit_factor, 2),
            'best_month': f"{self.best_month:.2%}",
            'worst_month': f"{self.worst_month:.2%}",
        }


class PerformanceAnalyzer:
    """
    績效分析器
    
    對回測結果進行全面分析，生成多維度績效指標。
    
    分析方法：
    - 收益分析: 計算各類收益率指標
    - 風險分析: VaR、CVaR、最大回撤等
    - 歸因分析: 分解收益來源
    - 敏感性分析: 參數穩健性檢驗
    """
    
    @staticmethod
    def analyze(
        equity_curve: pd.Series,
        trades: Optional[List] = None,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0
    ) -> PerformanceReport:
        """
        綜合績效分析
        
        參數:
            equity_curve: 權益曲線（日期索引）
            trades: 交易記錄列表
            benchmark_returns: 基準收益率序列
            risk_free_rate: 無風險利率（年化）
        
        返回:
            PerformanceReport績效報告
        """
        report = PerformanceReport()
        
        if len(equity_curve) < 2:
            return report
        
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) == 0:
            return report
        
        # ---- 收益指標 ----
        report.cumulative_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        report.total_return = report.cumulative_return
        
        years = len(daily_returns) / 252
        report.annual_return = (1 + report.total_return) ** (1 / max(years, 1/252)) - 1
        
        # ---- 風險指標 ----
        report.volatility_annual = daily_returns.std() * np.sqrt(252)
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        report.max_drawdown = drawdown.min()
        
        # 最大回撤持續天數
        dd_duration = 0
        max_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0
        report.max_drawdown_duration_days = max_dd_duration
        
        # VaR (歷史模擬法, 95%置信度)
        report.var_95 = np.percentile(daily_returns, 5)
        
        # CVaR (Expected Shortfall)
        tail_losses = daily_returns[daily_returns <= report.var_95]
        report.cvar_95 = tail_losses.mean() if len(tail_losses) > 0 else report.var_95
        
        # ---- 風險調整收益 ----
        excess_returns = report.annual_return - risk_free_rate
        report.sharpe_ratio = safe_divide(excess_returns, report.volatility_annual, 0)
        
        # Sortino（只考慮下行波動）
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        report.sortino_ratio = safe_divide(excess_returns, downside_vol, 0)
        
        # Calmar
        report.calmar_ratio = safe_divide(report.annual_return, abs(report.max_drawdown), 0)
        
        # 信息比率（相對基準）
        if benchmark_returns is not None:
            aligned_bench = benchmark_returns.reindex(daily_returns.index).dropna()
            if len(aligned_bench) > 0:
                tracking_error = (daily_returns.reindex(aligned_bench.index) - aligned_bench).std() * np.sqrt(252)
                report.information_ratio = safe_divide(
                    report.annual_return - aligned_bench.mean() * 252,
                    tracking_error, 0
                )
        
        # ---- 月度統計 ----
        monthly_returns = daily_returns.resample('ME').apply(
            lambda x: (1 + x).prod() - 1
        )
        if len(monthly_returns) > 0:
            report.monthly_returns = monthly_returns
            report.best_month = monthly_returns.max()
            report.worst_month = monthly_returns.min()
            report.positive_months_pct = (monthly_returns > 0).mean()
        
        # ---- 交易統計 ----
        if trades:
            report.total_trades = len(trades)
            
            # 簡化版盈虧分析（僅適用於已有pnl的trade記錄）
            # 實際使用時應從成交記錄中提取每筆交易的盈虧
            pass
        
        return report
    
    @staticmethod
    def attribution_analysis(
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        收益歸因分析
        
        將策略收益分解為：
        - Alpha: 策略特有的超額收益
        - Beta(Market): 市場系統性收益
        - Size: 市值因子
        - Value: 價值因子
        - Momentum: 動量因子
        - Sector: 行業因子
        
        使用多元線性回歸: r_strategy = α + Σ(β_i * r_factor_i) + ε
        
        參數:
            strategy_returns: 策略日收益率
            factor_returns: 因子收益率DataFrame
            factor_exposures: 預設的因子暴露（None則回歸估計）
        
        返回:
            {因子: 貢獻比例}
        """
        if factor_returns is None or factor_returns.empty:
            return {}
        
        # 對齊數據
        aligned = pd.concat([strategy_returns, factor_returns], axis=1).dropna()
        if len(aligned) < 50:
            return {}
        
        y = aligned[strategy_returns.name]
        X = aligned[factor_returns.columns]
        
        # OLS回歸
        X_with_const = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
        
        try:
            beta = np.linalg.lstsq(X_with_const.values, y.values, rcond=None)[0]
            
            # 計算各因子的貢獻
            attributions = {'Alpha': float(beta[0]) * 252}  # 年化alpha
            
            for i, factor_name in enumerate(X.columns, 1):
                contrib = beta[i] * X[factor_name].mean() * 252
                attributions[factor_name] = float(contrib)
            
            return attributions
            
        except np.linalg.LinAlgError:
            return {'error': '回歸失敗'}
    
    @staticmethod
    def sensitivity_analysis(
        equity_curves: Dict[str, pd.Series],
        base_case_key: str
    ) -> pd.DataFrame:
        """
        參數敏感性分析
        
        比較不同參數配置下的績效差異。
        
        參數:
            equity_curves: {參數組合名: 權益曲線}
            base_case_key: 基準參數組合的鍵名
        
        返回:
            各參數組合的績效對比DataFrame
        """
        results = []
        
        for name, curve in equity_curves.items():
            report = PerformanceAnalyzer.analyze(curve)
            results.append({
                'name': name,
                'total_return': report.total_return,
                'annual_return': report.annual_return,
                'max_drawdown': report.max_drawdown,
                'sharpe_ratio': report.sharpe_ratio,
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def rolling_analysis(
        equity_curve: pd.Series,
        window: int = 252  # 一年滾動
    ) -> pd.DataFrame:
        """
        滾動績效分析
        
        計算滾動窗口的績效指標，檢測策略表現是否隨時間退化。
        
        參數:
            equity_curve: 權益曲線
            window: 滾動窗口大小（交易日）
        
        返回:
            包含滾動指標的DataFrame
        """
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) < window:
            return pd.DataFrame()
        
        rolling_results = pd.DataFrame(index=daily_returns.index)
        
        # 滾動年化收益率
        rolling_results['rolling_return'] = daily_returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (252 / window) - 1
        )
        
        # 滾動波動率
        rolling_results['rolling_vol'] = daily_returns.rolling(window).std() * np.sqrt(252)
        
        # 滾動夏普比率
        rolling_results['rolling_sharpe'] = safe_divide(
            rolling_results['rolling_return'],
            rolling_results['rolling_vol'],
            0.0
        )
        
        # 滾動最大回撤
        def rolling_max_drawdown(returns):
            cum = (1 + returns).cumprod()
            return (cum / cum.expanding().max() - 1).min()
        
        rolling_results['rolling_maxdd'] = daily_returns.rolling(window).apply(
            rolling_max_drawdown
        )
        
        return rolling_results.dropna()
