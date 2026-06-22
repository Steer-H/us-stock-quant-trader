"""
交易系统状态持久化模块

功能：
- 将 PortfolioManager / AccuracyTracker / BenchmarkTracker 序列化为 JSON
- 从 JSON 恢复完整交易状态
- 自动保存 + 崩溃恢复

序列化策略：
- dataclass → dict → JSON
- datetime → ISO 8601 字符串
- deque → list
- defaultdict → dict
- set → list

设计原则：
- 保存文件: data/trading_state.json
- 自动保存间隔: 60秒（引擎循环中）
- 启动时自动检测并恢复
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# 默认状态文件路径
DEFAULT_STATE_PATH = Path('data/trading_state.json')


def serialize_portfolio(portfolio) -> Dict[str, Any]:
    """
    序列化 PortfolioManager
    
    保存所有关键状态：持仓、交易历史、现金、盈亏等
    """
    positions = {}
    for ticker, pos in portfolio.positions.items():
        positions[ticker] = {
            'ticker': pos.ticker,
            'quantity': pos.quantity,
            'avg_cost': pos.avg_cost,
            'current_price': pos.current_price,
            'market_value': pos.market_value,
            'cost_basis': pos.cost_basis,
            'unrealized_pnl': pos.unrealized_pnl,
            'unrealized_pnl_pct': pos.unrealized_pnl_pct,
            'day_change': pos.day_change,
            'day_change_pct': pos.day_change_pct,
            'weight': pos.weight,
            'last_update': pos.last_update,
        }
    
    trades = []
    for t in portfolio.trade_history:
        trades.append({
            'trade_id': t.trade_id,
            'ticker': t.ticker,
            'side': t.side,
            'quantity': t.quantity,
            'price': t.price,
            'total_value': t.total_value,
            'commission': t.commission,
            'timestamp': t.timestamp,
            'pnl': t.pnl,
            'reason': t.reason,
        })
    
    equity_history = [(date_str, eq) for date_str, eq in portfolio._equity_history]
    
    return {
        'initial_capital': portfolio.initial_capital,
        'cash': portfolio.cash,
        'realized_pnl': portfolio.realized_pnl,
        'trade_history': trades,
        '_trade_id_counter': portfolio._trade_id_counter,
        '_equity_history': equity_history,
        '_day_start_equity': portfolio._day_start_equity,
        '_day_start_date': str(portfolio._day_start_date) if portfolio._day_start_date else None,
        '_peak_equity': portfolio._peak_equity,
        'total_commission': portfolio.total_commission,
        'borrowed': portfolio.borrowed,
        'total_interest': portfolio.total_interest,
        'margin_used': portfolio.margin_used,
        'max_leverage': portfolio.max_leverage,
        '_interest_rate_annual': portfolio._interest_rate_annual,
        '_last_interest_calc': str(portfolio._last_interest_calc) if portfolio._last_interest_calc else None,
        'positions': positions,
    }


def deserialize_portfolio(portfolio, data: Dict[str, Any]) -> None:
    """
    从序列化数据恢复 PortfolioManager
    
    直接修改传入的 portfolio 对象，恢复其状态。
    """
    from live_trading.portfolio import HoldingPosition, TradeRecord
    
    portfolio.initial_capital = data['initial_capital']
    portfolio.cash = data['cash']
    portfolio.realized_pnl = data['realized_pnl']
    portfolio._trade_id_counter = data.get('_trade_id_counter', 0)
    portfolio.total_commission = data.get('total_commission', 0.0)
    portfolio.borrowed = data.get('borrowed', 0.0)
    portfolio.total_interest = data.get('total_interest', 0.0)
    portfolio.margin_used = data.get('margin_used', 0.0)
    portfolio.max_leverage = data.get('max_leverage', 2.0)
    portfolio._interest_rate_annual = data.get('_interest_rate_annual', 0.05)
    lic = data.get('_last_interest_calc')
    if lic and lic != 'None':
        from datetime import datetime as dt
        portfolio._last_interest_calc = dt.fromisoformat(lic)
    portfolio._peak_equity = data.get('_peak_equity', data['initial_capital'])
    
    # 恢复权益历史
    portfolio._equity_history = [
        (str(item[0]), float(item[1]))
        for item in data.get('_equity_history', [])
    ]
    
    # 恢复当日状态
    ds = data.get('_day_start_date')
    if ds and ds != 'None':
        from datetime import date
        portfolio._day_start_date = date.fromisoformat(ds)
    portfolio._day_start_equity = data.get('_day_start_equity', data['initial_capital'])
    
    # 恢复持仓
    portfolio.positions.clear()
    for ticker, pdict in data.get('positions', {}).items():
        portfolio.positions[ticker] = HoldingPosition(
            ticker=pdict['ticker'],
            quantity=pdict['quantity'],
            avg_cost=pdict['avg_cost'],
            current_price=pdict.get('current_price', pdict['avg_cost']),
            market_value=pdict.get('market_value', 0),
            cost_basis=pdict.get('cost_basis', pdict['quantity'] * pdict['avg_cost']),
            unrealized_pnl=pdict.get('unrealized_pnl', 0),
            unrealized_pnl_pct=pdict.get('unrealized_pnl_pct', 0),
            day_change=pdict.get('day_change', 0),
            day_change_pct=pdict.get('day_change_pct', 0),
            weight=pdict.get('weight', 0),
            last_update=pdict.get('last_update', ''),
        )
    
    # 恢复交易历史
    portfolio.trade_history.clear()
    for td in data.get('trade_history', []):
        portfolio.trade_history.append(TradeRecord(
            trade_id=td['trade_id'],
            ticker=td['ticker'],
            side=td['side'],
            quantity=td['quantity'],
            price=td['price'],
            total_value=td['total_value'],
            commission=td['commission'],
            timestamp=td['timestamp'],
            pnl=td.get('pnl', 0),
            reason=td.get('reason', ''),
        ))


def serialize_accuracy(accuracy) -> Dict[str, Any]:
    """序列化 AccuracyTracker"""
    predictions = {}
    for pid, pred in accuracy.predictions.items():
        predictions[str(pid)] = {
            'id': pred.id,
            'ticker': pred.ticker,
            'timestamp': pred.timestamp,
            'predicted_return': pred.predicted_return,
            'predicted_direction': pred.predicted_direction,
            'confidence': pred.confidence,
            'actual_return': pred.actual_return,
            'actual_direction': pred.actual_direction,
            'is_correct': pred.is_correct,
            'error': pred.error,
            'status': pred.status,
            'confirmed_at': pred.confirmed_at,
        }
    
    # 将 defaultdict 转为普通 dict，value 的 dict 也保持原样
    ticker_stats = {}
    for ticker, stats in accuracy.ticker_stats.items():
        ticker_stats[ticker] = dict(stats)
    
    return {
        'rolling_window': accuracy.rolling_window,
        'predictions': predictions,
        '_id_counter': accuracy._id_counter,
        'accuracy_history': list(accuracy.accuracy_history),
        'ticker_stats': ticker_stats,
        'degradation_threshold': accuracy.degradation_threshold,
    }


def deserialize_accuracy(accuracy, data: Dict[str, Any]) -> None:
    """从序列化数据恢复 AccuracyTracker"""
    from live_trading.accuracy_tracker import PredictionRecord
    
    accuracy.rolling_window = data.get('rolling_window', 50)
    accuracy._id_counter = data.get('_id_counter', 0)
    accuracy.degradation_threshold = data.get('degradation_threshold', 0.05)
    
    # 恢复预测记录
    accuracy.predictions.clear()
    for pid_str, pdict in data.get('predictions', {}).items():
        accuracy.predictions[int(pid_str)] = PredictionRecord(
            id=pdict['id'],
            ticker=pdict['ticker'],
            timestamp=pdict['timestamp'],
            predicted_return=pdict['predicted_return'],
            predicted_direction=pdict['predicted_direction'],
            confidence=pdict['confidence'],
            actual_return=pdict.get('actual_return'),
            actual_direction=pdict.get('actual_direction'),
            is_correct=pdict.get('is_correct'),
            error=pdict.get('error'),
            status=pdict.get('status', 'pending'),
            confirmed_at=pdict.get('confirmed_at'),
        )
    
    # 恢复准确率历史
    accuracy.accuracy_history = deque(
        data.get('accuracy_history', []),
        maxlen=200
    )
    
    # 恢复分股票统计
    accuracy.ticker_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for ticker, stats in data.get('ticker_stats', {}).items():
        accuracy.ticker_stats[ticker] = dict(stats)


def serialize_benchmark(benchmark) -> Dict[str, Any]:
    """序列化 BenchmarkTracker"""
    # 将 pandas Series 转为 list of (timestamp, value)
    nasdaq_curve = []
    if not benchmark.nasdaq_equity_curve.empty:
        for ts, val in benchmark.nasdaq_equity_curve.items():
            nasdaq_curve.append([str(ts), float(val)])
    
    strategy_curve = []
    if not benchmark.strategy_equity_curve.empty:
        for ts, val in benchmark.strategy_equity_curve.items():
            strategy_curve.append([str(ts), float(val)])
    
    # 序列化 nasdaq_returns (日收益率)
    nasdaq_returns = []
    if not benchmark.nasdaq_returns.empty:
        for ts, val in benchmark.nasdaq_returns.items():
            nasdaq_returns.append([str(ts), float(val)])
    
    return {
        'initial_capital': benchmark.initial_capital,
        'current_nasdaq_price': benchmark.current_nasdaq_price,
        'nasdaq_start_price': benchmark.nasdaq_start_price,
        'nasdaq_shares': benchmark.nasdaq_shares,
        'nasdaq_peak': benchmark.nasdaq_peak,
        'strategy_peak': benchmark.strategy_peak,
        'nasdaq_prev_close': benchmark.nasdaq_prev_close,
        '_nasdaq_worst_drawdown': benchmark._nasdaq_worst_drawdown,
        '_strategy_worst_drawdown': benchmark._strategy_worst_drawdown,
        'start_date': str(benchmark.start_date) if benchmark.start_date else None,
        'nasdaq_equity_curve': nasdaq_curve,
        'strategy_equity_curve': strategy_curve,
        'nasdaq_returns': nasdaq_returns,
    }


def deserialize_benchmark(benchmark, data: Dict[str, Any]) -> None:
    """从序列化数据恢复 BenchmarkTracker"""
    import pandas as pd
    
    benchmark.initial_capital = data.get('initial_capital', 100000)
    benchmark.current_nasdaq_price = data.get('current_nasdaq_price', 0)
    benchmark.nasdaq_start_price = data.get('nasdaq_start_price', 0)
    benchmark.nasdaq_shares = data.get('nasdaq_shares', 0)
    benchmark.nasdaq_peak = data.get('nasdaq_peak', 0)
    benchmark.strategy_peak = data.get('strategy_peak', benchmark.initial_capital)
    benchmark.nasdaq_prev_close = data.get('nasdaq_prev_close', 0.0)
    benchmark._nasdaq_worst_drawdown = data.get('_nasdaq_worst_drawdown', 0.0)
    benchmark._strategy_worst_drawdown = data.get('_strategy_worst_drawdown', 0.0)
    
    sd = data.get('start_date')
    if sd and sd != 'None':
        from datetime import date
        benchmark.start_date = date.fromisoformat(sd)
    
    # 恢复权益曲线
    nasdaq_curve = data.get('nasdaq_equity_curve', [])
    if nasdaq_curve:
        benchmark.nasdaq_equity_curve = pd.Series(
            {pd.Timestamp(ts): float(v) for ts, v in nasdaq_curve}
        )
    
    strategy_curve = data.get('strategy_equity_curve', [])
    if strategy_curve:
        benchmark.strategy_equity_curve = pd.Series(
            {pd.Timestamp(ts): float(v) for ts, v in strategy_curve}
        )
    
    # 恢复 dict 备份（避免后续 update 覆盖历史数据）
    if nasdaq_curve:
        benchmark._nasdaq_equity_dict = {
            pd.Timestamp(ts): float(v) for ts, v in nasdaq_curve
        }
    else:
        benchmark._nasdaq_equity_dict = {}
    
    if strategy_curve:
        benchmark._strategy_equity_dict = {
            pd.Timestamp(ts): float(v) for ts, v in strategy_curve
        }
    else:
        benchmark._strategy_equity_dict = {}
    
    # 恢复日收益率序列
    nasdaq_rets = data.get('nasdaq_returns', [])
    if nasdaq_rets:
        benchmark.nasdaq_returns = pd.Series(
            {pd.Timestamp(ts): float(v) for ts, v in nasdaq_rets}
        )


def save_state(
    portfolio,
    accuracy,
    benchmark,
    globals_dict: Dict[str, Any],
    filepath: Path = DEFAULT_STATE_PATH
) -> bool:
    """
    保存完整交易状态到 JSON 文件
    
    参数:
        portfolio: PortfolioManager 实例
        accuracy: AccuracyTracker 实例
        benchmark: BenchmarkTracker 实例
        globals_dict: 包含 _current_prices, _iteration_count,
                      _positions_initialized, _market_opened,
                      _previous_prices 等全局变量
        filepath: 保存路径
    
    返回:
        是否保存成功
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'saved_at': datetime.now().isoformat(),
            'version': 2,
            'portfolio': serialize_portfolio(portfolio),
            'accuracy': serialize_accuracy(accuracy),
            'benchmark': serialize_benchmark(benchmark),
            'globals': {
                'current_prices': dict(globals_dict.get('current_prices', {})),
                'previous_prices': dict(globals_dict.get('previous_prices', {})),
                'iteration_count': globals_dict.get('iteration_count', 0),
                'positions_initialized': globals_dict.get('positions_initialized', False),
                'market_opened': globals_dict.get('market_opened', False),
                'predictor': globals_dict.get('predictor', {}),
                'position_entry_time': globals_dict.get('position_entry_time', {}),
                'position_sell_cooldown': globals_dict.get('position_sell_cooldown', {}),
                'recent_signals': globals_dict.get('recent_signals', []),
                'ml_ready': globals_dict.get('ml_ready', False),
                'prediction_iters': globals_dict.get('prediction_iters', {}),
                'leverage_engine': globals_dict.get('leverage_engine', {}),
            },
        }
        
        # 原子写入：先写临时文件，再重命名
        tmp_path = filepath.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
        tmp_path.replace(filepath)
        
        return True
    except Exception as e:
        logger.error(f"保存状态失败: {e}", exc_info=True)
        return False


def load_state(filepath: Path = DEFAULT_STATE_PATH) -> Optional[Dict[str, Any]]:
    """
    从 JSON 文件加载交易状态
    
    参数:
        filepath: 状态文件路径
    
    返回:
        状态字典，文件不存在或损坏则返回 None
        {
            'portfolio': dict,
            'accuracy': dict,
            'benchmark': dict,
            'globals': dict,
        }
    """
    if not filepath.exists():
        logger.info("未找到状态文件，将使用全新状态")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        version = state.get('version', 1)
        saved_at = state.get('saved_at', '未知')
        logger.info(f"找到状态文件 (v{version}, 保存于 {saved_at})")
        
        return {
            'portfolio': state.get('portfolio', {}),
            'accuracy': state.get('accuracy', {}),
            'benchmark': state.get('benchmark', {}),
            'globals': state.get('globals', {}),
            'predictor': state.get('globals', {}).get('predictor', {}),
            'saved_at': saved_at,
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"状态文件损坏: {e}，将使用全新状态")
        # 备份损坏文件
        corrupt_path = filepath.with_suffix('.corrupt')
        filepath.rename(corrupt_path)
        logger.info(f"损坏文件已备份到 {corrupt_path}")
        return None
    except Exception as e:
        logger.error(f"加载状态失败: {e}")
        return None
