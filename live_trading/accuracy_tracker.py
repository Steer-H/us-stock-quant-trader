"""Track prediction directional accuracy over a rolling window."""

import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from collections import deque, defaultdict

import numpy as np
import pandas as pd

from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


# ============================================================================
# 數據結構
# ============================================================================
@dataclass
class PredictionRecord:
    """單次預測記錄"""
    id: int
    ticker: str
    timestamp: str
    
    # 預測
    predicted_return: float     # 預測收益率
    predicted_direction: int    # 預測方向 (1=漲, 0=跌)
    confidence: float           # 預測置信度
    
    # 實際結果（在結果確認後填寫）
    actual_return: Optional[float] = None
    actual_direction: Optional[int] = None
    is_correct: Optional[bool] = None  # 方向是否正確
    error: Optional[float] = None       # 預測誤差
    
    # 狀態
    status: str = 'pending'     # pending, confirmed, expired
    confirmed_at: Optional[str] = None


@dataclass
class AccuracySnapshot:
    """準確率快照"""
    timestamp: str = ''
    
    # 總體指標
    total_predictions: int = 0
    confirmed_predictions: int = 0
    direction_accuracy: float = 0.0      # 方向準確率
    recent_accuracy_50: float = 0.0      # 最近50次準確率
    rmse: float = 0.0                    # 均方根誤差
    mae: float = 0.0                     # 平均絕對誤差
    
    # 分類統計
    correct_long: int = 0   # 正確預測上漲
    correct_short: int = 0  # 正確預測下跌
    total_long: int = 0
    total_short: int = 0
    
    # 滾動指標
    accuracy_trend: str = 'stable'  # stable, improving, degrading
    
    # 按股票
    per_ticker_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # 模型狀態
    is_acceptable: bool = True  # 是否可接受（方向準確率 >= 55%）


class AccuracyTracker:
    """
    模型預測準確率追蹤器
    
    記錄每次預測並持續追蹤準確率。
    
    功能：
    - 記錄預測和實際結果
    - 計算方向準確率和RMSE
    - 滾動窗口準確率（檢測性能退化）
    - 模型漂移預警
    
    使用示例:
        tracker = AccuracyTracker()
        pred_id = tracker.record_prediction('AAPL', 0.012, 1, 0.85)
        # ... 確認結果後 ...
        tracker.confirm_prediction(pred_id, 0.015, 1)
        snapshot = tracker.get_snapshot()
    """
    
    def __init__(self, rolling_window: int = 50):
        """
        參數:
            rolling_window: 滾動窗口大小（用於檢測準確率趨勢）
        """
        self.rolling_window = rolling_window
        self.predictions: Dict[int, PredictionRecord] = {}
        self._id_counter: int = 0
        
        # 歷史準確率序列
        self.accuracy_history: deque = deque(maxlen=200)
        
        # 按股票統計
        self.ticker_stats: Dict[str, Dict] = defaultdict(
            lambda: {'correct': 0, 'total': 0}
        )
        
        # 性能退化檢測參數
        self.degradation_threshold: float = 0.05  # 準確率下降5%視為退化
        
        logger.info(f"準確率追蹤器初始化, 滾動窗口: {rolling_window}")
    
    def record_prediction(
        self,
        ticker: str,
        predicted_return: float,
        predicted_direction: int,
        confidence: float = 0.5
    ) -> int:
        """
        記錄一次預測
        
        參數:
            ticker: 股票代碼
            predicted_return: 預測收益率
            predicted_direction: 預測方向 (1=漲, 0=跌)
            confidence: 預測置信度
        
        返回:
            預測記錄ID
        """
        self._id_counter += 1
        
        record = PredictionRecord(
            id=self._id_counter,
            ticker=ticker.upper(),
            timestamp=datetime.now().isoformat(),
            predicted_return=predicted_return,
            predicted_direction=predicted_direction,
            confidence=confidence,
            status='pending'
        )
        
        self.predictions[self._id_counter] = record
        
        return self._id_counter
    
    def confirm_prediction(
        self,
        prediction_id: int,
        actual_return: float,
        actual_direction: int
    ) -> bool:
        """
        確認預測結果
        
        當真實結果出來後，更新預測記錄並重新計算準確率。
        
        參數:
            prediction_id: 預測記錄ID
            actual_return: 實際收益率
            actual_direction: 實際方向 (1=漲, 0=跌)
        
        返回:
            是否確認成功
        """
        record = self.predictions.get(prediction_id)
        if not record:
            logger.warning(f"未找到預測記錄: {prediction_id}")
            return False
        
        # 更新記錄
        record.actual_return = actual_return
        record.actual_direction = actual_direction
        record.is_correct = (record.predicted_direction == actual_direction)
        record.error = abs(record.predicted_return - actual_return)
        record.status = 'confirmed'
        record.confirmed_at = datetime.now().isoformat()
        
        # 更新統計
        self.accuracy_history.append(1 if record.is_correct else 0)
        
        ticker = record.ticker
        self.ticker_stats[ticker]['total'] += 1
        if record.is_correct:
            self.ticker_stats[ticker]['correct'] += 1
        
        logger.debug(
            f"預測確認 #{prediction_id}: {ticker} "
            f"預測{'漲' if record.predicted_direction else '跌'}, "
            f"實際{'漲' if actual_direction else '跌'}, "
            f"{'✓' if record.is_correct else '✗'}"
        )
        
        return True
    
    def get_direction_accuracy(self) -> float:
        """
        獲取總體方向準確率
        
        返回:
            準確率 (0.0 ~ 1.0)
        """
        confirmed = [r for r in self.predictions.values() if r.status == 'confirmed']
        if not confirmed:
            return 0.0
        
        correct = sum(1 for r in confirmed if r.is_correct)
        return correct / len(confirmed)
    
    def get_recent_accuracy(self, n: int = 50) -> float:
        """
        獲取最近N次預測的準確率
        
        用於檢測模型性能是否在退化。
        
        參數:
            n: 最近N次
        
        返回:
            準確率
        """
        recent = list(self.accuracy_history)[-n:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)
    
    def get_rmse(self) -> float:
        """
        獲取預測收益率的RMSE
        
        返回:
            均方根誤差
        """
        confirmed = [
            r for r in self.predictions.values() 
            if r.status == 'confirmed' and r.error is not None
        ]
        if not confirmed:
            return 0.0
        
        squared_errors = [r.error ** 2 for r in confirmed]
        return float(np.sqrt(np.mean(squared_errors)))
    
    def get_mae(self) -> float:
        """獲取平均絕對誤差"""
        confirmed = [
            r for r in self.predictions.values() 
            if r.status == 'confirmed' and r.error is not None
        ]
        if not confirmed:
            return 0.0
        
        errors = [r.error for r in confirmed]
        return float(np.mean(errors))
    
    def check_degradation(self) -> Tuple[bool, str]:
        """
        檢查模型性能是否退化
        
        比較最近窗口和歷史準確率，檢測下降趨勢。
        
        返回:
            (是否退化, 描述)
        """
        if len(self.accuracy_history) < self.rolling_window * 2:
            return False, '數據不足，無法檢測退化'
        
        # 取兩段窗口的準確率
        recent = list(self.accuracy_history)[-self.rolling_window:]
        older = list(self.accuracy_history)[-2*self.rolling_window:-self.rolling_window]
        
        recent_acc = sum(recent) / len(recent) if recent else 0
        older_acc = sum(older) / len(older) if older else 0
        
        diff = recent_acc - older_acc
        
        if diff < -self.degradation_threshold:
            return True, (
                f"⚠️ 模型性能退化: 最近{self.rolling_window}次準確率 {recent_acc:.1%}, "
                f"之前{self.rolling_window}次 {older_acc:.1%}, 下降 {abs(diff):.1%}"
            )
        
        return False, f'模型性能穩定 (變化 {diff:+.1%})'
    
    def get_snapshot(self) -> AccuracySnapshot:
        """
        獲取準確率快照
        
        返回:
            AccuracySnapshot對象
        """
        confirmed = [r for r in self.predictions.values() if r.status == 'confirmed']
        pending = [r for r in self.predictions.values() if r.status == 'pending']
        
        snap = AccuracySnapshot(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_predictions=len(self.predictions),
            confirmed_predictions=len(confirmed),
        )
        
        if confirmed:
            snap.direction_accuracy = self.get_direction_accuracy()
            snap.recent_accuracy_50 = self.get_recent_accuracy(50)
            snap.rmse = self.get_rmse()
            snap.mae = self.get_mae()
            
            # 分類統計
            for r in confirmed:
                if r.predicted_direction == 1:
                    snap.total_long += 1
                    if r.is_correct:
                        snap.correct_long += 1
                else:
                    snap.total_short += 1
                    if r.is_correct:
                        snap.correct_short += 1
            
            # 是否可接受
            snap.is_acceptable = snap.direction_accuracy >= 0.55
        
        # 按股票準確率
        for ticker, stats in self.ticker_stats.items():
            if stats['total'] > 0:
                snap.per_ticker_accuracy[ticker] = stats['correct'] / stats['total']
        
        # 趨勢判斷
        _, trend = self.check_degradation()
        snap.accuracy_trend = 'degrading' if '退化' in trend else 'stable'
        
        return snap
    
    def get_recent_predictions(self, n: int = 10) -> pd.DataFrame:
        """
        獲取最近N次預測記錄
        
        參數:
            n: 記錄數量
        
        返回:
            DataFrame
        """
        recent_ids = sorted(self.predictions.keys(), reverse=True)[:n]
        
        records = []
        for pid in recent_ids:
            r = self.predictions[pid]
            records.append({
                'ID': r.id,
                'Ticker': r.ticker,
                '預測收益率': f"{r.predicted_return:+.4%}",
                '預測方向': '📈漲' if r.predicted_direction else '📉跌',
                '置信度': f"{r.confidence:.1%}",
                '實際方向': '📈漲' if r.actual_direction else '📉跌' if r.actual_direction is not None else '⏳待定',
                '結果': '✓正確' if r.is_correct else '✗錯誤' if r.is_correct is not None else '⏳',
                '時間': r.timestamp[:19],
            })
        
        return pd.DataFrame(records)
    
    def get_accuracy_summary(self) -> Dict[str, str]:
        """
        獲取準確率摘要（格式化字符串）
        
        返回:
            {指標: 格式化值}
        """
        snap = self.get_snapshot()
        
        status_icon = '✅' if snap.is_acceptable else '⚠️'
        
        return {
            '總預測次數': str(snap.total_predictions),
            '已確認次數': str(snap.confirmed_predictions),
            '方向準確率': f"{snap.direction_accuracy:.1%}",
            '最近50次': f"{snap.recent_accuracy_50:.1%}",
            '漲預測正確': f"{snap.correct_long}/{snap.total_long}",
            '跌預測正確': f"{snap.correct_short}/{snap.total_short}",
            'RMSE': f"{snap.rmse:.6f}",
            'MAE': f"{snap.mae:.6f}",
            '準確率趨勢': snap.accuracy_trend,
            '模型狀態': f"{status_icon} {'可接受' if snap.is_acceptable else '需調優'}",
        }
