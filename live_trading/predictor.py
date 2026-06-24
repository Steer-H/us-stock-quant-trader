"""Statistical predictor using rolling window statistics."""

import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class RealtimePredictor:
    """
    實時統計預測器
    
    不需要GPU訓練，基於價格序列實時計算信號。
    可以作為Transformer離線訓練的基準參考。
    
    使用示例:
        pred = RealtimePredictor()
        signal, confidence = pred.predict('AAPL', current_price=295.50)
        # signal: 1=漲, 0=跌
    """
    
    def __init__(self, window_size: int = 120, lookback_minutes: int = 30):
        """
        參數:
            window_size: 價格歷史保留長度（秒級tick數）
            lookback_minutes: 回看窗口（分鐘）
        """
        self.window_size = window_size
        self.lookback_minutes = lookback_minutes
        
        # 每隻股票的價格歷史: {ticker: deque(maxlen=window_size)}
        self.price_history: Dict[str, deque] = {}
        
        # 成交量歷史
        self.volume_history: Dict[str, deque] = {}
        
        # 預測統計
        self.total_predictions: int = 0
        self.correct_predictions: int = 0
        self.prediction_history: deque = deque(maxlen=200)
        
        # 因子權重（可通過離線訓練調整）
        self.factor_weights = {
            'momentum': 0.40,       # 動量因子
            'mean_reversion': 0.25,  # 均值回歸
            'volume': 0.15,          # 成交量
            'volatility': 0.20,      # 波動率
        }
        
        # 因子滾動表現追蹤（用於自適應權重調整）
        self.factor_performance: Dict[str, Dict[str, int]] = {
            'momentum': {'correct': 0, 'total': 0},
            'mean_reversion': {'correct': 0, 'total': 0},
            'volume': {'correct': 0, 'total': 0},
            'volatility': {'correct': 0, 'total': 0},
        }
        self._factor_signal_cache: Dict[str, float] = {}  # 緩存最近信號用於學習
        self._regime: str = 'trending'  # 市場狀態: trending / mean_reverting / volatile
        self._regime_update_counter: int = 0
        
        logger.info(f"統計預測器初始化: window={window_size}, lookback={lookback_minutes}min")
    
    def update_price(self, ticker: str, price: float, volume: Optional[float] = None):
        """
        更新價格歷史
        
        每秒tick調用一次，追加最新價格。
        """
        if ticker not in self.price_history:
            self.price_history[ticker] = deque(maxlen=self.window_size)
            self.volume_history[ticker] = deque(maxlen=self.window_size)
        
        self.price_history[ticker].append(price)
        if volume is not None:
            self.volume_history[ticker].append(volume)
    
    def predict(self, ticker: str, current_price: float) -> Tuple[int, float, Dict]:
        """
        生成預測信號
        
        參數:
            ticker: 股票代碼
            current_price: 當前價格
        
        返回:
            (direction, confidence, factors_detail)
            direction: 1=漲, 0=跌
            confidence: 0-1置信度
            factors_detail: 各因子詳細得分
        """
        prices = self.price_history.get(ticker, deque(maxlen=self.window_size))
        
        # 數據不足時返回中立/微偏多信號(美股長期向上)
        if len(prices) < 10:
            return (1, 0.52, {'note': '數據不足,默認微偏多'})
        
        price_array = np.array(list(prices))
        current = current_price
        

        # 更新市場狀態
        self._update_regime(ticker)
        
        # ── 因子1: 短期動量 (最近5/10/20個tick的趨勢) ──
        momentum_score = self._calc_momentum(price_array)
        self._factor_signal_cache['momentum'] = momentum_score
        
        # ── 因子2: 均值回歸 (價格偏離均線的程度) ──
        mr_score = self._calc_mean_reversion(price_array, current)
        self._factor_signal_cache['mean_reversion'] = mr_score
        
        # ── 因子3: 成交量異常 ──
        volume_score = self._calc_volume_signal(ticker, price_array)
        self._factor_signal_cache['volume'] = volume_score
        
        # ── 因子4: 波動率 ──
        volatility_score = self._calc_volatility(price_array)
        self._factor_signal_cache['volatility'] = volatility_score
        
        # ── 綜合評分 ──
        composite = (
            self.factor_weights['momentum'] * momentum_score +
            self.factor_weights['mean_reversion'] * mr_score +
            self.factor_weights['volume'] * volume_score +
            self.factor_weights['volatility'] * volatility_score
        )
        
        # 方向判斷: 綜合分 > 0 → 漲, ≤ 0 → 跌
        direction = 1 if composite > 0 else 0
        
        # 置信度: 基於信號強度
        raw_confidence = 0.5 + abs(composite) * 0.5
        confidence = min(max(raw_confidence, 0.50), 0.90)
        
        # 記錄預測
        self.total_predictions += 1
        self.prediction_history.append({
            'ticker': ticker,
            'direction': direction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
        })
        
        factors_detail = {
            'momentum': round(momentum_score, 4),
            'mean_reversion': round(mr_score, 4),
            'volume': round(volume_score, 4),
            'volatility': round(volatility_score, 4),
            'composite': round(composite, 4),
        }
        
        return (direction, round(confidence, 4), factors_detail)
    
    def _calc_momentum(self, prices: np.ndarray) -> float:
        """
        短期動量因子
        
        計算最近多個窗口的收益率加權平均。
        正值=上漲趨勢，負值=下跌趨勢。
        
        返回: -1.0 ~ +1.0 歸一化得分
        """
        if len(prices) < 5:
            return 0.0
        
        n = len(prices)
        
        # 多個窗口的收益率
        windows = []
        for w in [3, 5, 10, 20]:
            if n > w:
                ret = (prices[-1] - prices[-w]) / prices[-w]
                windows.append(ret)
        
        if not windows:
            return 0.0
        
        # 近期窗口權重更高
        weights = [0.4, 0.3, 0.2, 0.1][:len(windows)]
        weights = np.array(weights) / sum(weights)
        
        weighted_ret = np.dot(windows, weights)
        
        # Tanh歸一化到[-1, 1]
        return float(np.tanh(weighted_ret * 20))
    
    def _calc_mean_reversion(self, prices: np.ndarray, current: float) -> float:
        """
        均值回歸因子
        
        價格偏離移動均線越遠，回歸概率越大。
        正值=超賣(看漲回歸)，負值=超買(看跌回歸)。
        
        返回: -1.0 ~ +1.0 歸一化得分
        """
        if len(prices) < 10:
            return 0.0
        
        # 計算多個周期的均線
        mas = {}
        for period in [10, 30, 60]:
            if len(prices) >= period:
                mas[period] = np.mean(prices[-period:])
        
        if not mas:
            return 0.0
        
        # 價格偏離度的加權平均
        deviations = []
        weights = []
        for period, ma in mas.items():
            dev = (current - ma) / ma  # 正=高於均線, 負=低於均線
            deviations.append(dev)
            weights.append(1.0 / period)  # 短期均線權重更高
        
        norm_weights = np.array(weights) / sum(weights)
        avg_dev = np.dot(deviations, norm_weights)
        
        # 符號取反: 高於均線→賣出信號(負), 低於均線→買入信號(正)
        return float(-np.tanh(avg_dev * 30))
    
    def _calc_volume_signal(self, ticker: str, prices: np.ndarray) -> float:
        """
        成交量異常檢測
        
        成交量突增往往預示趨勢變化。
        
        返回: -1.0 ~ +1.0
        """
        volumes = self.volume_history.get(ticker, deque(maxlen=self.window_size))
        if len(volumes) < 20:
            return 0.0
        
        vol_array = np.array(list(volumes))
        recent_vol = np.mean(vol_array[-5:])
        historical_vol = np.mean(vol_array[:-5]) if len(vol_array) > 5 else recent_vol
        
        if historical_vol <= 0:
            return 0.0
        
        vol_ratio = recent_vol / historical_vol
        
        if vol_ratio > 1.5:
            # 成交量放量: 跟隨趨勢
            if len(prices) >= 5:
                trend = (prices[-1] - prices[-5]) / prices[-5]
                return float(np.tanh(trend * 30))
            return 0.1  # 微偏多
        elif vol_ratio < 0.5:
            # 成交量萎縮: 趨勢減弱
            return -0.1
        
        return 0.0
    
    def _calc_volatility(self, prices: np.ndarray) -> float:
        """
        波動率因子
        
        高波動率→風險增加→偏空
        低波動率→穩定→偏多
        
        返回: -1.0 ~ +1.0
        """
        if len(prices) < 10:
            return 0.0
        
        # 計算收益率標準差（波動率）
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < 2:
            return 0.0
        
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
        
        if len(returns) >= 30:
            historical_vol = np.std(returns)
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        else:
            vol_ratio = 1.0
        
        # 波動率升高→負信號; 降低→正信號
        return float(-np.tanh((vol_ratio - 1.0) * 3))
    
    def get_accuracy(self) -> float:
        """獲取預測準確率"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
    
    def confirm(self, ticker: str, actual_direction: int) -> None:
        """
        確認最近一次對ticker的預測結果
        
        參數:
            ticker: 股票代碼
            actual_direction: 實際方向 (1=漲, 0=跌)
        """
        # 找最近一次對該ticker的未確認預測
        for i in range(len(self.prediction_history) - 1, -1, -1):
            pred = self.prediction_history[i]
            if pred['ticker'] == ticker and pred.get('confirmed') is None:
                pred['confirmed'] = True
                pred['actual_direction'] = actual_direction
                pred['is_correct'] = (pred['direction'] == actual_direction)
                if pred['is_correct']:
                    self.correct_predictions += 1
                
                # 更新各因子表現
                if hasattr(self, '_factor_signal_cache') and self._factor_signal_cache:
                    for factor, signal in self._factor_signal_cache.items():
                        if factor in self.factor_performance:
                            self.factor_performance[factor]['total'] += 1
                            factor_correct = (signal > 0 and actual_direction == 1) or (signal <= 0 and actual_direction == 0)
                            if factor_correct:
                                self.factor_performance[factor]['correct'] += 1
                break
    
    def adjust_weights(self, backtest_results: Dict[str, float]):
        """
        根據回測結果自動調整因子權重
        
        參數:
            backtest_results: {'momentum': accuracy, 'mean_reversion': accuracy, ...}
        """
        total_acc = sum(backtest_results.values())
        if total_acc <= 0:
            return
        
        for factor in self.factor_weights:
            if factor in backtest_results:
                self.factor_weights[factor] = backtest_results[factor] / total_acc
        
        logger.info(f"因子權重已調整: {self.factor_weights}")
    
    def to_dict(self) -> Dict:
        """序列化為字典（用於狀態持久化）"""
        price_hist = {}
        for ticker, dq in self.price_history.items():
            price_hist[ticker] = list(dq)
        volume_hist = {}
        for ticker, dq in self.volume_history.items():
            volume_hist[ticker] = list(dq)
        
        return {
            'window_size': self.window_size,
            'lookback_minutes': self.lookback_minutes,
            'price_history': price_hist,
            'volume_history': volume_hist,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'factor_weights': self.factor_weights,
            'factor_performance': self.factor_performance,
            'factor_signal_cache': dict(self._factor_signal_cache),
            'regime': self._regime,
            'regime_update_counter': self._regime_update_counter,
            'prediction_history': list(self.prediction_history),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RealtimePredictor':
        """從字典恢復"""
        pred = cls(
            window_size=data.get('window_size', 120),
            lookback_minutes=data.get('lookback_minutes', 30),
        )
        for ticker, prices in data.get('price_history', {}).items():
            pred.price_history[ticker] = deque(prices, maxlen=pred.window_size)
        for ticker, vols in data.get('volume_history', {}).items():
            pred.volume_history[ticker] = deque(vols, maxlen=pred.window_size)
        pred.total_predictions = data.get('total_predictions', 0)
        pred.correct_predictions = data.get('correct_predictions', 0)
        pred.factor_weights.update(data.get('factor_weights', {}))
        if 'factor_performance' in data:
            pred.factor_performance.update(data['factor_performance'])
        if 'factor_signal_cache' in data:
            pred._factor_signal_cache.update(data['factor_signal_cache'])
        if 'regime' in data:
            pred._regime = data['regime']
        if 'regime_update_counter' in data:
            pred._regime_update_counter = data['regime_update_counter']
        if 'prediction_history' in data:
            for entry in data['prediction_history']:
                pred.prediction_history.append(entry)
        return pred


def train_offline_transformer():
    """
    離線訓練Transformer模型（休市時調用）
    
    使用已處理的parquet數據訓練，保存模型到data/models/。
    
    返回:
        方向準確率 或 None
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from config.settings import model_config
    from ml_model.trainer import ModelTrainer
    from ml_model.data_loader import prepare_data
    
    logger.info("🔄 開始離線訓練 Transformer...")
    
    try:
        # 加載所有已處理股票的特徵數據
        train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)
        
        if train_loader is None:
            logger.warning("無訓練數據可用，跳過訓練")
            return None
        
        # 訓練
        trainer = ModelTrainer(model_config)
        trainer.train(train_loader, val_loader, epochs=30)
        
        # 評估
        result = trainer.evaluate(test_loader)
        acc = result.direction_accuracy
        
        # 保存模型
        trainer.save_model('transformer_stock_latest')
        logger.info(f"✅ Transformer訓練完成，方向準確率: {acc:.2%}")
        
        return acc
        
    except Exception as e:
        logger.error(f"離線訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return None
