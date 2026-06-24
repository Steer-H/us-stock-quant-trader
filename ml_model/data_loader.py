"""Data loader for training the Transformer model."""

import logging
from typing import Optional, List, Tuple
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config.settings import ModelConfig, PROCESSED_DATA_DIR
from data_pipeline.storage import ParquetStorage
from utils.exceptions import DataError

logger = logging.getLogger(__name__)


# ============================================================================
# 時間序列數據集
# ============================================================================
class TimeSeriesDataset(Dataset):
    """
    PyTorch時間序列數據集
    
    從歷史價格數據中生成 (輸入窗口, 未來收益率) 的監督學習樣本。
    
    樣本生成邏輯：
    對於長度為T的時間序列，使用大小為W的滑動窗口：
    - 輸入X: [t-W, t-W+1, ..., t-1] 的特徵（W天，n_features維）
    - 標籤y: [t, t+1, ..., t+H-1] 的收益率（H天）
    
    如果不重疊（stride=W），樣本數為 floor(T / W) - 1
    如果重疊（stride<S），樣本數為 T - W - H + 1
    
    時間複雜度: O(n_samples * window * features) - 切片時
    空間複雜度: O(n_samples * window * features) - 存儲切片
    
    參數:
        df: 包含特徵和技術指標的DataFrame
        feature_cols: 要使用的特徵列名列表
        lookback_window: 輸入窗口大小（天數）
        prediction_horizon: 預測窗口大小（天數）
        stride: 滑動窗口步長（1=最大重疊，window=無重疊）
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        lookback_window: int = 60,
        prediction_horizon: int = 5,
        stride: int = 1,
        scaler: Optional[object] = None,
        fit_scaler: bool = True
    ):
        """
        參數:
            df: 包含特徵列的DataFrame（索引為日期）
            feature_cols: 特徵列名列表
            lookback_window: 歷史窗口大小
            prediction_horizon: 預測窗口大小
            stride: 滑動步長
            scaler: 預訓練的scaler（為None時自動創建）
            fit_scaler: 是否在此數據集上擬合scaler
        """
        super().__init__()
        
        self.lookback = lookback_window
        self.horizon = prediction_horizon
        self.feature_cols = feature_cols
        
        # 1. 提取特徵矩陣
        # 確保所有特徵列存在
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise DataError(
                f"DataFrame缺少特徵列: {missing}",
                {'missing': list(missing), 'available': list(df.columns)}
            )
        
        # 丟棄含NaN的行（在序列開頭由於滾動窗口計算產生）
        self.data = df[feature_cols]
        self.data = self.data.dropna()
        
        if len(self.data) < lookback_window + prediction_horizon:
            raise DataError(
                f"數據長度({len(self.data)})不足，需要至少{lookback_window + prediction_horizon}行",
                {'length': len(self.data), 'required': lookback_window + prediction_horizon}
            )
        
        # 2. 特徵標準化
        if scaler is None:
            scaler = StandardScaler()
        
        if fit_scaler:
            self.scaler = scaler.fit(self.data.values)
        else:
            self.scaler = scaler
        
        self.scaled_data = self.scaler.transform(self.data.values)
        
        # 3. 提取目標變量（未來收益率）
        # 這裡假設 'target_1d' 和 'target_5d' 在最後一次 add_all_indicators 時已計算
        # 注意：這些目標列不參與特徵標準化
        self.returns = pd.DataFrame(index=self.data.index)
        
        # 嘗試從原始df中獲取目標列
        # 注意: 多股票合併後索引可能有重複，使用整數位置對齊
        # 記錄哪些行被保留（dropna後剩餘的行）
        keep_mask = df[feature_cols].notna().all(axis=1)
        for col in ['close', 'target_1d', 'target_5d', 'returns_1d', 'returns_5d']:
            if col in df.columns:
                self.returns[col] = df.loc[keep_mask, col].values
        
        if 'target_1d' not in self.returns.columns and 'close' in self.returns.columns:
            # 手動計算目標
            self.returns['target_1d'] = self.returns['close'].pct_change().shift(-1)
        
        self.returns = self.returns.dropna()
        self.returns = self.returns.values
        
        # 4. 計算樣本數量
        min_len = min(len(self.scaled_data), len(self.returns))
        self.stride = stride
        self.n_samples = max(0, (min_len - lookback_window - prediction_horizon) // stride + 1)
        
        logger.debug(
            f"TimeSeriesDataset: {self.n_samples} 樣本, "
            f"window={lookback_window}, horizon={prediction_horizon}, "
            f"features={len(feature_cols)}"
        )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        獲取單個訓練樣本
        
        參數:
            idx: 樣本索引
        
        返回:
            x: 輸入特徵 (lookback_window, n_features)
            y_reg: 回歸目標 (prediction_horizon,) - 未來收益率
            y_cls: 分類目標 (prediction_horizon,) - 漲跌方向(0/1)
        """
        start = idx * self.stride
        end = start + self.lookback
        
        # 輸入特徵窗口
        x = torch.FloatTensor(self.scaled_data[start:end])
        
        # 目標窗口
        target_start = end
        target_end = target_start + self.horizon
        
        # 目標列取最後一列（對應 returns_{horizon}d，由 build_returns 保證列順序）
        y_reg = torch.FloatTensor(self.returns[target_start:target_end, -1])
        
        # 分類標籤：收益率>0為1（漲），否則為0（跌）
        y_cls = (y_reg > 0).float()
        
        return x, y_reg, y_cls


# ============================================================================
# 數據準備函數
# ============================================================================
def prepare_data(
    tickers: Optional[List[str]] = None,
    config: Optional[ModelConfig] = None,
    data_dir: Path = PROCESSED_DATA_DIR,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    準備訓練/驗證/測試數據
    
    將多隻股票的加工數據合併，創建標準化的DataLoader。
    
    重要：時序數據不能隨機shuffle訓練集！
    我們按時間順序劃分每個股票的數據，然後合併。
    
    劃分策略（時序感知）：
    - 每隻股票自己的數據按時間順序劃分為 train/val/test
    - train: 前70%（最早的時間段）
    - val: 中間15%
    - test: 最後15%（最接近當前的時間段）
    
    這種劃分方式模擬了真實的交易場景：
    用歷史數據訓練，用未來數據測試。
    
    時間複雜度: O(t*n)，t=股票數，n=平均每隻股票的數據行數
    空間複雜度: O(t*n)
    
    參數:
        tickers: 股票代碼列表（None則加載所有可用的特徵數據）
        config: 模型配置
        data_dir: 加工數據目錄
        test_size: 測試集比例
        val_size: 驗證集比例
        random_state: 隨機種子
    
    返回:
        (train_loader, val_loader, test_loader, scaler)
    """
    if config is None:
        config = ModelConfig()
    
    storage = ParquetStorage(data_dir)
    
    # 1. 確定股票列表
    if tickers is None:
        available_keys = storage.list_keys()
        tickers = list(set(
            k.replace('_features', '') 
            for k in available_keys 
            if k.endswith('_features')
        ))
    
    if not tickers:
        raise DataError(f"在 {data_dir} 中未找到特徵數據")
    
    logger.info(f"準備數據: {len(tickers)} 只股票")
    
    # 2. 加載並合併所有股票的數據
    all_train_features = []
    all_val_features = []
    all_test_features = []
    
    scaler = StandardScaler()
    
    for ticker in tickers:
        try:
            df = storage.load(f"{ticker}_features")
            
            if df is None or len(df) < config.lookback_window + config.prediction_horizon + 50:
                logger.debug(f"跳過 {ticker}: 數據量不足 ({len(df) if df is not None else 0} 行)")
                continue
            
            # 按時間順序劃分
            n = len(df)
            train_end = int(n * (1 - val_size - test_size))
            val_end = int(n * (1 - test_size))
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # 收集特徵數據用於擬合scaler（所有股票參與partial_fit）
            if len(train_df) > 0:
                train_features = train_df[config.features].dropna().values
                if len(train_features) > 0:
                    scaler.partial_fit(train_features)
            
            all_train_features.append(train_df)
            all_val_features.append(val_df)
            all_test_features.append(test_df)
            
        except Exception as e:
            logger.warning(f"加載 {ticker} 數據失敗: {e}")
            continue
    
    if not all_train_features:
        raise DataError("沒有可用的訓練數據")
    
    # 3. 合併數據
    train_combined = pd.concat(all_train_features)
    val_combined = pd.concat(all_val_features) if all_val_features else pd.DataFrame()
    test_combined = pd.concat(all_test_features) if all_test_features else pd.DataFrame()
    
    logger.info(
        f"數據劃分: train={len(train_combined)}, "
        f"val={len(val_combined)}, test={len(test_combined)}"
    )
    
    # 4. 創建Dataset
    train_dataset = TimeSeriesDataset(
        train_combined, config.features,
        config.lookback_window, config.prediction_horizon,
        scaler=scaler, fit_scaler=False  # 已經partial_fit了
    )
    
    val_dataset = TimeSeriesDataset(
        val_combined, config.features,
        config.lookback_window, config.prediction_horizon,
        scaler=scaler, fit_scaler=False
    ) if not val_combined.empty else None
    
    test_dataset = TimeSeriesDataset(
        test_combined, config.features,
        config.lookback_window, config.prediction_horizon,
        scaler=scaler, fit_scaler=False
    ) if not test_combined.empty else None
    
    # 5. 創建DataLoader
    # 注意：對於時序數據，shuffle=True會破壞時間依賴關係
    # 但在訓練集中輕微shuffle（batch級別）有助於隨機梯度下降
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 對訓練集shuffle batch順序
        num_workers=0,  # 避免多進程問題
        drop_last=True  # 丟棄最後不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    ) if val_dataset is not None else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    ) if test_dataset is not None else None
    
    logger.info(
        f"DataLoader已創建: "
        f"train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader) if val_loader else 0}, "
        f"test_batches={len(test_loader) if test_loader else 0}"
    )
    
    return train_loader, val_loader, test_loader, scaler


def load_single_ticker_data(
    ticker: str,
    config: ModelConfig,
    data_dir: Path = PROCESSED_DATA_DIR
) -> Tuple[DataLoader, object, pd.DataFrame]:
    """
    加載單只股票的數據用於預測
    
    參數:
        ticker: 股票代碼
        config: 模型配置
        data_dir: 數據目錄
    
    返回:
        (data_loader, scaler, original_df)
    """
    storage = ParquetStorage(data_dir)
    
    try:
        df = storage.load(f"{ticker}_features")
    except Exception:
        raise DataError(f"未找到 {ticker} 的特徵數據")
    
    if len(df) < config.lookback_window:
        raise DataError(f"{ticker} 數據量({len(df)})不足窗口大小({config.lookback_window})")
    
    # 創建scaler
    scaler = StandardScaler()
    features = df[config.features].dropna().values
    scaler.fit(features)
    
    dataset = TimeSeriesDataset(
        df, config.features,
        config.lookback_window, config.prediction_horizon,
        scaler=scaler, fit_scaler=False
    )
    
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    
    return loader, scaler, df
