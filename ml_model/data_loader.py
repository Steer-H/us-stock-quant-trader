"""
美股量化交易系统 - ML模型数据加载模块

负责将清洗后的行情数据转换为PyTorch模型可用的训练数据，
包括：
- 时间序列窗口切片
- 特征标准化/归一化
- 训练/验证/测试集划分
- DataLoader创建

数据流：
原始行情数据 → 技术指标计算 → 窗口切片 → 标准化 → DataLoader → 模型训练

关键设计：
- 时序划分：严格按时间顺序划分（不能随机shuffle），防止未来信息泄露
- 窗口重叠：支持滑动窗口重叠，增加样本量
- 多股票联合：支持多只股票的数据混合训练
"""

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
# 时间序列数据集
# ============================================================================
class TimeSeriesDataset(Dataset):
    """
    PyTorch时间序列数据集
    
    从历史价格数据中生成 (输入窗口, 未来收益率) 的监督学习样本。
    
    样本生成逻辑：
    对于长度为T的时间序列，使用大小为W的滑动窗口：
    - 输入X: [t-W, t-W+1, ..., t-1] 的特征（W天，n_features维）
    - 标签y: [t, t+1, ..., t+H-1] 的收益率（H天）
    
    如果不重叠（stride=W），样本数为 floor(T / W) - 1
    如果重叠（stride<S），样本数为 T - W - H + 1
    
    时间复杂度: O(n_samples * window * features) - 切片时
    空间复杂度: O(n_samples * window * features) - 存储切片
    
    参数:
        df: 包含特征和技术指标的DataFrame
        feature_cols: 要使用的特征列名列表
        lookback_window: 输入窗口大小（天数）
        prediction_horizon: 预测窗口大小（天数）
        stride: 滑动窗口步长（1=最大重叠，window=无重叠）
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
        参数:
            df: 包含特征列的DataFrame（索引为日期）
            feature_cols: 特征列名列表
            lookback_window: 历史窗口大小
            prediction_horizon: 预测窗口大小
            stride: 滑动步长
            scaler: 预训练的scaler（为None时自动创建）
            fit_scaler: 是否在此数据集上拟合scaler
        """
        super().__init__()
        
        self.lookback = lookback_window
        self.horizon = prediction_horizon
        self.feature_cols = feature_cols
        
        # 1. 提取特征矩阵
        # 确保所有特征列存在
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise DataError(
                f"DataFrame缺少特征列: {missing}",
                {'missing': list(missing), 'available': list(df.columns)}
            )
        
        # 丢弃含NaN的行（在序列开头由于滚动窗口计算产生）
        self.data = df[feature_cols]
        self.data = self.data.dropna()
        
        if len(self.data) < lookback_window + prediction_horizon:
            raise DataError(
                f"数据长度({len(self.data)})不足，需要至少{lookback_window + prediction_horizon}行",
                {'length': len(self.data), 'required': lookback_window + prediction_horizon}
            )
        
        # 2. 特征标准化
        if scaler is None:
            scaler = StandardScaler()
        
        if fit_scaler:
            self.scaler = scaler.fit(self.data.values)
        else:
            self.scaler = scaler
        
        self.scaled_data = self.scaler.transform(self.data.values)
        
        # 3. 提取目标变量（未来收益率）
        # 这里假设 'target_1d' 和 'target_5d' 在最后一次 add_all_indicators 时已计算
        # 注意：这些目标列不参与特征标准化
        self.returns = pd.DataFrame(index=self.data.index)
        
        # 尝试从原始df中获取目标列
        # 注意: 多股票合并后索引可能有重复，使用整数位置对齐
        # 记录哪些行被保留（dropna后剩余的行）
        keep_mask = df[feature_cols].notna().all(axis=1)
        for col in ['close', 'target_1d', 'target_5d', 'returns_1d', 'returns_5d']:
            if col in df.columns:
                self.returns[col] = df.loc[keep_mask, col].values
        
        if 'target_1d' not in self.returns.columns and 'close' in self.returns.columns:
            # 手动计算目标
            self.returns['target_1d'] = self.returns['close'].pct_change().shift(-1)
        
        self.returns = self.returns.dropna()
        self.returns = self.returns.values
        
        # 4. 计算样本数量
        min_len = min(len(self.scaled_data), len(self.returns))
        self.stride = stride
        self.n_samples = max(0, (min_len - lookback_window - prediction_horizon) // stride + 1)
        
        logger.debug(
            f"TimeSeriesDataset: {self.n_samples} 样本, "
            f"window={lookback_window}, horizon={prediction_horizon}, "
            f"features={len(feature_cols)}"
        )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个训练样本
        
        参数:
            idx: 样本索引
        
        返回:
            x: 输入特征 (lookback_window, n_features)
            y_reg: 回归目标 (prediction_horizon,) - 未来收益率
            y_cls: 分类目标 (prediction_horizon,) - 涨跌方向(0/1)
        """
        start = idx * self.stride
        end = start + self.lookback
        
        # 输入特征窗口
        x = torch.FloatTensor(self.scaled_data[start:end])
        
        # 目标窗口
        target_start = end
        target_end = target_start + self.horizon
        
        # 目标列取最后一列（对应 returns_{horizon}d，由 build_returns 保证列顺序）
        y_reg = torch.FloatTensor(self.returns[target_start:target_end, -1])
        
        # 分类标签：收益率>0为1（涨），否则为0（跌）
        y_cls = (y_reg > 0).float()
        
        return x, y_reg, y_cls


# ============================================================================
# 数据准备函数
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
    准备训练/验证/测试数据
    
    将多只股票的加工数据合并，创建标准化的DataLoader。
    
    重要：时序数据不能随机shuffle训练集！
    我们按时间顺序划分每个股票的数据，然后合并。
    
    划分策略（时序感知）：
    - 每只股票自己的数据按时间顺序划分为 train/val/test
    - train: 前70%（最早的时间段）
    - val: 中间15%
    - test: 最后15%（最接近当前的时间段）
    
    这种划分方式模拟了真实的交易场景：
    用历史数据训练，用未来数据测试。
    
    时间复杂度: O(t*n)，t=股票数，n=平均每只股票的数据行数
    空间复杂度: O(t*n)
    
    参数:
        tickers: 股票代码列表（None则加载所有可用的特征数据）
        config: 模型配置
        data_dir: 加工数据目录
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
    
    返回:
        (train_loader, val_loader, test_loader, scaler)
    """
    if config is None:
        config = ModelConfig()
    
    storage = ParquetStorage(data_dir)
    
    # 1. 确定股票列表
    if tickers is None:
        available_keys = storage.list_keys()
        tickers = list(set(
            k.replace('_features', '') 
            for k in available_keys 
            if k.endswith('_features')
        ))
    
    if not tickers:
        raise DataError(f"在 {data_dir} 中未找到特征数据")
    
    logger.info(f"准备数据: {len(tickers)} 只股票")
    
    # 2. 加载并合并所有股票的数据
    all_train_features = []
    all_val_features = []
    all_test_features = []
    
    scaler = StandardScaler()
    
    for ticker in tickers:
        try:
            df = storage.load(f"{ticker}_features")
            
            if df is None or len(df) < config.lookback_window + config.prediction_horizon + 50:
                logger.debug(f"跳过 {ticker}: 数据量不足 ({len(df) if df is not None else 0} 行)")
                continue
            
            # 按时间顺序划分
            n = len(df)
            train_end = int(n * (1 - val_size - test_size))
            val_end = int(n * (1 - test_size))
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # 收集特征数据用于拟合scaler（所有股票参与partial_fit）
            if len(train_df) > 0:
                train_features = train_df[config.features].dropna().values
                if len(train_features) > 0:
                    scaler.partial_fit(train_features)
            
            all_train_features.append(train_df)
            all_val_features.append(val_df)
            all_test_features.append(test_df)
            
        except Exception as e:
            logger.warning(f"加载 {ticker} 数据失败: {e}")
            continue
    
    if not all_train_features:
        raise DataError("没有可用的训练数据")
    
    # 3. 合并数据
    train_combined = pd.concat(all_train_features)
    val_combined = pd.concat(all_val_features) if all_val_features else pd.DataFrame()
    test_combined = pd.concat(all_test_features) if all_test_features else pd.DataFrame()
    
    logger.info(
        f"数据划分: train={len(train_combined)}, "
        f"val={len(val_combined)}, test={len(test_combined)}"
    )
    
    # 4. 创建Dataset
    train_dataset = TimeSeriesDataset(
        train_combined, config.features,
        config.lookback_window, config.prediction_horizon,
        scaler=scaler, fit_scaler=False  # 已经partial_fit了
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
    
    # 5. 创建DataLoader
    # 注意：对于时序数据，shuffle=True会破坏时间依赖关系
    # 但在训练集中轻微shuffle（batch级别）有助于随机梯度下降
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 对训练集shuffle batch顺序
        num_workers=0,  # 避免多进程问题
        drop_last=True  # 丢弃最后不完整的batch
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
        f"DataLoader已创建: "
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
    加载单只股票的数据用于预测
    
    参数:
        ticker: 股票代码
        config: 模型配置
        data_dir: 数据目录
    
    返回:
        (data_loader, scaler, original_df)
    """
    storage = ParquetStorage(data_dir)
    
    try:
        df = storage.load(f"{ticker}_features")
    except Exception:
        raise DataError(f"未找到 {ticker} 的特征数据")
    
    if len(df) < config.lookback_window:
        raise DataError(f"{ticker} 数据量({len(df)})不足窗口大小({config.lookback_window})")
    
    # 创建scaler
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
