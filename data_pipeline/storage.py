"""
美股量化交易系统 - 数据存储模块

提供高效的数据持久化方案，支持：
- HDF5格式: 适合大规模时间序列数据的读写（推荐）
- Parquet格式: 列式存储，适合分析和查询
- CSV格式: 通用格式，适合小数据量和人工查看

存储层次结构：
data/
├── raw/          # 原始爬取数据（不可变）
│   └── {ticker}_daily.parquet
├── processed/    # 清洗后数据（含技术指标）
│   └── {ticker}_features.parquet
└── models/       # 训练好的ML模型
    └── transformer_*.pt
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
from abc import ABC, abstractmethod
import pickle

import pandas as pd
import numpy as np

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from utils.exceptions import DataError

logger = logging.getLogger(__name__)


# ============================================================================
# 抽象存储接口
# ============================================================================
class DataStorage(ABC):
    """
    数据存储抽象基类
    
    所有存储后端必须实现此接口。
    """
    
    @abstractmethod
    def save(self, df: pd.DataFrame, key: str) -> None:
        """
        保存DataFrame
        
        参数:
            df: 要保存的数据
            key: 唯一标识（如 'AAPL_daily'）
        """
        pass
    
    @abstractmethod
    def load(self, key: str) -> pd.DataFrame:
        """
        加载DataFrame
        
        参数:
            key: 数据标识
        
        返回:
            加载的DataFrame
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查数据是否存在
        
        参数:
            key: 数据标识
        
        返回:
            是否存在
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """
        删除数据
        
        参数:
            key: 数据标识
        """
        pass
    
    @abstractmethod
    def list_keys(self) -> List[str]:
        """
        列出所有存储的数据标识
        
        返回:
            标识列表
        """
        pass


# ============================================================================
# Parquet格式存储（推荐用于历史行情数据）
# ============================================================================
class ParquetStorage(DataStorage):
    """
    基于Apache Parquet格式的数据存储
    
    Parquet优势：
    - 列式存储：只读取需要的列，I/O效率高
    - 高压缩比：通常比CSV小5-10倍
    - 支持谓词下推：可在读取时过滤数据
    - 支持分区：按日期/股票代码分区
    
    适用场景：历史日线数据、分钟数据、特征数据
    """
    
    def __init__(self, base_dir: Path = PROCESSED_DATA_DIR):
        """
        参数:
            base_dir: 数据存储根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """根据key生成文件路径"""
        return self.base_dir / f"{key}.parquet"
    
    def save(self, df: pd.DataFrame, key: str, 
             compression: str = 'snappy') -> None:
        """
        保存DataFrame为Parquet格式
        
        参数:
            df: 要保存的数据
            key: 数据标识
            compression: 压缩算法 ('snappy', 'gzip', 'brotli', 'zstd')
                        snappy速度快，gzip压缩率高
        """
        path = self._get_path(key)
        df.to_parquet(path, compression=compression, index=True)
        logger.debug(f"已保存 {key} → {path} ({len(df)} 行)")
    
    def load(self, key: str, columns: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加载Parquet数据
        
        支持列选择和日期过滤，减少不必要的I/O。
        
        参数:
            key: 数据标识
            columns: 要加载的列名列表（None加载全部）
            start_date: 起始日期过滤
            end_date: 结束日期过滤
        
        返回:
            加载的DataFrame
        """
        path = self._get_path(key)
        
        if not path.exists():
            raise DataError(f"数据不存在: {key}", {'path': str(path)})
        
        df = pd.read_parquet(path, columns=columns)
        
        # 日期范围过滤
        if start_date or end_date:
            if isinstance(df.index, pd.DatetimeIndex):
                if start_date:
                    df = df[df.index >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]
        
        return df
    
    def exists(self, key: str) -> bool:
        return self._get_path(key).exists()
    
    def delete(self, key: str) -> None:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            logger.debug(f"已删除: {key}")
    
    def list_keys(self) -> List[str]:
        return [p.stem for p in self.base_dir.glob("*.parquet")]


# ============================================================================
# HDF5格式存储（适合高效批量读写）
# ============================================================================
class HDF5Storage(DataStorage):
    """
    基于HDF5格式的数据存储
    
    HDF5优势：
    - 适合存储大量结构化数据
    - 支持分层存储（group/dataset）
    - 单个文件存储多张表
    - 压缩和分块读取
    
    注意：HDF5跨平台兼容性不如Parquet，
    在生产环境中建议使用Parquet。
    """
    
    def __init__(self, file_path: Path = PROCESSED_DATA_DIR / "market_data.h5"):
        """
        参数:
            file_path: HDF5文件路径
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, df: pd.DataFrame, key: str) -> None:
        """
        保存DataFrame到HDF5文件
        
        使用'table'格式以支持查询，'fixed'格式更快但不支持查询。
        """
        df.to_hdf(
            self.file_path, key=key,
            mode='a',           # 追加模式
            format='table',     # 支持条件查询
            complib='blosc',    # 使用blosc压缩（速度快）
            complevel=5
        )
        logger.debug(f"已保存 {key} → {self.file_path} ({len(df)} 行)")
    
    def load(self, key: str, 
             where: Optional[str] = None,
             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        从HDF5加载数据
        
        参数:
            key: 数据集键名
            where: PyTables查询条件（如 'date > "2023-01-01"'）
            columns: 要加载的列名
        
        返回:
            加载的DataFrame
        """
        try:
            df = pd.read_hdf(
                self.file_path, key=key,
                where=where, columns=columns
            )
            return df
        except KeyError:
            raise DataError(f"数据集不存在: {key}", {'file': str(self.file_path)})
    
    def exists(self, key: str) -> bool:
        try:
            with pd.HDFStore(self.file_path, mode='r') as store:
                return key in store
        except Exception:
            return False
    
    def delete(self, key: str) -> None:
        with pd.HDFStore(self.file_path, mode='a') as store:
            if key in store:
                del store[key]
                logger.debug(f"已删除数据集: {key}")
    
    def list_keys(self) -> List[str]:
        try:
            with pd.HDFStore(self.file_path, mode='r') as store:
                return list(store.keys())
        except Exception:
            return []


# ============================================================================
# 原始数据存储（不可变）
# ============================================================================
class RawDataStore:
    """
    原始数据存储管理器
    
    设计原则：
    - 原始数据一旦写入，永不修改（immutable）
    - 每次爬取的数据带版本标记
    - 提供数据完整性校验
    """
    
    def __init__(self, base_dir: Path = RAW_DATA_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_store = ParquetStorage(self.base_dir)
    
    def save_raw(self, df: pd.DataFrame, ticker: str, 
                 version: Optional[str] = None) -> str:
        """
        保存原始数据
        
        参数:
            df: 原始OHLCV数据
            ticker: 股票代码
            version: 版本标识（None则自动生成时间戳版本）
        
        返回:
            存储的key
        """
        if version is None:
            version = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        key = f"{ticker}_daily_v{version}"
        self.parquet_store.save(df, key)
        logger.info(f"原始数据已存档: {key}")
        return key
    
    def load_raw(self, ticker: str, version: Optional[str] = None) -> pd.DataFrame:
        """
        加载原始数据
        
        参数:
            ticker: 股票代码
            version: 版本标识（None则加载最新版本）
        
        返回:
            原始DataFrame
        """
        if version:
            key = f"{ticker}_daily_v{version}"
            return self.parquet_store.load(key)
        
        # 加载最新版本
        all_keys = self.parquet_store.list_keys()
        matching = [k for k in all_keys if k.startswith(f"{ticker}_daily_v")]
        
        if not matching:
            raise DataError(f"未找到 {ticker} 的原始数据")
        
        # 按版本号排序，取最新
        latest_key = sorted(matching, reverse=True)[0]
        return self.parquet_store.load(latest_key)


# ============================================================================
# 模型存储
# ============================================================================
class ModelStorage:
    """
    ML模型持久化存储
    
    支持PyTorch模型和scikit-learn模型的保存和加载。
    """
    
    def __init__(self, base_dir: Path = MODELS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_torch_model(self, model, name: str, 
                         metadata: Optional[Dict] = None) -> Path:
        """
        保存PyTorch模型
        
        同时保存模型状态字典和训练元数据。
        
        参数:
            model: PyTorch模型对象
            name: 模型名称
            metadata: 训练元数据（超参数、精度等）
        
        返回:
            模型文件路径
        """
        import torch
        
        path = self.base_dir / f"{name}.pt"
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {},
            'save_time': pd.Timestamp.now().isoformat(),
        }
        
        torch.save(save_dict, path)
        logger.info(f"模型已保存: {path}")
        return path
    
    def load_torch_model(self, model_class, name: str) -> tuple:
        """
        加载PyTorch模型
        
        参数:
            model_class: 模型类（用于实例化）
            name: 模型名称
        
        返回:
            (model, metadata) 元组
        """
        import torch
        
        path = self.base_dir / f"{name}.pt"
        if not path.exists():
            raise DataError(f"模型文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # 从元数据中重建模型配置
        metadata = checkpoint.get('metadata', {})
        
        # 实例化模型并加载权重
        model = model_class(**metadata.get('model_params', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"模型已加载: {path}")
        return model, metadata
    
    def save_sklearn_model(self, model, name: str) -> Path:
        """
        保存scikit-learn模型（用于标准化器等预处理对象）
        
        参数:
            model: sklearn模型对象
            name: 模型名称
        
        返回:
            文件路径
        """
        path = self.base_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        return path
    
    def load_sklearn_model(self, name: str):
        """
        加载scikit-learn模型
        
        参数:
            name: 模型名称
        
        返回:
            模型对象
        """
        path = self.base_dir / f"{name}.pkl"
        if not path.exists():
            raise DataError(f"模型文件不存在: {path}")
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def list_models(self) -> List[str]:
        """列出所有已保存的模型"""
        return [p.stem for p in self.base_dir.glob("*")]
