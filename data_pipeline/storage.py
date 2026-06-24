"""Parquet-based feature storage."""

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
# 抽象存儲接口
# ============================================================================
class DataStorage(ABC):
    """
    數據存儲抽象基類
    
    所有存儲後端必須實現此接口。
    """
    
    @abstractmethod
    def save(self, df: pd.DataFrame, key: str) -> None:
        """
        保存DataFrame
        
        參數:
            df: 要保存的數據
            key: 唯一標識（如 'AAPL_daily'）
        """
        pass
    
    @abstractmethod
    def load(self, key: str) -> pd.DataFrame:
        """
        加載DataFrame
        
        參數:
            key: 數據標識
        
        返回:
            加載的DataFrame
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        檢查數據是否存在
        
        參數:
            key: 數據標識
        
        返回:
            是否存在
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """
        刪除數據
        
        參數:
            key: 數據標識
        """
        pass
    
    @abstractmethod
    def list_keys(self) -> List[str]:
        """
        列出所有存儲的數據標識
        
        返回:
            標識列表
        """
        pass


# ============================================================================
# Parquet格式存儲（推薦用於歷史行情數據）
# ============================================================================
class ParquetStorage(DataStorage):
    """
    基於Apache Parquet格式的數據存儲
    
    Parquet優勢：
    - 列式存儲：只讀取需要的列，I/O效率高
    - 高壓縮比：通常比CSV小5-10倍
    - 支持謂詞下推：可在讀取時過濾數據
    - 支持分區：按日期/股票代碼分區
    
    適用場景：歷史日線數據、分鐘數據、特徵數據
    """
    
    def __init__(self, base_dir: Path = PROCESSED_DATA_DIR):
        """
        參數:
            base_dir: 數據存儲根目錄
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """根據key生成文件路徑"""
        return self.base_dir / f"{key}.parquet"
    
    def save(self, df: pd.DataFrame, key: str, 
             compression: str = 'snappy') -> None:
        """
        保存DataFrame為Parquet格式
        
        參數:
            df: 要保存的數據
            key: 數據標識
            compression: 壓縮算法 ('snappy', 'gzip', 'brotli', 'zstd')
                        snappy速度快，gzip壓縮率高
        """
        path = self._get_path(key)
        df.to_parquet(path, compression=compression, index=True)
        logger.debug(f"已保存 {key} → {path} ({len(df)} 行)")
    
    def load(self, key: str, columns: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加載Parquet數據
        
        支持列選擇和日期過濾，減少不必要的I/O。
        
        參數:
            key: 數據標識
            columns: 要加載的列名列表（None加載全部）
            start_date: 起始日期過濾
            end_date: 結束日期過濾
        
        返回:
            加載的DataFrame
        """
        path = self._get_path(key)
        
        if not path.exists():
            raise DataError(f"數據不存在: {key}", {'path': str(path)})
        
        df = pd.read_parquet(path, columns=columns)
        
        # 日期範圍過濾
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
            logger.debug(f"已刪除: {key}")
    
    def list_keys(self) -> List[str]:
        return [p.stem for p in self.base_dir.glob("*.parquet")]


# ============================================================================
# HDF5格式存儲（適合高效批量讀寫）
# ============================================================================
class HDF5Storage(DataStorage):
    """
    基於HDF5格式的數據存儲
    
    HDF5優勢：
    - 適合存儲大量結構化數據
    - 支持分層存儲（group/dataset）
    - 單個文件存儲多張表
    - 壓縮和分塊讀取
    
    注意：HDF5跨平臺兼容性不如Parquet，
    在生產環境中建議使用Parquet。
    """
    
    def __init__(self, file_path: Path = PROCESSED_DATA_DIR / "market_data.h5"):
        """
        參數:
            file_path: HDF5文件路徑
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def save(self, df: pd.DataFrame, key: str) -> None:
        """
        保存DataFrame到HDF5文件
        
        使用'table'格式以支持查詢，'fixed'格式更快但不支持查詢。
        """
        df.to_hdf(
            self.file_path, key=key,
            mode='a',           # 追加模式
            format='table',     # 支持條件查詢
            complib='blosc',    # 使用blosc壓縮（速度快）
            complevel=5
        )
        logger.debug(f"已保存 {key} → {self.file_path} ({len(df)} 行)")
    
    def load(self, key: str, 
             where: Optional[str] = None,
             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        從HDF5加載數據
        
        參數:
            key: 數據集鍵名
            where: PyTables查詢條件（如 'date > "2023-01-01"'）
            columns: 要加載的列名
        
        返回:
            加載的DataFrame
        """
        try:
            df = pd.read_hdf(
                self.file_path, key=key,
                where=where, columns=columns
            )
            return df
        except KeyError:
            raise DataError(f"數據集不存在: {key}", {'file': str(self.file_path)})
    
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
                logger.debug(f"已刪除數據集: {key}")
    
    def list_keys(self) -> List[str]:
        try:
            with pd.HDFStore(self.file_path, mode='r') as store:
                return list(store.keys())
        except Exception:
            return []


# ============================================================================
# 原始數據存儲（不可變）
# ============================================================================
class RawDataStore:
    """
    原始數據存儲管理器
    
    設計原則：
    - 原始數據一旦寫入，永不修改（immutable）
    - 每次爬取的數據帶版本標記
    - 提供數據完整性校驗
    """
    
    def __init__(self, base_dir: Path = RAW_DATA_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_store = ParquetStorage(self.base_dir)
    
    def save_raw(self, df: pd.DataFrame, ticker: str, 
                 version: Optional[str] = None) -> str:
        """
        保存原始數據
        
        參數:
            df: 原始OHLCV數據
            ticker: 股票代碼
            version: 版本標識（None則自動生成時間戳版本）
        
        返回:
            存儲的key
        """
        if version is None:
            version = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        key = f"{ticker}_daily_v{version}"
        self.parquet_store.save(df, key)
        logger.info(f"原始數據已存檔: {key}")
        return key
    
    def load_raw(self, ticker: str, version: Optional[str] = None) -> pd.DataFrame:
        """
        加載原始數據
        
        參數:
            ticker: 股票代碼
            version: 版本標識（None則加載最新版本）
        
        返回:
            原始DataFrame
        """
        if version:
            key = f"{ticker}_daily_v{version}"
            return self.parquet_store.load(key)
        
        # 加載最新版本
        all_keys = self.parquet_store.list_keys()
        matching = [k for k in all_keys if k.startswith(f"{ticker}_daily_v")]
        
        if not matching:
            raise DataError(f"未找到 {ticker} 的原始數據")
        
        # 按版本號排序，取最新
        latest_key = sorted(matching, reverse=True)[0]
        return self.parquet_store.load(latest_key)


# ============================================================================
# 模型存儲
# ============================================================================
class ModelStorage:
    """
    ML模型持久化存儲
    
    支持PyTorch模型和scikit-learn模型的保存和加載。
    """
    
    def __init__(self, base_dir: Path = MODELS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_torch_model(self, model, name: str, 
                         metadata: Optional[Dict] = None) -> Path:
        """
        保存PyTorch模型
        
        同時保存模型狀態字典和訓練元數據。
        
        參數:
            model: PyTorch模型對象
            name: 模型名稱
            metadata: 訓練元數據（超參數、精度等）
        
        返回:
            模型文件路徑
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
        加載PyTorch模型
        
        參數:
            model_class: 模型類（用於實例化）
            name: 模型名稱
        
        返回:
            (model, metadata) 元組
        """
        import torch
        
        path = self.base_dir / f"{name}.pt"
        if not path.exists():
            raise DataError(f"模型文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # 從元數據中重建模型配置
        metadata = checkpoint.get('metadata', {})
        
        # 實例化模型並加載權重
        model = model_class(**metadata.get('model_params', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"模型已加載: {path}")
        return model, metadata
    
    def save_sklearn_model(self, model, name: str) -> Path:
        """
        保存scikit-learn模型（用於標準化器等預處理對象）
        
        參數:
            model: sklearn模型對象
            name: 模型名稱
        
        返回:
            文件路徑
        """
        path = self.base_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        return path
    
    def load_sklearn_model(self, name: str):
        """
        加載scikit-learn模型
        
        參數:
            name: 模型名稱
        
        返回:
            模型對象
        """
        path = self.base_dir / f"{name}.pkl"
        if not path.exists():
            raise DataError(f"模型文件不存在: {path}")
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def list_models(self) -> List[str]:
        """列出所有已保存的模型"""
        return [p.stem for p in self.base_dir.glob("*")]
