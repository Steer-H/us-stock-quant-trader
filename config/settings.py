"""Trading and model configuration."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Literal
from dataclasses import dataclass, field

# ============================================================================
# 項目根路徑
# ============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"        # 原始爬取數據
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # 清洗後數據
MODELS_DIR: Path = DATA_DIR / "models"        # 訓練好的模型
LOGS_DIR: Path = PROJECT_ROOT / "logs"        # 日誌目錄


# ============================================================================
# 系統基礎配置
# ============================================================================
@dataclass
class SystemConfig:
    """系統級別的全局配置"""
    # 運行模式: 'backtest'(回測), 'paper'(模擬交易), 'live'(實盤交易)
    mode: Literal['backtest', 'paper', 'live'] = 'backtest'
    
    # 日誌級別: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = 'INFO'
    
    # 日誌文件最大大小（MB），超出後自動輪轉
    log_max_size_mb: int = 100
    
    # 保留的日誌文件備份數量
    log_backup_count: int = 10
    
    # 性能監控：是否記錄每個模塊的執行耗時
    enable_perf_logging: bool = True
    
    # 隨機種子，確保實驗可復現
    random_seed: int = 42
    
    # 最大並發線程數（用於爬蟲、數據拉取等IO密集型任務）
    max_workers: int = 8
    
    @property
    def is_live(self) -> bool:
        """是否處於實盤交易模式"""
        return self.mode == 'live'

    @property
    def is_paper(self) -> bool:
        """是否處於模擬交易模式"""
        return self.mode == 'paper'

    @property
    def is_backtest(self) -> bool:
        """是否處於回測模式"""
        return self.mode == 'backtest'


# ============================================================================
# 數據源配置
# ============================================================================
@dataclass
class DataSourceConfig:
    """
    數據源相關配置
    
    美股量化交易依賴多類數據：
    - Level-1行情: 買賣報價、成交價量（基礎）
    - Level-2行情: 深度報價（高級策略用）
    - 歷史數據: 復權價格、分時數據（回測用）
    - 參考數據: 基本面、行業分類、財報日曆等
    - 另類數據: 做空餘額、新聞情緒等（可選）
    """
    
    # ---- Yahoo Finance（免費歷史數據，爬蟲使用） ----
    yahoo_enabled: bool = True
    
    # ---- Polygon.io（付費實時+歷史行情） ----
    polygon_api_key: Optional[str] = None
    polygon_enabled: bool = False
    
    # ---- Alpha Vantage（免費/付費行情備選） ----
    alpha_vantage_api_key: Optional[str] = None
    alpha_vantage_enabled: bool = False
    
    # ---- Interactive Brokers（券商數據源，實盤用） ----
    ibkr_host: str = '127.0.0.1'
    ibkr_port: int = 7497  # TWS默認7497，IB Gateway默認4002
    ibkr_client_id: int = 1
    
    # ---- 數據時間範圍 ----
    # 歷史數據起始年份
    history_start_year: int = 2010
    # 歷史數據結束年份
    history_end_year: int = 2025
    # 默認K線周期: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    default_bar_interval: str = '1d'
    
    # ---- 數據清洗 ----
    # 是否自動復權（forward adjust for splits & dividends）
    auto_adjust_prices: bool = True
    # 是否填充缺失數據（forward fill後backward fill）
    fill_missing_data: bool = True
    # 異常價格波動倍數閾值（超過此倍數的跳變視為數據錯誤）
    price_spike_threshold: float = 5.0
    
    # ---- 股票池 ----
    # 默認交易的股票池文件路徑（TXT，每行一個ticker）
    stock_universe_file: Optional[str] = None
    # 默認跟蹤的指數代碼列表
    benchmark_indices: List[str] = field(default_factory=lambda: ['^GSPC', '^IXIC', '^DJI'])
    
    # ---- 另類數據（可選） ----
    # 是否拉取做空餘額數據
    fetch_short_interest: bool = False
    # 是否拉取新聞情緒數據
    fetch_news_sentiment: bool = False


# ============================================================================
# ML模型配置 - Transformer 架構
# ============================================================================
@dataclass
class ModelConfig:
    """
    機器學習模型配置
    
    採用Transformer架構進行時序預測，核心思想：
    - 將股票的歷史價格序列、技術指標、市場特徵編碼為嵌入向量
    - 通過多頭自注意力機制捕捉長期依賴關係
    - 輸出未來N期的價格走勢預測
    
    參考文獻: "Attention Is All You Need" (Vaswani et al., 2017)
    """
    
    # ---- Transformer 架構參數 ----
    # 模型維度（嵌入向量的維度），通常為64的倍數
    d_model: int = 192
    
    # 多頭注意力的頭數，d_model必須能被n_heads整除
    n_heads: int = 6
    
    # Encoder層數
    n_encoder_layers: int = 3
    
    # Decoder層數
    n_decoder_layers: int = 4
    
    # 前饋網絡的隱藏層維度（通常是d_model的4倍）
    d_ff: int = 768
    
    # Dropout率，防止過擬合
    dropout: float = 0.2
    
    # Stochastic Depth drop path rate (0=disabled)
    drop_path_rate: float = 0.1
    
    # 位置編碼的最大序列長度
    max_seq_len: int = 252  # 一年約252個交易日
    
    # ---- 輸入特徵 ----
    # 歷史時間窗口大小（用多少天的數據預測）
    lookback_window: int = 60
    
    # 預測時間窗口大小（預測未來多少天）
    prediction_horizon: int = 5
    
    # 使用的特徵列表
    features: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',          # OHLCV
        'returns_1d', 'returns_5d', 'returns_20d',          # 收益率
        'volatility_5d', 'volatility_20d',                  # 波動率
        'sma_5', 'sma_20', 'sma_60',                        # 簡單移動均線
        'ema_12', 'ema_26',                                  # 指數移動均線
        'rsi_14',                                            # 相對強弱指標
        'macd', 'macd_signal', 'macd_hist',
                          # MACD
        'bb_upper', 'bb_middle', 'bb_lower',                 # 布林帶
        # 新聞情感特徵 (小權重輔助信號)
        'news_sentiment_3d', 'news_sentiment_7d',              # 新聞情感得分
        'earnings_surprise_pct',                                # 盈利驚喜
        'has_earnings_report',                                  # 財報發布日標記
        'atr_14',                                            # 平均真實波幅
        'volume_ratio',                                      # 量比
    ])
    
    # 情感特徵名稱列表（用於模型中的分組加權）
    sentiment_features: List[str] = field(default_factory=lambda: [
        'news_sentiment_3d', 'news_sentiment_7d',
        'earnings_surprise_pct', 'has_earnings_report',
    ])
    # 新聞情感特徵權重（小權重，僅作輔助參考）
    news_feature_weight: float = 0.05
    
    # 特徵歸一化方式: 'standard'(標準化), 'minmax'(最大最小), 'robust'(魯棒)
    normalization: str = 'standard'
    
    # 時序數據增強（防止過擬合）
    use_data_augmentation: bool = True
    aug_noise_std: float = 0.005       # 高斯噪聲標準差（相對價格）
    aug_time_warp_sigma: float = 0.1   # 時間扭曲強度
    aug_scale_sigma: float = 0.05      # 幅度縮放強度
    
    # ---- 訓練超參數 ----
    # 每批次的樣本數
    batch_size: int = 32
    
    # 最大訓練輪數
    epochs: int = 30
    
    # 學習率
    learning_rate: float = 3e-4
    
    # 學習率調度器衰減因子
    lr_decay: float = 0.95
    
    # CosineAnnealingWarmRestarts 參數
    cosine_T_0: int = 20     # 第一個重啟周期的epoch數
    cosine_T_mult: int = 2   # 每個重啟周期倍增因子
    cosine_eta_min: float = 1e-6  # 最小學習率
    
    # 優化器: 'adam', 'adamw', 'sgd'
    optimizer: str = 'adamw'
    
    # 權重衰減（L2正則化），防止過擬合
    weight_decay: float = 1e-5
    
    # ---- Focal Loss 參數 ----
    # Focal Loss gamma (focus on hard examples)
    focal_gamma: float = 2.0
    # Focal Loss alpha (class balance)
    focal_alpha: float = 0.25
    # Label smoothing
    label_smoothing: float = 0.1
    # 新聞情感特徵權重（小權重，僅作輔助參考）
    
    # 學習率預熱步數佔總步數的比例
    warmup_ratio: float = 0.1
    
    # Label Smoothing (防止過置信，金融數據信號噪聲比低)
    # 新聞情感特徵權重（小權重，僅作輔助參考）
    
    # Stochastic Depth (DropPath) 概率
    
    # 注意力Dropout（高於普通dropout，金融數據噪聲大）
    attn_dropout: float = 0.2
    
    # Stochastic Depth drop path rate (0=disabled)
    
    # 梯度累積步數（小batch增大等效batch）
    gradient_accumulation_steps: int = 2
    
    # 早停耐心值：驗證損失連續N輪不改善則停止訓練
    early_stopping_patience: int = 20
    
    # ---- 訓練策略 ----
    # 學習率調度器類型: 'plateau' 或 'cosine'
    scheduler_type: str = 'cosine'
    # Cosine退火第一個restart周期(epoch)
    # 梯度裁剪norm閾值
    grad_clip_norm: float = 1.0
    
    # 梯度裁剪閾值，防止梯度爆炸
    grad_clip: float = 1.0
    
    # ---- 訓練/驗證/測試集劃分 ----
    # 訓練集比例
    train_ratio: float = 0.70
    # 驗證集比例
    val_ratio: float = 0.15
    # 測試集比例（自動計算: 1 - train_ratio - val_ratio）
    
    # ---- 獎勵函數參數 ----
    # 預測方向準確的獎勵權重
    direction_reward_weight: float = 0.6
    # 預測幅度準確的獎勵權重
    magnitude_reward_weight: float = 0.4
    
    # ---- 精度閾值 ----
    # 方向預測準確率低於此閾值時觸發調參
    min_direction_accuracy: float = 0.55
    # 均方根誤差高於此閾值時觸發調參
    max_rmse: float = 0.06  # 適當放寬RMSE閾值，方向準確率是核心指標
    
    # ---- 硬體配置 ----
    # 設備: 'cpu', 'cuda', 'mps'(Apple Silicon)
    device: str = 'mps' if __import__('torch').backends.mps.is_available() else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    
    # 是否使用混合精度訓練（節省顯存，加速訓練）
    use_amp: bool = True   # Automatic Mixed Precision (MPS不支持則自動fallback)


# ============================================================================
# 交易配置
# ============================================================================
@dataclass
class TradingConfig:
    """
    交易相關配置
    
    涵蓋券商接入、訂單類型、風控參數等。
    """
    
    # ---- 券商接入 ----
    # 選用的券商: 'ibkr'(Interactive Brokers), 'alpaca', 'td_ameritrade'
    broker: str = 'ibkr'
    
    # IBKR TWS/Gateway連接地址
    ibkr_account_id: Optional[str] = None
    
    # Alpaca API密鑰
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = 'https://paper-api.alpaca.markets'
    
    # ---- 帳戶參數 ----
    # 初始資金（回測/模擬用）
    initial_capital: float = 100_000.0
    
    # 基礎貨幣
    base_currency: str = 'USD'
    
    # ---- 交易時段 ----
    # 是否允許盤前交易（美東 4:00-9:30）
    allow_pre_market: bool = False
    # 是否允許盤後交易（美東 16:00-20:00）
    allow_after_hours: bool = False
    # 正常交易時段（美東 9:30-16:00）始終開啟
    
    # ---- 費率設置 ----
    # 每股佣金（美元，Interactive Brokers階梯式收費約$0.005/股）
    commission_per_share: float = 0.005
    # 每筆最低佣金
    commission_min: float = 1.0
    # 每筆最高佣金（佔成交額百分比）
    commission_max_pct: float = 0.01
    # SEC費用（銷售金額的0.00278%，2025年）
    sec_fee_rate: float = 0.0000278
    # TAF費用（每股$0.000166，賣出時收取，上限$8.30）
    taf_fee_per_share: float = 0.000166
    taf_fee_max: float = 8.30
    
    # 做空借券年化費率（估算默認值）
    short_borrow_rate_annual: float = 0.003  # 0.3% for easy-to-borrow
    
    # 滑點模型參數
    # 滑點 = base_bps + vol_coeff * 波動率 + size_coeff * sqrt(訂單量/日均成交量)
    slippage_base_bps: float = 1.0   # 基礎滑點（基點）
    slippage_vol_coeff: float = 5.0  # 波動率係數
    slippage_size_coeff: float = 10.0  # 訂單規模係數
    
    # ---- 默認訂單設置 ----
    # 默認訂單類型: 'MKT'(市價), 'LMT'(限價), 'STP'(止損)
    default_order_type: str = 'LMT'
    # 限價單的偏移量（相對於當前價格的百分比）
    limit_price_offset_pct: float = 0.002  # 0.2%
    # 默認訂單有效期: 'DAY', 'GTC', 'IOC'
    default_time_in_force: str = 'DAY'
    
    # ---- 風控參數（全局限制，各策略可進一步收緊） ----
    # 單筆最大成交金額（佔帳戶比例）
    max_order_amount_pct: float = 0.10  # 10%
    # 單日最大成交次數
    max_daily_trades: int = 50
    # 最大持倉數量
    max_positions: int = 20
    # 單股最大持倉比例
    max_position_pct: float = 0.20  # 20%
    # 總體最大槓桿倍數
    max_leverage: float = 2.0
    # 最大回撤比例（觸及則停止交易）
    max_drawdown_pct: float = 0.25  # 25%
    # PDT規則：日內交易次數限制（5個交易日4次）
    pdt_day_trade_limit: int = 3  # 保守設為3，留有餘地
    # PDT最低帳戶資金
    pdt_min_equity: float = 25_000.0


# ============================================================================
# 全局配置實例（單例模式）
# ============================================================================
# 創建全局配置實例，各模塊通過 `from config import system_config` 引入
system_config: SystemConfig = SystemConfig()
data_source_config: DataSourceConfig = DataSourceConfig()
model_config: ModelConfig = ModelConfig()
trading_config: TradingConfig = TradingConfig()
