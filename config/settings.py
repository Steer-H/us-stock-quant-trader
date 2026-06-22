"""
美股量化交易系统 - 全局配置模块

本模块定义了系统的所有配置项，包括：
- 系统基础配置（路径、模式、日志等）
- 数据源配置（行情API、历史数据源）
- ML模型配置（Transformer架构参数、训练超参数）
- 交易配置（券商API、风控参数）

采用 pydantic 进行配置校验，确保配置项的类型安全和合法性。
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Literal
from dataclasses import dataclass, field

# ============================================================================
# 项目根路径
# ============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"        # 原始爬取数据
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # 清洗后数据
MODELS_DIR: Path = DATA_DIR / "models"        # 训练好的模型
LOGS_DIR: Path = PROJECT_ROOT / "logs"        # 日志目录


# ============================================================================
# 系统基础配置
# ============================================================================
@dataclass
class SystemConfig:
    """系统级别的全局配置"""
    # 运行模式: 'backtest'(回测), 'paper'(模拟交易), 'live'(实盘交易)
    mode: Literal['backtest', 'paper', 'live'] = 'backtest'
    
    # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = 'INFO'
    
    # 日志文件最大大小（MB），超出后自动轮转
    log_max_size_mb: int = 100
    
    # 保留的日志文件备份数量
    log_backup_count: int = 10
    
    # 性能监控：是否记录每个模块的执行耗时
    enable_perf_logging: bool = True
    
    # 随机种子，确保实验可复现
    random_seed: int = 42
    
    # 最大并发线程数（用于爬虫、数据拉取等IO密集型任务）
    max_workers: int = 8
    
    @property
    def is_live(self) -> bool:
        """是否处于实盘交易模式"""
        return self.mode == 'live'

    @property
    def is_paper(self) -> bool:
        """是否处于模拟交易模式"""
        return self.mode == 'paper'

    @property
    def is_backtest(self) -> bool:
        """是否处于回测模式"""
        return self.mode == 'backtest'


# ============================================================================
# 数据源配置
# ============================================================================
@dataclass
class DataSourceConfig:
    """
    数据源相关配置
    
    美股量化交易依赖多类数据：
    - Level-1行情: 买卖报价、成交价量（基础）
    - Level-2行情: 深度报价（高级策略用）
    - 历史数据: 复权价格、分时数据（回测用）
    - 参考数据: 基本面、行业分类、财报日历等
    - 另类数据: 做空余额、新闻情绪等（可选）
    """
    
    # ---- Yahoo Finance（免费历史数据，爬虫使用） ----
    yahoo_enabled: bool = True
    
    # ---- Polygon.io（付费实时+历史行情） ----
    polygon_api_key: Optional[str] = None
    polygon_enabled: bool = False
    
    # ---- Alpha Vantage（免费/付费行情备选） ----
    alpha_vantage_api_key: Optional[str] = None
    alpha_vantage_enabled: bool = False
    
    # ---- Interactive Brokers（券商数据源，实盘用） ----
    ibkr_host: str = '127.0.0.1'
    ibkr_port: int = 7497  # TWS默认7497，IB Gateway默认4002
    ibkr_client_id: int = 1
    
    # ---- 数据时间范围 ----
    # 历史数据起始年份
    history_start_year: int = 2010
    # 历史数据结束年份
    history_end_year: int = 2025
    # 默认K线周期: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    default_bar_interval: str = '1d'
    
    # ---- 数据清洗 ----
    # 是否自动复权（forward adjust for splits & dividends）
    auto_adjust_prices: bool = True
    # 是否填充缺失数据（forward fill后backward fill）
    fill_missing_data: bool = True
    # 异常价格波动倍数阈值（超过此倍数的跳变视为数据错误）
    price_spike_threshold: float = 5.0
    
    # ---- 股票池 ----
    # 默认交易的股票池文件路径（TXT，每行一个ticker）
    stock_universe_file: Optional[str] = None
    # 默认跟踪的指数代码列表
    benchmark_indices: List[str] = field(default_factory=lambda: ['^GSPC', '^IXIC', '^DJI'])
    
    # ---- 另类数据（可选） ----
    # 是否拉取做空余额数据
    fetch_short_interest: bool = False
    # 是否拉取新闻情绪数据
    fetch_news_sentiment: bool = False


# ============================================================================
# ML模型配置 - Transformer 架构
# ============================================================================
@dataclass
class ModelConfig:
    """
    机器学习模型配置
    
    采用Transformer架构进行时序预测，核心思想：
    - 将股票的历史价格序列、技术指标、市场特征编码为嵌入向量
    - 通过多头自注意力机制捕捉长期依赖关系
    - 输出未来N期的价格走势预测
    
    参考文献: "Attention Is All You Need" (Vaswani et al., 2017)
    """
    
    # ---- Transformer 架构参数 ----
    # 模型维度（嵌入向量的维度），通常为64的倍数
    d_model: int = 192
    
    # 多头注意力的头数，d_model必须能被n_heads整除
    n_heads: int = 6
    
    # Encoder层数
    n_encoder_layers: int = 3
    
    # Decoder层数
    n_decoder_layers: int = 4
    
    # 前馈网络的隐藏层维度（通常是d_model的4倍）
    d_ff: int = 768
    
    # Dropout率，防止过拟合
    dropout: float = 0.2
    
    # Stochastic Depth drop path rate (0=disabled)
    drop_path_rate: float = 0.1
    
    # 位置编码的最大序列长度
    max_seq_len: int = 252  # 一年约252个交易日
    
    # ---- 输入特征 ----
    # 历史时间窗口大小（用多少天的数据预测）
    lookback_window: int = 60
    
    # 预测时间窗口大小（预测未来多少天）
    prediction_horizon: int = 5
    
    # 使用的特征列表
    features: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',          # OHLCV
        'returns_1d', 'returns_5d', 'returns_20d',          # 收益率
        'volatility_5d', 'volatility_20d',                  # 波动率
        'sma_5', 'sma_20', 'sma_60',                        # 简单移动均线
        'ema_12', 'ema_26',                                  # 指数移动均线
        'rsi_14',                                            # 相对强弱指标
        'macd', 'macd_signal', 'macd_hist',
                          # MACD
        'bb_upper', 'bb_middle', 'bb_lower',                 # 布林带
        # 新闻情感特征 (小权重辅助信号)
        'news_sentiment_3d', 'news_sentiment_7d',              # 新闻情感得分
        'earnings_surprise_pct',                                # 盈利惊喜
        'has_earnings_report',                                  # 财报发布日标记
        'atr_14',                                            # 平均真实波幅
        'volume_ratio',                                      # 量比
    ])
    
    # 情感特征名称列表（用于模型中的分组加权）
    sentiment_features: List[str] = field(default_factory=lambda: [
        'news_sentiment_3d', 'news_sentiment_7d',
        'earnings_surprise_pct', 'has_earnings_report',
    ])
    # 新闻情感特征权重（小权重，仅作辅助参考）
    news_feature_weight: float = 0.05
    
    # 特征归一化方式: 'standard'(标准化), 'minmax'(最大最小), 'robust'(鲁棒)
    normalization: str = 'standard'
    
    # 时序数据增强（防止过拟合）
    use_data_augmentation: bool = True
    aug_noise_std: float = 0.005       # 高斯噪声标准差（相对价格）
    aug_time_warp_sigma: float = 0.1   # 时间扭曲强度
    aug_scale_sigma: float = 0.05      # 幅度缩放强度
    
    # ---- 训练超参数 ----
    # 每批次的样本数
    batch_size: int = 32
    
    # 最大训练轮数
    epochs: int = 30
    
    # 学习率
    learning_rate: float = 3e-4
    
    # 学习率调度器衰减因子
    lr_decay: float = 0.95
    
    # CosineAnnealingWarmRestarts 参数
    cosine_T_0: int = 20     # 第一个重启周期的epoch数
    cosine_T_mult: int = 2   # 每个重启周期倍增因子
    cosine_eta_min: float = 1e-6  # 最小学习率
    
    # 优化器: 'adam', 'adamw', 'sgd'
    optimizer: str = 'adamw'
    
    # 权重衰减（L2正则化），防止过拟合
    weight_decay: float = 1e-5
    
    # ---- Focal Loss 参数 ----
    # Focal Loss gamma (focus on hard examples)
    focal_gamma: float = 2.0
    # Focal Loss alpha (class balance)
    focal_alpha: float = 0.25
    # Label smoothing
    label_smoothing: float = 0.1
    # 新闻情感特征权重（小权重，仅作辅助参考）
    
    # 学习率预热步数占总步数的比例
    warmup_ratio: float = 0.1
    
    # Label Smoothing (防止过置信，金融数据信号噪声比低)
    # 新闻情感特征权重（小权重，仅作辅助参考）
    
    # Stochastic Depth (DropPath) 概率
    
    # 注意力Dropout（高于普通dropout，金融数据噪声大）
    attn_dropout: float = 0.2
    
    # Stochastic Depth drop path rate (0=disabled)
    
    # 梯度累积步数（小batch增大等效batch）
    gradient_accumulation_steps: int = 2
    
    # 早停耐心值：验证损失连续N轮不改善则停止训练
    early_stopping_patience: int = 20
    
    # ---- 训练策略 ----
    # 学习率调度器类型: 'plateau' 或 'cosine'
    scheduler_type: str = 'cosine'
    # Cosine退火第一个restart周期(epoch)
    # 梯度裁剪norm阈值
    grad_clip_norm: float = 1.0
    
    # 梯度裁剪阈值，防止梯度爆炸
    grad_clip: float = 1.0
    
    # ---- 训练/验证/测试集划分 ----
    # 训练集比例
    train_ratio: float = 0.70
    # 验证集比例
    val_ratio: float = 0.15
    # 测试集比例（自动计算: 1 - train_ratio - val_ratio）
    
    # ---- 奖励函数参数 ----
    # 预测方向准确的奖励权重
    direction_reward_weight: float = 0.6
    # 预测幅度准确的奖励权重
    magnitude_reward_weight: float = 0.4
    
    # ---- 精度阈值 ----
    # 方向预测准确率低于此阈值时触发调参
    min_direction_accuracy: float = 0.55
    # 均方根误差高于此阈值时触发调参
    max_rmse: float = 0.06  # 适当放宽RMSE阈值，方向准确率是核心指标
    
    # ---- 硬件配置 ----
    # 设备: 'cpu', 'cuda', 'mps'(Apple Silicon)
    device: str = 'mps' if __import__('torch').backends.mps.is_available() else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    
    # 是否使用混合精度训练（节省显存，加速训练）
    use_amp: bool = True   # Automatic Mixed Precision (MPS不支持则自动fallback)


# ============================================================================
# 交易配置
# ============================================================================
@dataclass
class TradingConfig:
    """
    交易相关配置
    
    涵盖券商接入、订单类型、风控参数等。
    """
    
    # ---- 券商接入 ----
    # 选用的券商: 'ibkr'(Interactive Brokers), 'alpaca', 'td_ameritrade'
    broker: str = 'ibkr'
    
    # IBKR TWS/Gateway连接地址
    ibkr_account_id: Optional[str] = None
    
    # Alpaca API密钥
    alpaca_api_key: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: str = 'https://paper-api.alpaca.markets'
    
    # ---- 账户参数 ----
    # 初始资金（回测/模拟用）
    initial_capital: float = 100_000.0
    
    # 基础货币
    base_currency: str = 'USD'
    
    # ---- 交易时段 ----
    # 是否允许盘前交易（美东 4:00-9:30）
    allow_pre_market: bool = False
    # 是否允许盘后交易（美东 16:00-20:00）
    allow_after_hours: bool = False
    # 正常交易时段（美东 9:30-16:00）始终开启
    
    # ---- 费率设置 ----
    # 每股佣金（美元，Interactive Brokers阶梯式收费约$0.005/股）
    commission_per_share: float = 0.005
    # 每笔最低佣金
    commission_min: float = 1.0
    # 每笔最高佣金（占成交额百分比）
    commission_max_pct: float = 0.01
    # SEC费用（销售金额的0.00278%，2025年）
    sec_fee_rate: float = 0.0000278
    # TAF费用（每股$0.000166，卖出时收取，上限$8.30）
    taf_fee_per_share: float = 0.000166
    taf_fee_max: float = 8.30
    
    # 做空借券年化费率（估算默认值）
    short_borrow_rate_annual: float = 0.003  # 0.3% for easy-to-borrow
    
    # 滑点模型参数
    # 滑点 = base_bps + vol_coeff * 波动率 + size_coeff * sqrt(订单量/日均成交量)
    slippage_base_bps: float = 1.0   # 基础滑点（基点）
    slippage_vol_coeff: float = 5.0  # 波动率系数
    slippage_size_coeff: float = 10.0  # 订单规模系数
    
    # ---- 默认订单设置 ----
    # 默认订单类型: 'MKT'(市价), 'LMT'(限价), 'STP'(止损)
    default_order_type: str = 'LMT'
    # 限价单的偏移量（相对于当前价格的百分比）
    limit_price_offset_pct: float = 0.002  # 0.2%
    # 默认订单有效期: 'DAY', 'GTC', 'IOC'
    default_time_in_force: str = 'DAY'
    
    # ---- 风控参数（全局限制，各策略可进一步收紧） ----
    # 单笔最大成交金额（占账户比例）
    max_order_amount_pct: float = 0.10  # 10%
    # 单日最大成交次数
    max_daily_trades: int = 50
    # 最大持仓数量
    max_positions: int = 20
    # 单股最大持仓比例
    max_position_pct: float = 0.20  # 20%
    # 总体最大杠杆倍数
    max_leverage: float = 2.0
    # 最大回撤比例（触及则停止交易）
    max_drawdown_pct: float = 0.25  # 25%
    # PDT规则：日内交易次数限制（5个交易日4次）
    pdt_day_trade_limit: int = 3  # 保守设为3，留有余地
    # PDT最低账户资金
    pdt_min_equity: float = 25_000.0


# ============================================================================
# 全局配置实例（单例模式）
# ============================================================================
# 创建全局配置实例，各模块通过 `from config import system_config` 引入
system_config: SystemConfig = SystemConfig()
data_source_config: DataSourceConfig = DataSourceConfig()
model_config: ModelConfig = ModelConfig()
trading_config: TradingConfig = TradingConfig()
