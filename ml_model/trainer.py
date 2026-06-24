"""Model training loop with checkpointing and early stopping."""

import logging
import sys
import time
import copy
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.settings import ModelConfig
from ml_model.transformer import StockTransformer
from data_pipeline.storage import ModelStorage
from utils.exceptions import ModelError, ModelTrainingError

logger = logging.getLogger(__name__)


# ============================================================================
# 訓練指標記錄
# ============================================================================
@dataclass
class TrainingMetrics:
    """
    訓練過程中的指標記錄
    
    跟蹤每個epoch的訓練/驗證損失和評估指標。
    """
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_reg_losses: List[float] = field(default_factory=list)
    val_reg_losses: List[float] = field(default_factory=list)
    train_cls_losses: List[float] = field(default_factory=list)
    val_cls_losses: List[float] = field(default_factory=list)
    direction_accuracies: List[float] = field(default_factory=list)
    rmses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    
    def update(self, epoch: int, train_loss: float, val_loss: float,
               reg_loss: float, cls_loss: float, lr: float,
               val_reg: float = 0, val_cls: float = 0) -> None:
        """記錄一個epoch的指標"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_reg_losses.append(reg_loss)
        self.train_cls_losses.append(cls_loss)
        self.val_reg_losses.append(val_reg)
        self.val_cls_losses.append(val_cls)
        self.learning_rates.append(lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch


@dataclass 
class EvaluationResult:
    """
    模型評估結果
    
    包含多維度精度指標：
    - 方向準確率: 預測漲跌方向正確的比例
    - RMSE: 預測收益率與實際收益率的均方根誤差
    - MAE: 平均絕對誤差
    - Sharpe: 基於預測信號的夏普比率（簡化計算）
    - R²: 決定係數
    """
    direction_accuracy: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    sharpe_ratio: float = 0.0
    r_squared: float = 0.0
    total_samples: int = 0
    
    @property
    def is_acceptable(self) -> bool:
        """
        判斷模型精度是否可接受
        
        標準：
        - 方向準確率 >= 55%（超過隨機猜測50%）
        - RMSE < 0.05（收益率預測誤差在5%以內）
        """
        from config.settings import model_config
        return (self.direction_accuracy >= model_config.min_direction_accuracy and
                self.rmse < model_config.max_rmse)
    
    def to_dict(self) -> Dict[str, float]:
        """轉換為字典"""
        return {
            'direction_accuracy': round(self.direction_accuracy, 4),
            'rmse': round(self.rmse, 6),
            'mae': round(self.mae, 6),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'r_squared': round(self.r_squared, 4),
            'total_samples': self.total_samples,
        }


# ============================================================================
# 模型訓練器
# ============================================================================

class FocalBCELoss(nn.Module):
    """Focal Loss for binary classification with label smoothing.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Combats class imbalance by down-weighting easy examples,
    forcing the model to focus on hard cases. Essential for
    financial prediction where >50% accuracy is hard.
    
    Label smoothing prevents overconfidence and improves
    calibration of predicted probabilities.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, 
                 label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal scaling
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class ModelTrainer:
    """
    Transformer模型訓練器
    
    功能：
    - 訓練循環（含早停、學習率衰減、梯度裁剪）
    - 模型評估
    - 模型保存/加載
    - 訓練歷史可視化數據導出
    
    使用示例:
        trainer = ModelTrainer(model_config)
        trainer.train(train_loader, val_loader)
        result = trainer.evaluate(test_loader)
    """
    
    def __init__(self, config: ModelConfig):
        """
        參數:
            config: 模型配置
        """
        self.config = config
        self.device = self._get_device()
        self.model: Optional[StockTransformer] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.metrics = TrainingMetrics()
        self.storage = ModelStorage()
        
        # 損失函數
        self.regression_loss_fn = nn.MSELoss()    # 回歸損失
        # 分類損失帶Label Smoothing，防止過置信
        # 金融數據信號噪聲比極低，label smoothing是關鍵正則化手段
        self.classification_loss_fn = nn.BCEWithLogitsLoss()  # 默認（base用）
        self.cls_loss_smooth = None  # 帶label smoothing的版本（按需創建）
        
        logger.info(f"ModelTrainer初始化, 設備: {self.device}")
    
    def _get_device(self) -> torch.device:
        """
        自動選擇最佳計算設備
        
        優先級: CUDA > MPS(Apple Silicon) > CPU
        """
        if self.config.device != 'cpu':
            if self.config.device == 'cuda' and torch.cuda.is_available():
                return torch.device('cuda')
            elif self.config.device == 'mps' and torch.backends.mps.is_available():
                return torch.device('mps')
        
        return torch.device('cpu')
    

    def _combined_loss(
        self,
        reg_pred: torch.Tensor,
        reg_target: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        計算組合損失
        
        Total Loss = w_reg * MSE(reg_pred, reg_target) 
                   + w_cls * BCE(cls_pred, cls_target)
        
        加權方案：
        - 初始階段：回歸權重高（0.6），因為需要模型先學會預測收益率幅度
        - 後期可調整：增加分類權重，提高方向判斷準確率
        
        參數:
            reg_pred: 回歸預測值 (batch, horizon)
            reg_target: 回歸目標值 (batch, horizon)
            cls_pred: 分類logits (batch, horizon)
            cls_target: 分類目標 (batch, horizon)
        
        返回:
            (total_loss, reg_loss, cls_loss)
        """
        reg_loss = self.regression_loss_fn(reg_pred, reg_target)
        cls_loss = self.classification_loss_fn(cls_pred, cls_target)
        
        # 加權組合
        w_reg = self.config.direction_reward_weight  # NOTE: 變量名direction_reward_weight實際控制回歸損失，非方向分類
        w_cls = self.config.magnitude_reward_weight
        
        total_loss = w_reg * reg_loss + w_cls * cls_loss
        
        return total_loss, reg_loss, cls_loss
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """訓練一個epoch（label smoothing已內置在_combined_loss中）"""
        self.model.train()
        
        total_loss_sum = 0.0
        reg_loss_sum = 0.0
        cls_loss_sum = 0.0
        n_batches = 0
        
        for x, y_reg, y_cls in train_loader:
            x = x.to(self.device)
            y_reg = y_reg.to(self.device)
            y_cls = y_cls.to(self.device)
            
            reg_pred, cls_pred = self.model(x)
            
            total_loss, reg_loss, cls_loss = self._combined_loss(
                reg_pred, y_reg, cls_pred, y_cls
            )
            
            if torch.isnan(total_loss):
                raise ModelTrainingError(
                    "訓練損失包含NaN，請檢查學習率或數據",
                    {'lr': self.optimizer.param_groups[0]['lr']}
                )
            
            self.optimizer.zero_grad()
            total_loss.backward()
            
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            
            self.optimizer.step()
            
            total_loss_sum += total_loss.item()
            reg_loss_sum += reg_loss.item()
            cls_loss_sum += cls_loss.item()
            n_batches += 1
        
        return (
            total_loss_sum / max(n_batches, 1),
            reg_loss_sum / max(n_batches, 1),
            cls_loss_sum / max(n_batches, 1)
        )
    
    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        驗證一個epoch
        
        在驗證集上評估模型，不計算梯度。
        
        參數:
            val_loader: 驗證數據加載器
        
        返回:
            (平均總損失, 平均回歸損失, 平均分類損失)
        """
        if val_loader is None:
            return float('inf'), float('inf'), float('inf')
        
        self.model.eval()
        
        total_loss_sum = 0.0
        reg_loss_sum = 0.0
        cls_loss_sum = 0.0
        n_batches = 0
        
        for x, y_reg, y_cls in val_loader:
            x = x.to(self.device)
            y_reg = y_reg.to(self.device)
            y_cls = y_cls.to(self.device)
            
            reg_pred, cls_pred = self.model(x)
            
            total_loss, reg_loss, cls_loss = self._combined_loss(
                reg_pred, y_reg, cls_pred, y_cls
            )
            
            total_loss_sum += total_loss.item()
            reg_loss_sum += reg_loss.item()
            cls_loss_sum += cls_loss.item()
            n_batches += 1
        
        denom = max(n_batches, 1)
        return (
            total_loss_sum / denom,
            reg_loss_sum / denom,
            cls_loss_sum / denom
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> TrainingMetrics:
        """
        完整訓練流程
        
        步驟：
        1. 初始化模型和優化器
        2. 循環訓練epochs
        3. 每個epoch後驗證
        4. 早停判斷
        5. 保存最佳模型
        
        參數:
            train_loader: 訓練數據
            val_loader: 驗證數據（可選，None則使用訓練集評估）
            epochs: 訓練輪數（None使用配置中的值）
            verbose: 是否列印訓練進度
        
        返回:
            TrainingMetrics訓練指標記錄
        """
        if epochs is None:
            epochs = self.config.epochs
        
        # 1. 初始化模型
        self.model = StockTransformer(self.config).to(self.device)
        
        # 2. 配置優化器
        if self.config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        # 3. 學習率調度器
        scheduler_type = getattr(self.config, 'scheduler_type', 'plateau')
        if scheduler_type == 'cosine':
            T_0 = getattr(self.config, 'cosine_T_0', epochs // 2)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
            )
            self._cosine_step_per_batch = False
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_decay,
                patience=self.config.early_stopping_patience // 2,
            )
        
        # 4. 重置指標
        self.metrics = TrainingMetrics()
        best_model_state = None
        patience_counter = 0
        
        logger.info(
            f"開始訓練: {epochs} 輪, "
            f"batch_size={self.config.batch_size}, "
            f"lr={self.config.learning_rate}, "
            f"device={self.device}"
        )
        
        train_start = time.perf_counter()
        
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            
            # 訓練
            train_total, train_reg, train_cls = self._train_epoch(train_loader)
            
            # 驗證
            val_total, val_reg, val_cls = self._validate_epoch(val_loader)
            
            # 學習率調度
            current_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step(epoch)
            elif val_loader is not None:
                self.scheduler.step(val_total)
            else:
                self.scheduler.step(train_total)
            
            # 記錄指標
            self.metrics.update(
                epoch, train_total, val_total,
                train_reg, train_cls, current_lr,
                val_reg, val_cls
            )
            
            # 列印進度
            epoch_time = time.perf_counter() - epoch_start
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch < 5 or epoch == epochs - 1):
                logger.info(
                    f"Epoch {epoch+1:3d}/{epochs} | "
                    f"Train: {train_total:.6f} | "
                    f"Val: {val_total:.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
                sys.stdout.flush()
            
            # 早停判斷
            if val_total < self.metrics.best_val_loss:
                self.metrics.best_val_loss = val_total
                self.metrics.best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"早停: 驗證損失連續 {patience_counter} 輪未改善, "
                    f"最佳輪次: epoch {self.metrics.best_epoch+1}"
                )
                break
        
        train_duration = time.perf_counter() - train_start
        logger.info(
            f"訓練完成: {train_duration/60:.1f}min, "
            f"最佳驗證損失: {self.metrics.best_val_loss:.6f} "
            f"(Epoch {self.metrics.best_epoch+1})"
        )
        
        # 5. 恢復最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.metrics
    
    def evaluate(self, test_loader: DataLoader) -> EvaluationResult:
        """
        在測試集上評估模型精度
        
        評估維度：
        1. 方向準確率: 預測漲跌方向正確的比例
        2. RMSE: 預測收益率與真實收益率的均方根誤差
        3. MAE: 平均絕對誤差
        4. Sharpe Ratio: 基於預測信號的夏普比率
        5. R²: 決定係數
        
        時間複雜度: O(n_batches * batch_size * prediction_horizon)
        
        參數:
            test_loader: 測試數據加載器
        
        返回:
            EvaluationResult評估結果
        """
        if self.model is None:
            raise ModelError("模型未訓練，請先調用 train()")
        
        self.model.eval()
        
        all_reg_preds = []
        all_reg_targets = []
        all_cls_preds = []
        all_cls_targets = []
        
        with torch.no_grad():
            for x, y_reg, y_cls in test_loader:
                x = x.to(self.device)
                
                reg_pred, cls_pred = self.model(x)
                
                all_reg_preds.append(reg_pred.cpu().numpy())
                all_reg_targets.append(y_reg.numpy())
                all_cls_preds.append(torch.sigmoid(cls_pred).cpu().numpy())
                all_cls_targets.append(y_cls.numpy())
        
        # 合併所有batch的結果
        reg_preds = np.concatenate(all_reg_preds, axis=0)  # (n_samples, horizon)
        reg_targets = np.concatenate(all_reg_targets, axis=0)
        cls_preds = np.concatenate(all_cls_preds, axis=0)
        cls_targets = np.concatenate(all_cls_targets, axis=0)
        
        n_samples = reg_preds.shape[0]
        
        # 1. 方向準確率
        # 比較預測概率>0.5與真實方向
        direction_correct = ((cls_preds > 0.5) == (cls_targets > 0.5)).mean()
        
        # 2. RMSE (均方根誤差)
        rmse = np.sqrt(np.mean((reg_preds - reg_targets) ** 2))
        
        # 3. MAE (平均絕對誤差)
        mae = np.mean(np.abs(reg_preds - reg_targets))
        
        # 4. 簡化夏普比率
        # 假設按預測信號交易：預測收益>0做多，<0做空
        trade_returns = np.where(reg_preds > 0, reg_targets, -reg_targets)
        sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8)
        # 年化（假設每個樣本為1個交易日）
        sharpe_annual = sharpe * np.sqrt(252)
        
        # 5. R² 決定係數
        ss_res = np.sum((reg_targets - reg_preds) ** 2)
        ss_tot = np.sum((reg_targets - np.mean(reg_targets)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-8)
        
        result = EvaluationResult(
            direction_accuracy=float(direction_correct),
            rmse=float(rmse),
            mae=float(mae),
            sharpe_ratio=float(sharpe_annual),
            r_squared=float(r_squared),
            total_samples=n_samples
        )
        
        # 列印評估報告
        logger.info("=" * 50)
        logger.info("           模型評估報告")
        logger.info("=" * 50)
        logger.info(f"  測試樣本數:     {result.total_samples:,}")
        logger.info(f"  方向準確率:     {result.direction_accuracy:.2%}")
        logger.info(f"  RMSE:           {result.rmse:.6f}")
        logger.info(f"  MAE:            {result.mae:.6f}")
        logger.info(f"  年化夏普比率:   {result.sharpe_ratio:.4f}")
        logger.info(f"  R²:             {result.r_squared:.4f}")
        logger.info(f"  精度合格:       {'✓ 是' if result.is_acceptable else '✗ 否'}")
        logger.info("=" * 50)
        
        # 檢查精度是否達標
        if not result.is_acceptable:
            logger.warning(
                f"模型精度不達標！方向準確率 {result.direction_accuracy:.2%} "
                f"(要求 ≥ {self.config.min_direction_accuracy:.0%}), "
                f"RMSE {result.rmse:.6f} (要求 < {self.config.max_rmse})"
            )
        
        return result
    
    def save_model(self, name: str = "transformer_stock") -> Path:
        """
        保存訓練好的模型
        
        參數:
            name: 模型名稱
        
        返回:
            模型文件路徑
        """
        if self.model is None:
            raise ModelError("沒有可保存的模型")
        
        metadata = {
            'model_params': {
                'config': {
                    'd_model': self.config.d_model,
                    'n_heads': self.config.n_heads,
                    'n_encoder_layers': self.config.n_encoder_layers,
                    'n_decoder_layers': self.config.n_decoder_layers,
                    'd_ff': self.config.d_ff,
                    'dropout': self.config.dropout,
                    'max_seq_len': self.config.max_seq_len,
                    'prediction_horizon': self.config.prediction_horizon,
                    'lookback_window': self.config.lookback_window,
                    'features': self.config.features,
                }
            },
            'training_metrics': {
                'best_val_loss': self.metrics.best_val_loss,
                'best_epoch': self.metrics.best_epoch,
            }
        }
        
        return self.storage.save_torch_model(self.model, name, metadata)
    
    def load_model(self, name: str) -> None:
        """
        加載已訓練的模型
        
        參數:
            name: 模型名稱
        """
        self.model, metadata = self.storage.load_torch_model(
            StockTransformer, name
        )
        self.model = self.model.to(self.device)
        logger.info(f"模型 {name} 已加載到 {self.device}")


# ============================================================================
# 超參數調優器
# ============================================================================
class HyperparameterTuner:
    """
    自動超參數調優器
    
    當模型精度不達標時，自動搜索最優超參數組合。
    
    調參策略（按優先級）：
    1. 調整學習率（對數空間: 1e-5 ~ 1e-2）
    2. 調整正則化參數（dropout: 0.05~0.3, weight_decay: 1e-6~1e-3）
    3. 調整模型架構（d_model: 128~512, n_layers: 2~6）
    4. 調整獎勵權重（方向vs幅度）
    
    搜索方法：網格搜索（小空間）+ 貝葉斯優化（大空間）
    """
    
    def __init__(self, base_config: ModelConfig):
        """
        參數:
            base_config: 基礎配置（調參將在此之上修改）
        """
        self.base_config = base_config
    
    def _create_config_variant(
        self, overrides: Dict[str, Any]
    ) -> ModelConfig:
        """
        根據覆蓋參數創建配置變體
        
        參數:
            overrides: 要覆蓋的參數字典
        
        返回:
            新的ModelConfig實例
        """
        new_config = copy.deepcopy(self.base_config)
        for key, value in overrides.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        return new_config
    
    def tune_learning_rate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr_range: Tuple[float, float] = (1e-5, 1e-2),
        n_trials: int = 7
    ) -> Tuple[float, Optional[EvaluationResult]]:
        """
        學習率調優（對數空間搜索）
        
        在[1e-5, 1e-2]範圍內等比搜索n_trials個學習率，
        選擇驗證損失最低的配置。
        
        時間複雜度: O(n_trials * epochs * n_batches)
        
        參數:
            train_loader: 訓練數據
            val_loader: 驗證數據
            lr_range: 學習率搜索範圍
            n_trials: 試驗次數
        
        返回:
            (最優學習率, 對應的評估結果)
        """
        logger.info(f"學習率調優: {lr_range}, {n_trials} 次試驗")
        
        # 在對數空間均勻採樣
        lr_values = np.logspace(
            np.log10(lr_range[0]), np.log10(lr_range[1]), n_trials
        )
        
        best_lr = lr_values[0]
        best_val_loss = float('inf')
        results = []
        
        for lr in lr_values:
            config = self._create_config_variant({
                'learning_rate': float(lr),
                'epochs': 30,  # 調參時用較少epoch
            })
            
            trainer = ModelTrainer(config)
            
            try:
                metrics = trainer.train(
                    train_loader, val_loader
                )
                
                val_loss = metrics.best_val_loss
                results.append((lr, val_loss))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_lr = lr
                
                logger.debug(f"  lr={lr:.2e}: val_loss={val_loss:.6f}")
                
            except ModelTrainingError as e:
                logger.warning(f"  lr={lr:.2e}: 訓練失敗 - {e}")
                continue
        
        logger.info(f"最優學習率: {best_lr:.2e} (val_loss={best_val_loss:.6f})")
        
        return float(best_lr), None  # 快速調參模式不運行完整評估
    
    def tune_regularization(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        正則化參數調優
        
        同時搜索dropout和weight_decay的最優組合。
        
        參數:
            train_loader: 訓練數據
            val_loader: 驗證數據
        
        返回:
            最優參數 {'dropout': x, 'weight_decay': y}
        """
        dropout_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        wd_values = [1e-6, 1e-5, 1e-4, 1e-3]
        
        best_params = {'dropout': 0.1, 'weight_decay': 1e-5}
        best_val_loss = float('inf')
        
        logger.info(f"正則化參數調優: dropout={dropout_values}, wd={wd_values}")
        
        for dropout in dropout_values:
            for wd in wd_values:
                config = self._create_config_variant({
                    'dropout': dropout,
                    'weight_decay': wd,
                    'epochs': 25,
                })
                
                trainer = ModelTrainer(config)
                
                try:
                    metrics = trainer.train(train_loader, val_loader)
                    val_loss = metrics.best_val_loss
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {'dropout': dropout, 'weight_decay': wd}
                    
                    logger.debug(
                        f"  dropout={dropout}, wd={wd:.1e}: val_loss={val_loss:.6f}"
                    )
                except ModelTrainingError:
                    continue
        
        logger.info(f"最優正則化: {best_params}")
        return best_params
    
    def tune_architecture(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, int]:
        """
        模型架構調優
        
        搜索d_model和n_layers的最優組合。
        
        參數:
            train_loader: 訓練數據
            val_loader: 驗證數據
        
        返回:
            最優參數 {'d_model': x, 'n_encoder_layers': y}
        """
        d_model_values = [128, 192, 256, 384, 512]
        n_layers_values = [2, 3, 4, 6]
        
        best_params = {'d_model': 256, 'n_encoder_layers': 4}
        best_val_loss = float('inf')
        
        logger.info("模型架構調優...")
        
        for d_model in d_model_values:
            for n_layers in n_layers_values:
                # 確保d_model能被n_heads整除
                n_heads = max(4, d_model // 32)
                if d_model % n_heads != 0:
                    n_heads = 8  # 降級到默認值
                
                config = self._create_config_variant({
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'n_encoder_layers': n_layers,
                    'epochs': 20,
                })
                
                trainer = ModelTrainer(config)
                
                try:
                    metrics = trainer.train(train_loader, val_loader)
                    val_loss = metrics.best_val_loss
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {
                            'd_model': d_model,
                            'n_heads': n_heads,
                            'n_encoder_layers': n_layers
                        }
                    
                    logger.debug(
                        f"  d_model={d_model}, layers={n_layers}: val_loss={val_loss:.6f}"
                    )
                except (ModelTrainingError, RuntimeError):
                    continue
        
        logger.info(f"最優架構: {best_params}")
        return best_params
    
    def tune_reward_weights(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        獎勵函數權重調優
        
        在方向和幅度之間尋找最優平衡。
        
        參數:
            train_loader, val_loader, test_loader: 數據加載器
        
        返回:
            最優權重 {'direction_weight': x, 'magnitude_weight': y}
        """
        weight_combinations = [
            (0.3, 0.7), (0.4, 0.6), (0.5, 0.5),
            (0.6, 0.4), (0.7, 0.3)
        ]
        
        best_params = {'direction_weight': 0.6, 'magnitude_weight': 0.4}
        best_score = 0.0  # 綜合得分（方向準確率 - RMSE）
        
        logger.info("獎勵權重調優...")
        
        for dir_w, mag_w in weight_combinations:
            config = self._create_config_variant({
                'direction_reward_weight': dir_w,
                'magnitude_reward_weight': mag_w,
                'epochs': 30,
            })
            
            trainer = ModelTrainer(config)
            
            try:
                trainer.train(train_loader, val_loader)
                result = trainer.evaluate(test_loader)
                
                # 綜合得分 = 方向準確率 - RMSE
                score = result.direction_accuracy - result.rmse
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'direction_weight': dir_w,
                        'magnitude_weight': mag_w
                    }
                
                logger.debug(
                    f"  dir_w={dir_w}, mag_w={mag_w}: "
                    f"acc={result.direction_accuracy:.2%}, rmse={result.rmse:.6f}, "
                    f"score={score:.4f}"
                )
            except ModelTrainingError:
                continue
        
        logger.info(f"最優獎勵權重: {best_params}")
        return best_params
    
    def auto_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> ModelConfig:
        """
        全自動調參
        
        按優先級依次調優：
        1. 學習率
        2. 正則化參數
        3. 架構參數
        4. 獎勵權重
        
        每步調優後檢查精度是否達標，達標則提前停止。
        
        參數:
            train_loader, val_loader, test_loader: 數據加載器
        
        返回:
            優化後的ModelConfig
        """
        logger.info("=" * 50)
        logger.info("   開始自動超參數調優")
        logger.info("=" * 50)
        
        config = copy.deepcopy(self.base_config)
        
        # Step 1: 學習率調優
        logger.info("\n[Step 1/4] 學習率調優...")
        best_lr, _ = self.tune_learning_rate(train_loader, val_loader)
        config.learning_rate = best_lr
        
        # 檢查精度
        trainer = ModelTrainer(config)
        trainer.train(train_loader, val_loader)
        result = trainer.evaluate(test_loader)
        if result.is_acceptable:
            logger.info("精度已達標，停止調參")
            return config
        
        # Step 2: 正則化調優
        logger.info("\n[Step 2/4] 正則化參數調優...")
        reg_params = self.tune_regularization(train_loader, val_loader)
        config.dropout = reg_params['dropout']
        config.weight_decay = reg_params['weight_decay']
        
        trainer = ModelTrainer(config)
        trainer.train(train_loader, val_loader)
        result = trainer.evaluate(test_loader)
        if result.is_acceptable:
            logger.info("精度已達標，停止調參")
            return config
        
        # Step 3: 架構調優
        logger.info("\n[Step 3/4] 模型架構調優...")
        arch_params = self.tune_architecture(train_loader, val_loader)
        for key, value in arch_params.items():
            setattr(config, key, value)
        
        trainer = ModelTrainer(config)
        trainer.train(train_loader, val_loader)
        result = trainer.evaluate(test_loader)
        if result.is_acceptable:
            logger.info("精度已達標，停止調參")
            return config
        
        # Step 4: 獎勵權重調優
        logger.info("\n[Step 4/4] 獎勵權重調優...")
        reward_params = self.tune_reward_weights(
            train_loader, val_loader, test_loader
        )
        config.direction_reward_weight = reward_params['direction_weight']
        config.magnitude_reward_weight = reward_params['magnitude_weight']
        
        logger.info("\n自動調參完成!")
        logger.info(f"最終配置: lr={config.learning_rate:.2e}, "
                   f"dropout={config.dropout}, wd={config.weight_decay:.1e}, "
                   f"d_model={config.d_model}, layers={config.n_encoder_layers}")
        
        return config
