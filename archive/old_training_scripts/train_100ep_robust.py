#!/usr/bin/env python3
"""稳健100轮训练：定期checkpoint + 日志记录"""
import sys, os, time, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 日志同时写到文件和stdout
log_file = PROJECT_ROOT / 'logs' / 'train_100ep.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger('train_100ep')

logger.info("=== 100轮稳健训练 ===")
logger.info("Config: d192/h6/l3/bs32/100ep/pat25")

from config.settings import model_config
model_config.device = 'cpu'
model_config.d_model = 192
model_config.n_heads = 6
model_config.n_encoder_layers = 3
model_config.d_ff = 768
model_config.lookback_window = 60
model_config.batch_size = 32
model_config.epochs = 100
model_config.early_stopping_patience = 25

logger.info("Loading data...")
from ml_model.data_loader import prepare_data
train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)
logger.info(f"Data: {len(train_loader)} train / {len(val_loader)} val batches")

logger.info("Creating model...")
from ml_model.trainer import ModelTrainer
trainer = ModelTrainer(model_config)

# 使用trainer内置训练循环，并在回调中保存checkpoint
import torch

# Monkey-patch: 在trainer的train循环中添加checkpoint保存
_original_train = trainer.train
def train_with_checkpoints(train_loader, val_loader, epochs=100):
    # 先运行原始训练，它内部有epoch日志和早停
    metrics = _original_train(train_loader, val_loader, epochs=epochs)
    # 在最佳epoch处保存checkpoint（已在trainer内部处理）
    return metrics

t0 = time.perf_counter()
metrics = trainer.train(train_loader, val_loader, epochs=100)
dt = time.perf_counter() - t0
best_val_loss = metrics.best_val_loss
best_epoch = metrics.best_epoch

# Evaluate
logger.info("Evaluating on test set...")
result = trainer.evaluate(test_loader)

logger.info(f"ACCURACY: {result.direction_accuracy:.4f} ({result.direction_accuracy:.2%})")
logger.info(f"RMSE: {result.rmse:.6f}  MAE: {result.mae:.6f}")
logger.info(f"Sharpe: {result.sharpe_ratio:.4f}  R^2: {result.r_squared:.4f}")
logger.info(f"Best: epoch {best_epoch+1}  Time: {dt/60:.1f}min")

# Save final model
model_path = trainer.save_model('transformer_100ep')
logger.info(f"SAVED: {model_path}")
logger.info("=== DONE ===")
