#!/usr/bin/env python3
"""CPU-optimized upgraded Transformer training.
Preserves all algo improvements: Learnable PE, GELU, DropPath, Focal Loss, Cosine.
"""
import sys, os, time, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Override config for CPU efficiency while keeping algo improvements
from config.settings import model_config, system_config
from config.logging_config import setup_logging

model_config.device = 'cpu'
model_config.d_model = 256       # CPU-friendly
model_config.n_heads = 8
model_config.n_encoder_layers = 4
model_config.d_ff = 1024
model_config.dropout = 0.2
model_config.drop_path_rate = 0.1
model_config.lookback_window = 60
model_config.batch_size = 16
model_config.learning_rate = 3e-4
model_config.epochs = 40
model_config.early_stopping_patience = 15
model_config.scheduler_type = 'cosine'
model_config.grad_clip_norm = 1.0
# Focal Loss params
model_config.focal_gamma = 2.0
model_config.focal_alpha = 0.25
model_config.label_smoothing = 0.1

setup_logging(system_config)
logger = logging.getLogger('train_upgraded_cpu')
logger.info('=' * 70)
logger.info('  UPGRADED Transformer CPU Training')
logger.info(f'  Arch: d_model={model_config.d_model}, heads={model_config.n_heads}, layers={model_config.n_encoder_layers}')
logger.info(f'  d_ff={model_config.d_ff}, dropout={model_config.dropout}, drop_path={model_config.drop_path_rate}')
logger.info(f'  Algo: Learnable PE, GELU, DropPath, Attention Pooling, Kaiming Init')
logger.info(f'  Loss: Focal(gamma={model_config.focal_gamma}) + LabelSmooth({model_config.label_smoothing})')
logger.info(f'  Scheduler: {model_config.scheduler_type}, grad_clip={model_config.grad_clip_norm}')
logger.info(f'  Batch={model_config.batch_size}, LR={model_config.learning_rate}, Epochs={model_config.epochs}')
logger.info('=' * 70)

from ml_model.trainer import ModelTrainer
from ml_model.data_loader import prepare_data

logger.info('Loading data...')
train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)

if train_loader is None:
    logger.error('No training data!')
    sys.exit(1)

trainer = ModelTrainer(model_config)
logger.info(f'Training on {model_config.device}, {len(train_loader)} batches/epoch')

start = time.perf_counter()
metrics = trainer.train(train_loader, val_loader, epochs=model_config.epochs)
duration = time.perf_counter() - start

logger.info(f'Training done in {duration/60:.1f} min')

logger.info('Evaluating on test set...')
result = trainer.evaluate(test_loader)

print('\n' + '=' * 60)
print('  FINAL RESULTS (Upgraded Transformer)')
print('=' * 60)
print(f'  Direction Accuracy: {result.direction_accuracy:.4f} ({result.direction_accuracy:.2%})')
print(f'  RMSE:              {result.rmse:.6f}')
print(f'  MAE:               {result.mae:.6f}')
print(f'  Sharpe Ratio:      {result.sharpe_ratio:.4f}')
print(f'  R-squared:         {result.r_squared:.4f}')
print(f'  Acceptable:        {"YES" if result.is_acceptable else "NO"}')
print(f'  Test Samples:      {result.total_samples}')
print(f'  Best Epoch:        {metrics.best_epoch + 1}/{model_config.epochs}')
print(f'  Best Val Loss:     {metrics.best_val_loss:.6f}')
print(f'  Training Time:     {duration/60:.1f} min')
print('=' * 60)

model_path = trainer.save_model('transformer_upgraded_cpu_v1')
logger.info(f'Model saved: {model_path}')
