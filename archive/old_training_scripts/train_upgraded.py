#!/usr/bin/env python3
"""训练升级版 Transformer (Learnable PE + GELU + DropPath + Focal Loss)"""
import sys, os, time, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import model_config, system_config
from config.logging_config import setup_logging

model_config.device = 'cpu'
setup_logging(system_config)

logger = logging.getLogger('train_upgraded')
logger.info('=' * 70)
logger.info('  UPGRADED Transformer Training')
logger.info(f'  d_model={model_config.d_model}, heads={model_config.n_heads}, layers={model_config.n_encoder_layers}')
logger.info(f'  d_ff={model_config.d_ff}, dropout={model_config.dropout}, drop_path={model_config.drop_path_rate}')
logger.info(f'  lookback={model_config.lookback_window}, horizon={model_config.prediction_horizon}')
logger.info(f'  Focal Loss gamma={model_config.focal_gamma}, scheduler={model_config.scheduler_type}')
logger.info('=' * 70)

from ml_model.trainer import ModelTrainer
from ml_model.data_loader import prepare_data

logger.info('Loading data...')
train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)

if train_loader is None:
    logger.error('No training data!')
    sys.exit(1)

trainer = ModelTrainer(model_config)
logger.info(f'Training on {model_config.device}, {len(train_loader)} batches')

start = time.perf_counter()
metrics = trainer.train(train_loader, val_loader, epochs=model_config.epochs)
duration = time.perf_counter() - start

logger.info(f'Training done in {duration/60:.1f} min')

result = trainer.evaluate(test_loader)

print('\n' + '=' * 60)
print('  RESULTS')
print('=' * 60)
print(f'  Direction Acc: {result.direction_accuracy:.4f} ({result.direction_accuracy:.2%})')
print(f'  RMSE: {result.rmse:.6f}')
print(f'  MAE:  {result.mae:.6f}')
print(f'  Sharpe: {result.sharpe_ratio:.4f}')
print(f'  R^2:   {result.r_squared:.4f}')
print(f'  Acceptable: {"YES" if result.is_acceptable else "NO"}')
print(f'  Best epoch: {metrics.best_epoch + 1}/{model_config.epochs}')
print(f'  Best val loss: {metrics.best_val_loss:.6f}')
print('=' * 60)

model_path = trainer.save_model('transformer_upgraded_v1')
logger.info(f'Saved: {model_path}')
