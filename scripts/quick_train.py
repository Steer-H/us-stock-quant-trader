#!/usr/bin/env python3
"""Quick training test with explicit flush for monitoring."""
import sys, os, time, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup unbuffered logging to file
log_file = PROJECT_ROOT / 'logs' / 'quick_train.log'
# Use unbuffered file for real-time log monitoring
log_fd = open(str(log_file), 'w', buffering=1)  # line-buffered
file_handler = logging.StreamHandler(log_fd)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)-20s | %(message)s'))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)-20s | %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler],
    force=True
)
logger = logging.getLogger('quick_train')

from config.settings import model_config
model_config.device = 'cpu'
model_config.d_model = 192
model_config.n_heads = 6
model_config.n_encoder_layers = 3
model_config.d_ff = 768
model_config.dropout = 0.2
model_config.drop_path_rate = 0.1
model_config.lookback_window = 60
model_config.batch_size = 32
model_config.epochs = 15
model_config.early_stopping_patience = 10
model_config.scheduler_type = 'cosine'
model_config.grad_clip_norm = 1.0

logger.info(f'Config: d_model={model_config.d_model}, heads={model_config.n_heads}, layers={model_config.n_encoder_layers}')
logger.info(f'Features: {len(model_config.features)}, lookback={model_config.lookback_window}, batch={model_config.batch_size}')

from ml_model.trainer import ModelTrainer
from ml_model.data_loader import prepare_data

logger.info('Loading data...')
train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)
logger.info(f'Data: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test batches')

trainer = ModelTrainer(model_config)
logger.info('Starting training...')
t0 = time.perf_counter()
metrics = trainer.train(train_loader, val_loader, epochs=model_config.epochs)
dt = time.perf_counter() - t0
logger.info(f'Training done in {dt/60:.1f} min, best_loss={metrics.best_val_loss:.6f}')

logger.info('Evaluating...')
result = trainer.evaluate(test_loader)

print(f'\nACCURACY: {result.direction_accuracy:.4f} ({result.direction_accuracy:.2%})')
print(f'RMSE: {result.rmse:.6f}')
print(f'MAE: {result.mae:.6f}')
print(f'Sharpe: {result.sharpe_ratio:.4f}')
print(f'R^2: {result.r_squared:.4f}')
print(f'Best epoch: {metrics.best_epoch + 1}')
print(f'Time: {dt/60:.1f} min')

model_path = trainer.save_model('transformer_upgraded_v2')
logger.info(f'Saved: {model_path}')
