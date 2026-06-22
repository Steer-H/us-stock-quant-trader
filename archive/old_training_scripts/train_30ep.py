#!/usr/bin/env python3
"""30轮训练 - 保守但可完成的配置"""
import sys, os, time, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    stream=sys.stdout,
    force=True
)

print(f'{time.strftime("%H:%M:%S")} | === 30轮训练 ===', flush=True)

from config.settings import model_config
model_config.device = 'cpu'
model_config.d_model = 192
model_config.n_heads = 6
model_config.n_encoder_layers = 3
model_config.d_ff = 768
model_config.lookback_window = 60
model_config.batch_size = 32
model_config.epochs = 30
model_config.early_stopping_patience = 15

print(f'{time.strftime("%H:%M:%S")} | d192/h6/l3/bs32/30ep/pat15', flush=True)

from ml_model.data_loader import prepare_data
train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)
print(f'{time.strftime("%H:%M:%S")} | Data: {len(train_loader)} batches', flush=True)

from ml_model.trainer import ModelTrainer
trainer = ModelTrainer(model_config)

t0 = time.perf_counter()
metrics = trainer.train(train_loader, val_loader, epochs=30)
dt = time.perf_counter() - t0

print(f'{time.strftime("%H:%M:%S")} | Done: {dt/60:.1f}min best@e{metrics.best_epoch+1}', flush=True)

result = trainer.evaluate(test_loader)
print(f'ACCURACY: {result.direction_accuracy:.4f} ({result.direction_accuracy:.2%})', flush=True)
print(f'RMSE: {result.rmse:.6f} MAE: {result.mae:.6f}', flush=True)
print(f'Sharpe: {result.sharpe_ratio:.4f} R^2: {result.r_squared:.4f}', flush=True)
print(f'Best: epoch {metrics.best_epoch+1} Time: {dt/60:.1f}min', flush=True)

path = trainer.save_model('transformer_30ep')
print(f'SAVED: {path}', flush=True)
