#!/usr/bin/env python3
"""CPU训练40只股票Transformer模型"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import model_config, system_config
from config.logging_config import setup_logging

# 强制CPU模式
model_config.device = 'cpu'
model_config.epochs = 5

setup_logging(system_config)

import logging
logger = logging.getLogger('train_cpu_40')
logger.info('=' * 60)
logger.info('  CPU训练 40只股票 Transformer (5轮)')
logger.info('=' * 60)

from ml_model import ModelTrainer, prepare_data

logger.info('正在加载训练数据...')
train_loader, val_loader, test_loader, scaler = prepare_data(config=model_config)

trainer = ModelTrainer(model_config)
logger.info(f'设备: {model_config.device}, 开始训练...')
metrics = trainer.train(train_loader, val_loader, epochs=5)

logger.info('正在评估模型...')
result = trainer.evaluate(test_loader)

print('\n' + '=' * 50)
print('  评估结果')
print('=' * 50)
print(f'  方向准确率: {result.direction_accuracy:.2%}')
print(f'  RMSE:        {result.rmse:.6f}')
print(f'  MAE:         {result.mae:.6f}')
print(f'  Sharpe:      {result.sharpe_ratio:.4f}')
print(f'  R²:          {result.r_squared:.4f}')
print(f'  精度合格:    {"✅" if result.is_acceptable else "❌"}')
print(f'  测试样本:    {result.total_samples}')
print('=' * 50)

model_path = trainer.save_model('transformer_cpu_40stocks')
logger.info(f'模型已保存: {model_path}')
