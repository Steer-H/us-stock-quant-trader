#!/usr/bin/env python3
"""对比 MPS 和 CPU 训练结果"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def extract_metrics(text: str) -> dict:
    m = {}
    patterns = {
        'direction_accuracy': r'方向准确率[:\s]+([\d.]+)%',
        'rmse': r'RMSE[:\s]+([\d.]+)',
        'mae': r'MAE[:\s]+([\d.]+)',
        'sharpe': r'(?:年化)?(?:夏普比率|Sharpe)[:\s]+([-\d.]+)',
        'r_squared': r'R²[:\s]+([-\d.]+)',
        'samples': r'测试样本数?[:\s]+([\d,]+)',
    }
    for k, p in patterns.items():
        match = re.search(p, text)
        if match:
            val = match.group(1).replace(',', '')
            if k == 'samples':
                m[k] = int(val)
            elif k == 'direction_accuracy':
                m[k] = float(val) / 100.0  # 52.94% → 0.5294
            else:
                m[k] = float(val)
    return m

def fmt_pct(v):
    return f'{v:.2%}'

def fmt_float(v):
    return f'{v:.6f}'

def fmt_val(v, is_pct):
    return fmt_pct(v) if is_pct else fmt_float(v)

def check(k, v):
    if k == 'direction_accuracy':
        return ' ✅' if v >= 0.55 else ' ❌'
    elif k == 'rmse':
        return ' ✅' if v < 0.06 else ' ❌'
    return ''

# Read data
mps_text = (ROOT / 'logs' / 'train_40stocks.log').read_text()
mps = extract_metrics(mps_text)

cpu_rf = ROOT / 'logs' / 'train_cpu_40_result.txt'
cpu_text = cpu_rf.read_text() if cpu_rf.exists() else ''
cpu = extract_metrics(cpu_text)

if not cpu.get('direction_accuracy'):
    cpu_log = ROOT / 'logs' / 'train_cpu_40stocks.log'
    if cpu_log.exists():
        cpu2 = extract_metrics(cpu_log.read_text())
        if cpu2:
            cpu = cpu2

MIN_DIR = 0.55
MAX_RMSE = 0.06

print()
print('=' * 60)
print('  Transformer 训练对比: MPS vs CPU (40只股票)')
print('=' * 60)
print()
print(f'{"指标":<20} {"MPS":>15} {"CPU":>15} {"变化":>10}')
print('-' * 60)

for key, label, is_pct in [
    ('direction_accuracy', '方向准确率', True),
    ('rmse', 'RMSE', False),
    ('mae', 'MAE', False),
    ('sharpe', '夏普比率', False),
    ('r_squared', 'R²', False),
]:
    mv, cv = mps.get(key), cpu.get(key)
    if mv is None and cv is None:
        continue
    
    ms = (fmt_val(mv, is_pct) + check(key, mv)) if mv is not None else 'N/A'
    cs = (fmt_val(cv, is_pct) + check(key, cv)) if cv is not None else 'N/A'
    
    if mv is not None and cv is not None:
        diff = cv - mv
        ds = fmt_pct(diff) if is_pct else f'{diff:+.6f}'
    else:
        ds = '-'
    
    print(f'{label:<20} {ms:>15} {cs:>15} {ds:>10}')

print('-' * 60)

sm = mps.get('samples', '-')
sc = cpu.get('samples', '-')
if sm != '-' or sc != '-':
    print(f'{"测试样本":<20} {str(sm):>15} {str(sc):>15}')

print()

cd, cr = cpu.get('direction_accuracy', 0), cpu.get('rmse', 999)
md = mps.get('direction_accuracy', 0)

cpu_ok = cd >= MIN_DIR and cr < MAX_RMSE
mps_ok = md >= MIN_DIR and mps.get('rmse', 999) < MAX_RMSE

if not cpu:
    print('⚠️  CPU 训练结果尚未生成，请等待训练完成')
elif cpu_ok:
    print(f'✅ CPU 训练达标！方向准确率 {cd:.2%} ≥ {MIN_DIR:.0%}, RMSE {cr:.6f} < {MAX_RMSE}')
    if not mps_ok:
        print(f'📊 MPS 精度问题确认：CPU 达标而 MPS 不达标（方向准确率 {md:.2%} < {MIN_DIR:.0%}）')
    else:
        print('📊 CPU 和 MPS 均达标')
else:
    print(f'❌ CPU 训练未达标！方向准确率 {cd:.2%} (要求 ≥ {MIN_DIR:.0%}), RMSE {cr:.6f} (要求 < {MAX_RMSE})')

print()
