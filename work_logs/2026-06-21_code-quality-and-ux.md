# 代码质量审查与前端UX优化

**日期**: 2026-06-21

## 修复清单

### 🟡 异常处理静默吞没（6个文件）
- **问题**: 多个文件使用 `except Exception: pass` 或裸 `except:` 静默吞掉异常
- **修复**: 
  - `model_inference.py`: 裸 `except:` → `except Exception:` 
  - `web_server.py`: 5处 `except: pass` → 添加 `logger.debug()` 日志
  - `cleaner.py`: 1处 → 添加日志
  - `benchmark.py`: 1处 → 添加日志
  - `watchdog.py`: 1处 → 添加日志
  - `stock_crawler.py`: 1处 → 添加日志
- **影响**: 之前这些异常完全不可见，现在可通过DEBUG级别日志追踪

### 🟢 前端UX优化（dashboard.html）
- **移除**: 模型路径、运行设备、注意力头数、编码器层数、训练参数（epochs/bs/lr/lookback）
- **标签友好化**:
  - "Transformer 模型信息" → "AI 预测模型"
  - "特征数" → "分析指标数"
  - "情感特征" → "辅助信号"
  - "模型维度" → "模型规模"（加"维"后缀）
  - "情感特征Badges" → "辅助数据源"
  - badge名称：`news_sentiment_3d` → `news sentiment 3d`（空格可读）
- **Overview修复**: ovML标签"数据源"→"预测引擎"（修正标签与内容不匹配）
- **文案**: "✅ ML已就绪" → "✅ AI模型预测"

## 验证结果
- 全部60个.py文件语法通过
- safe_divide边缘情况正确（除零/Series NaN）
- ModelConfig参数范围合理
- sentiment_features ⊂ features 一致性确认
- 服务器重启后正常运行（28特征模型）

## 边缘检查
| 检查项 | 结果 |
|:--|:--|
| safe_divide(10, 0) | 0.0 ✅ |
| safe_divide Series NaN | 正确处理 ✅ |
| d_model % n_heads | 整除 ✅ |
| dropout范围 | (0, 1) ✅ |
| sentiment ⊂ features | 全部4个 ✅ |
| news_feature_weight | 0.05 ✅ |

## 修改文件
| 文件 | 操作 |
|------|------|
| `live_trading/model_inference.py` | 裸except修复 |
| `live_trading/web_server.py` | 5处异常日志 |
| `data_pipeline/cleaner.py` | 1处异常日志 |
| `live_trading/benchmark.py` | 1处异常日志 |
| `live_trading/watchdog.py` | 1处异常日志 |
| `crawler/stock_crawler.py` | 1处异常日志 |
| `live_trading/templates/dashboard.html` | UX全面优化 |
