# 2026-06-17 全面代码审计与漏洞修复

## 审计范围
全项目46个Python文件 + 前端HTML + 配置 + 数据流

---

## 一、发现的漏洞与修复

### 1.1 推理引擎架构不匹配 (Critical)
- **文件**: `live_trading/model_inference.py`
- **问题**: 推理时使用PyTorch内置 `nn.TransformerEncoder`，但训练使用的是自定义 `StockTransformer`（Pre-LN + DropPath + AttnDropout + Final LayerNorm），state_dict键完全不匹配，**模型从未真正加载成功**
- **影响**: ML预测功能形同虚设，始终fallback到统计预测器
- **修复**: 推理直接使用 `StockTransformer` 类加载模型，确保架构完全匹配

### 1.2 模型路径过时 (Moderate)
- **文件**: `live_trading/model_inference.py:29`
- **问题**: 默认路径 `transformer_20stocks.pt`，但最新模型为 `transformer_stock_latest.pt`
- **修复**: 更新为 `transformer_stock_latest.pt`

### 1.3 分类头维度错误 (Critical)
- **文件**: `live_trading/model_inference.py:258-264`
- **问题**: 推理时使用5-way softmax分类（strong_down/down/flat/up/strong_up），但训练模型使用BCEWithLogits（二分类涨/跌）。输出维度不匹配
- **修复**: 改用sigmoid二分类，与训练一致

### 1.4 离线训练函数引用错误 (Moderate)
- **文件**: `live_trading/predictor.py:384-433`
- **问题**: `train_offline_transformer()` 引用不存在的类名（`TransformerTrainer`、`load_historical_data`）和错误的参数名（`nhead`/`num_layers`），函数永远执行失败
- **修复**: 改为使用正确的 `ModelTrainer` + `prepare_data` + `ModelConfig`

### 1.5 缩进错误 (Minor)
- **文件**: `live_trading/predictor.py:117-118`
- **问题**: `self.update_regime(ticker)` 行缩进异常（多了8个空格）
- **修复**: 修正缩进

### 1.6 状态持久化遗漏 (Moderate)
- **文件**: `live_trading/web_server.py`
- **问题1**: `_shutdown_handler` 保存的globals_dict缺少 `prediction_iters`
- **问题2**: `engine_loop` 最终保存缺少 `position_entry_time`、`position_sell_cooldown`、`recent_signals`
- **影响**: 优雅关闭时丢失预测迭代映射；引擎退出时丢失持仓计时和冷却状态
- **修复**: 统一三个保存点的字段列表

### 1.7 global声明遗漏 (已修复)
- **文件**: `live_trading/web_server.py:527`
- **问题**: `_price_is_stale` 和 `_data_source` 未在 `tick_engine()` 的global声明中，变成了局部变量
- **修复**: 添加 `global _price_is_stale, _data_source`

### 1.8 Yahoo拉取被跳过 (已修复)
- **文件**: `live_trading/web_server.py:525`
- **问题**: 未建仓时 `tick_engine()` 直接return，Yahoo价格永不拉取
- **修复**: 将Yahoo拉取移到建仓检查之前

---

## 二、架构一致性验证

| 组件 | 训练 | 推理(修复前) | 推理(修复后) |
|------|------|-------------|-------------|
| 模型类 | StockTransformer | InferenceTransformer | StockTransformer ✅ |
| 注意力机制 | Pre-LN + DropPath | Post-LN(内置) | Pre-LN + DropPath ✅ |
| 分类头 | BCEWithLogits | 5-way Softmax | Sigmoid ✅ |
| 输出维度 | (batch, horizon) | (batch, 5) | (batch, horizon) ✅ |
| State Dict | 自定义key | 内置key | 自定义key ✅ |

---

## 三、数据流完整性验证

```
Yahoo → fetch → tick_engine → _current_prices → build_status_data → API → 前端
  ✅        ✅        ✅ (修复后)      ✅              ✅            ✅     ✅
```

---

## 四、状态持久化对比

| 字段 | 定期保存(60s) | 关机保存 | 引擎退出 | 状态恢复 |
|------|:---:|:---:|:---:|:---:|
| current_prices | ✅ | ✅ | ✅ | ✅ |
| iteration_count | ✅ | ✅ | ✅ | ✅ |
| positions_initialized | ✅ | ✅ | ✅ | ✅ |
| position_entry_time | ✅ | ✅ | ✅(修复) | ✅ |
| position_sell_cooldown | ✅ | ✅ | ✅(修复) | ✅ |
| recent_signals | ✅ | ✅ | ✅(修复) | ✅ |
| prediction_iters | ✅ | ✅(修复) | ✅ | ✅ |
| ml_ready | ✅ | ✅ | ✅ | ✅ |

---

## 五、训练状态

| 指标 | Epoch 1 | Epoch 2 |
|------|---------|---------|
| 训练损失 | 0.3058 | 0.2199 ↓ |
| 验证损失 | 0.5337 | 0.4388 ↓ |
| 每轮耗时 | 393s | 395s |
| 设备 | MPS | MPS |

---

## 六、文件变更汇总

| 文件 | 变更 |
|------|------|
| `live_trading/model_inference.py` | 架构重写：InferenceTransformer→StockTransformer + 分类头修复 + 路径更新 |
| `live_trading/predictor.py` | 修复离线训练函数 + 缩进修正 |
| `live_trading/web_server.py` | 状态持久化补全 + global声明修复 + Yahoo拉取位置修复 |
