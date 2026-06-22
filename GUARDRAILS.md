# ⚠️ 项目警示文件 — 每次工作前必读

> **规则**: 任何人（包括AI）在修改本项目任何代码之前，必须先完整阅读此文件。
> **最后更新**: 2026-06-19

---

## 一、已经出现过的错误（血泪教训）

### 🔴 致命级（导致系统崩溃或数据错误）

| # | 错误 | 症状 | 修复 |
|---|------|------|------|
| 1 | **模型架构不匹配**：inference用`InferenceTransformer`加载`StockTransformer`的权重 | 模型从未真正加载，state_dict键全部不匹配 | 推理直接使用`StockTransformer`类 |
| 2 | **方法名错误**：`ParquetStorage.list()` 不存在 | AttributeError崩溃 | 改用`list_keys()` |
| 3 | **缺少global声明**：`_price_data_age_s`在函数内赋值但未声明global | 变量不更新，数据年龄永远为999 | 添加`global _price_data_age_s` |
| 4 | **成功被误判为失败**：Yahoo抓取成功但`completed`变量用错→标记为失败 | 真实价格被丢弃，回退到缓存 | 变量名`completed`→`fetched` |
| 5 | **TimeSeriesTransformer传参过多**：向`TransformerEncoder`传了7个参数但只接受5个 | 死代码实例化即崩溃 | 移除多余参数 |
| 6 | **闭市时执行交易**：缺少`is_trading_session`门控 | 休市时系统仍在买卖 | 在交易入口添加门控 |
| 7 | **状态持久化遗漏关键字段**：`save_state()`丢弃`ml_ready`/`prediction_iters`/`leverage_engine` | 重启后ML模型需重加载，杠杆状态丢失 | 补全3个字段 |
| 8 | **`saved_at`未传递**：`load_state()`读取了但返回字典里没有 | 前端显示"保存于 ?" | 返回字典添加`'saved_at'` |
| 9 | **旧进程不杀导致代码更新无效**：screen会话残留 | 改了代码但旧进程仍在运行旧逻辑 | 每次重启必须`screen -X quit` + 杀端口 |
| 10 | **随机数滥用**：价格伪造/止盈止损使用`random()` | 决策不稳定，结果不可复现 | 全部清零，使用确定性逻辑 |

### 🟡 严重级（导致功能异常或数据不准确）

| # | 错误 | 症状 | 修复 |
|---|------|------|------|
| 11 | `safe_divide`不支持pandas Series | Series运算报错 | 添加`isinstance(pd.Series)`分支 |
| 12 | `datetime.utcnow()`已弃用 | Python 3.12+ DeprecationWarning | 改为`datetime.now(timezone.utc)` |
| 13 | 多股票合并产生重复索引 | 训练数据混乱 | 合并后去重 |
| 14 | `ReduceLROnPlateau` verbose参数不兼容 | 训练中断 | 移除verbose参数 |
| 15 | 模型加载条件过严（迭代限制） | 模型拒绝加载 | 移除限制，无条件加载 |
| 16 | 2027年美股假期缺失 | 节假日判断错误 | 补充完整2027假期 |
| 17 | `INITIAL_POSITIONS`价格过时 | 初始持仓成本错误 | 更新到2025年中 |
| 18 | Werkzeug macOS kqueue错误 | 每个请求都报TypeError | 添加`PollSelector`替换 |
| 19 | `import random`在函数内部 | 代码风格 | 提升至模块级别 |
| 20 | 三元表达式被压缩成单行（大量异常空白） | 不可维护 | 改为标准多行格式 |
| 21 | `model_inference.py`特征列KeyError风险 | checkpoint特征数>config时崩溃 | 过滤只保留df中存在的列 |
| 22 | `StockTransformer`缺少`final_norm`层 | Pre-LN架构不完整 | 添加`nn.LayerNorm` |
| 23 | 基准曲线图初始化时机错误 | 图表容器隐藏时初始化→0尺寸→永不显示 | 移到容器可见时触发 |
| 24 | 爬虫Yahoo时区bug | tz-aware vs tz-naive比较报错 | `df.index.tz_localize(None)` |
| 25 | frozen dataclass mutable default | `Dict={}`导致ValueError | 改用`field(default_factory=lambda: {...})` |

### 🟢 轻微级（不影响功能但有隐患）

| # | 错误 |
|---|------|
| 26 | 方向准确率在预测总数为0时显示"0.0%"（应显示"--"） |
| 27 | 做空交易标签显示为"卖出"（应为"做空"） |
| 28 | `run_server.sh`缺少崩溃次数上限（已补1小时20次） |

---

## 二、类似错误预警（根据以上模式推断）

### 模式1：变量作用域 — 全局变量在函数内赋值
```python
# ❌ 危险
def foo():
    _some_global = new_value  # 创建了局部变量！

# ✅ 正确
def foo():
    global _some_global
    _some_global = new_value
```
**曾出现的文件**: `web_server.py` (_price_data_age_s, _v7_active)

### 模式2：序列化/反序列化不对称
```python
# ❌ save时存了字段A，load时没恢复字段A
# ❌ save时用dict存，load时用list取

# ✅ 每次改save_state必同步改load_state
# ✅ 每次加字段必须两端都加
```
**曾出现的文件**: `state_persistence.py` (ml_ready, prediction_iters, leverage_engine, saved_at)

### 模式3：导入已删除/不存在的类
```python
# ❌ 注释说"不再使用X"但代码仍然 import X
# ❌ 类改名了但某些引用没更新

# ✅ 改名后全项目搜索旧名称
```
**曾出现的文件**: `model_inference.py` (InferenceTransformer)

### 模式4：方法名假设
```python
# ❌ 假设对象有某个方法就直接调用
storage.list()  # 实际是 list_keys()

# ✅ 先确认API再调用
```
**曾出现的文件**: `model_inference.py`, `web_server.py`

### 模式5：配置写了但代码没用
```python
# ❌ config里有attn_dropout=0.2但StockTransformer根本没读这个参数
# ❌ config里有drop_path_rate=0.1但没实现DropPath

# ✅ 新增config参数时，确认所有使用方都已接入
# ✅ 写完优化文档后，验证代码确实改了
```
**曾出现的文件**: `ml_model/transformer.py`, `config/settings.py`

### 模式6：前端DOM引用与tab切换不同步
```html
<!-- ❌ 图表div在panel-charts，但初始化绑在analysis tab -->
<!-- ❌ 容器display:none时初始化图表→尺寸0→永不渲染 -->

<!-- ✅ 图表初始化必须在容器可见时触发 -->
```
**曾出现的文件**: `dashboard.html` (benchmarkChart)

---


### 模式7：训练日志静默 — Logger未配置Handler
```python
# ❌ 危险：trainer内部用logger.info()但root logger未配置
# 结果：训练20分钟看不到任何epoch进度输出
python3 scripts/quick_train.py  # 无输出！

# ✅ 正确：训练脚本开头必须配置root logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-20s | %(message)s', stream=sys.stdout, force=True)
```
**曾出现的文件**: `ml_model/trainer.py` (logger.info到root但root无handler)

### 模式8：前端DOM元素只在JS引用但HTML中缺失
```html
<!-- ❌ 危险：JS中getElementById('panel-model')但HTML中没有对应div -->
<!-- 结果：点击tab报错，模型面板永不显示 -->

<!-- ✅ 正确：所有JS引用的ID必须在HTML中存在 -->
```
**曾出现的文件**: `dashboard.html` (模型tab按钮存在但panel-model和所有m-*元素缺失)

### 模式9：前端import随机数在函数体内
```python
# ❌ 危险：函数内import random
def generate_mock():
    import random  # 每次调用都重新导入
    return random.random()

# ✅ 正确：import在模块顶部
import random
random.seed(42)  # 确定性种子
```
**曾出现的文件**: `dashboard.py`

## 三、不可轻易改动的区域

### 🔒 绝对禁区（动了必出事）

| 区域 | 原因 |
|------|------|
| `state_persistence.py` 的 `save_state()` 和 `load_state()` | 序列化/反序列化必须严格对称，改一个必改另一个 |
| `web_server.py` 的 `tick_engine()` 交易门控逻辑 | 闭市门控(`is_trading_session`)、价格过期保护、建仓逻辑 |
| `ml_model/transformer.py` 的 `StockTransformer` 类名和forward签名 | 模型权重与类名绑定，改名会导致所有已训练模型无法加载 |
| `live_trading/state_persistence.py` 的 `serialize_*` / `deserialize_*` | 改了序列化格式，历史状态文件全部失效 |
| `data/trading_state.json` | 正在运行的状态文件，手动改会损坏数据 |

### ⚠️ 高风险区（需充分测试后改）

| 区域 | 注意事项 |
|------|---------|
| `live_trading/web_server.py` 全局变量 | 任何新全局变量=必须加`global`声明，必须加入`_collect_globals_dict`或确认不需持久化 |
| `config/settings.py` 的 `ModelConfig` | 改了模型参数→旧checkpoint可能不兼容→需重新训练 |
| `live_trading/templates/dashboard.html` | JS是单文件无编译，改完必须浏览器硬刷新（Cmd+Shift+R）验证 |
| `requirements.txt` | 加新依赖需确认兼容性，尤其是yfinance/werkzeug版本 |

### 📌 已知设计权衡（不是bug，不要修）

| 项目 | 说明 |
|------|------|
| `ml_model/data_loader.py` shuffle=True | batch级shuffle有轻微时序泄漏但有助于SGD，已评估为可接受 |
| `TimeSeriesTransformer` | encoder-decoder架构，死代码但保留参考，不影响运行 |
| Flask开发服务器 | 生产应切waitress/gunicorn，但`threaded=True`已缓解kqueue问题 |
| 前端轮询1秒 | WebSocket已移除改为轮询，1秒间隔可接受 |

---

## 四、工作规则（必须遵守）

### 每次修改代码前
1. ✅ **先读本文件** — 了解历史错误和禁区
2. ✅ **先杀旧进程** — `screen -S trading -X quit` + `lsof -ti tcp:8080 | xargs kill -9`
3. ✅ **先读相关日志** — `work_logs/` 最新几篇了解上下文
4. ✅ **先跑语法检查** — `python3 -c "import ast; ast.parse(open('文件.py').read())"`

### 每次修改代码后
1. ✅ **写工作日志** — 存入 `work_logs/YYYY-MM-DD_描述.md`
2. ✅ **全项目语法验证** — 所有.py文件通过ast.parse()
3. ✅ **重启服务** — `screen -dmS trading python3 -u live_trading/web_server.py`
4. ✅ **验证功能** — `curl http://localhost:8080/api/status` 检查数据正常
5. ✅ **检查前端** — 浏览器打开 `http://localhost:8080` 硬刷新验证
6. ✅ **更新CHANGELOG.md** — 记录本次改动摘要

### 日志规范
- 文件名: `YYYY-MM-DD_简短描述.md`
- 内容必须包含: 修改文件列表、问题描述、修复方案、影响评估
- 如果有bug修复，标注严重程度: 🔴致命 / 🟡严重 / 🟢轻微

### 禁止事项
- ❌ 不要修改 `data/trading_state.json`（运行状态）
- ❌ 不要删除 `work_logs/` 中的历史日志
- ❌ 不要移除全局变量而不检查所有引用
- ❌ 不要改变 `StockTransformer` 类名或模型保存格式
- ❌ 不要在未测试的情况下提交涉及交易逻辑的改动
- ❌ 不要忽略`AGENTS.md`中的项目规范

---

## 五、快速检查清单（每次工作结束时逐项打勾）

```
[ ] 所有.py文件通过 ast.parse() 语法检查
[ ] 旧进程已杀，新进程已启动
[ ] curl /api/status 返回正常数据
[ ] 数据源显示正确（非"缓存"即"v7"）
[ ] 浏览器硬刷新后可正常访问
[ ] work_logs 已写入本次工作日志
[ ] CHANGELOG.md 已更新
[ ] 未触碰GUARDRAILS.md中的"不可改动"区域（如触碰则已充分测试）
```

---

> **记住**: 这个项目的交易逻辑是经过多轮审计才稳定下来的。每次改动都可能是新的bug来源。谨慎、记录、验证。
