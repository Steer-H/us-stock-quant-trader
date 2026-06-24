# ⚠️ 項目警示文件 — 每次工作前必讀

> **規則**: 任何人（包括AI）在修改本項目任何代碼之前，必須先完整閱讀此文件。
> **最後更新**: 2026-06-19

---

## 一、已經出現過的錯誤（血淚教訓）

### 🔴 致命級（導致系統崩潰或數據錯誤）

| # | 錯誤 | 症狀 | 修復 |
|---|------|------|------|
| 1 | **模型架構不匹配**：inference用`InferenceTransformer`加載`StockTransformer`的權重 | 模型從未真正加載，state_dict鍵全部不匹配 | 推理直接使用`StockTransformer`類 |
| 2 | **方法名錯誤**：`ParquetStorage.list()` 不存在 | AttributeError崩潰 | 改用`list_keys()` |
| 3 | **缺少global聲明**：`_price_data_age_s`在函數內賦值但未聲明global | 變量不更新，數據年齡永遠為999 | 添加`global _price_data_age_s` |
| 4 | **成功被誤判為失敗**：Yahoo抓取成功但`completed`變量用錯→標記為失敗 | 真實價格被丟棄，回退到緩存 | 變量名`completed`→`fetched` |
| 5 | **TimeSeriesTransformer傳參過多**：向`TransformerEncoder`傳了7個參數但只接受5個 | 死代碼實例化即崩潰 | 移除多餘參數 |
| 6 | **閉市時執行交易**：缺少`is_trading_session`門控 | 休市時系統仍在買賣 | 在交易入口添加門控 |
| 7 | **狀態持久化遺漏關鍵欄位**：`save_state()`丟棄`ml_ready`/`prediction_iters`/`leverage_engine` | 重啟後ML模型需重加載，槓桿狀態丟失 | 補全3個欄位 |
| 8 | **`saved_at`未傳遞**：`load_state()`讀取了但返回字典裡沒有 | 前端顯示"保存於 ?" | 返回字典添加`'saved_at'` |
| 9 | **舊進程不殺導致代碼更新無效**：screen會話殘留 | 改了代碼但舊進程仍在運行舊邏輯 | 每次重啟必須`screen -X quit` + 殺埠 |
| 10 | **隨機數濫用**：價格偽造/止盈止損使用`random()` | 決策不穩定，結果不可復現 | 全部清零，使用確定性邏輯 |

### 🟡 嚴重級（導致功能異常或數據不準確）

| # | 錯誤 | 症狀 | 修復 |
|---|------|------|------|
| 11 | `safe_divide`不支持pandas Series | Series運算報錯 | 添加`isinstance(pd.Series)`分支 |
| 12 | `datetime.utcnow()`已棄用 | Python 3.12+ DeprecationWarning | 改為`datetime.now(timezone.utc)` |
| 13 | 多股票合併產生重複索引 | 訓練數據混亂 | 合併後去重 |
| 14 | `ReduceLROnPlateau` verbose參數不兼容 | 訓練中斷 | 移除verbose參數 |
| 15 | 模型加載條件過嚴（迭代限制） | 模型拒絕加載 | 移除限制，無條件加載 |
| 16 | 2027年美股假期缺失 | 節假日判斷錯誤 | 補充完整2027假期 |
| 17 | `INITIAL_POSITIONS`價格過時 | 初始持倉成本錯誤 | 更新到2025年中 |
| 18 | Werkzeug macOS kqueue錯誤 | 每個請求都報TypeError | 添加`PollSelector`替換 |
| 19 | `import random`在函數內部 | 代碼風格 | 提升至模塊級別 |
| 20 | 三元表達式被壓縮成單行（大量異常空白） | 不可維護 | 改為標準多行格式 |
| 21 | `model_inference.py`特徵列KeyError風險 | checkpoint特徵數>config時崩潰 | 過濾只保留df中存在的列 |
| 22 | `StockTransformer`缺少`final_norm`層 | Pre-LN架構不完整 | 添加`nn.LayerNorm` |
| 23 | 基準曲線圖初始化時機錯誤 | 圖表容器隱藏時初始化→0尺寸→永不顯示 | 移到容器可見時觸發 |
| 24 | 爬蟲Yahoo時區bug | tz-aware vs tz-naive比較報錯 | `df.index.tz_localize(None)` |
| 25 | frozen dataclass mutable default | `Dict={}`導致ValueError | 改用`field(default_factory=lambda: {...})` |

### 🟢 輕微級（不影響功能但有隱患）

| # | 錯誤 |
|---|------|
| 26 | 方向準確率在預測總數為0時顯示"0.0%"（應顯示"--"） |
| 27 | 做空交易標籤顯示為"賣出"（應為"做空"） |
| 28 | `run_server.sh`缺少崩潰次數上限（已補1小時20次） |

---

## 二、類似錯誤預警（根據以上模式推斷）

### 模式1：變量作用域 — 全局變量在函數內賦值
```python
# ❌ 危險
def foo():
    _some_global = new_value  # 創建了局部變量！

# ✅ 正確
def foo():
    global _some_global
    _some_global = new_value
```
**曾出現的文件**: `web_server.py` (_price_data_age_s, _v7_active)

### 模式2：序列化/反序列化不對稱
```python
# ❌ save時存了欄位A，load時沒恢復欄位A
# ❌ save時用dict存，load時用list取

# ✅ 每次改save_state必同步改load_state
# ✅ 每次加欄位必須兩端都加
```
**曾出現的文件**: `state_persistence.py` (ml_ready, prediction_iters, leverage_engine, saved_at)

### 模式3：導入已刪除/不存在的類
```python
# ❌ 注釋說"不再使用X"但代碼仍然 import X
# ❌ 類改名了但某些引用沒更新

# ✅ 改名後全項目搜索舊名稱
```
**曾出現的文件**: `model_inference.py` (InferenceTransformer)

### 模式4：方法名假設
```python
# ❌ 假設對象有某個方法就直接調用
storage.list()  # 實際是 list_keys()

# ✅ 先確認API再調用
```
**曾出現的文件**: `model_inference.py`, `web_server.py`

### 模式5：配置寫了但代碼沒用
```python
# ❌ config裡有attn_dropout=0.2但StockTransformer根本沒讀這個參數
# ❌ config裡有drop_path_rate=0.1但沒實現DropPath

# ✅ 新增config參數時，確認所有使用方都已接入
# ✅ 寫完優化文檔後，驗證代碼確實改了
```
**曾出現的文件**: `ml_model/transformer.py`, `config/settings.py`

### 模式6：前端DOM引用與tab切換不同步
```html
<!-- ❌ 圖表div在panel-charts，但初始化綁在analysis tab -->
<!-- ❌ 容器display:none時初始化圖表→尺寸0→永不渲染 -->

<!-- ✅ 圖表初始化必須在容器可見時觸發 -->
```
**曾出現的文件**: `dashboard.html` (benchmarkChart)

---


### 模式7：訓練日誌靜默 — Logger未配置Handler
```python
# ❌ 危險：trainer內部用logger.info()但root logger未配置
# 結果：訓練20分鐘看不到任何epoch進度輸出
python3 scripts/quick_train.py  # 無輸出！

# ✅ 正確：訓練腳本開頭必須配置root logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)-20s | %(message)s', stream=sys.stdout, force=True)
```
**曾出現的文件**: `ml_model/trainer.py` (logger.info到root但root無handler)

### 模式8：前端DOM元素只在JS引用但HTML中缺失
```html
<!-- ❌ 危險：JS中getElementById('panel-model')但HTML中沒有對應div -->
<!-- 結果：點擊tab報錯，模型面板永不顯示 -->

<!-- ✅ 正確：所有JS引用的ID必須在HTML中存在 -->
```
**曾出現的文件**: `dashboard.html` (模型tab按鈕存在但panel-model和所有m-*元素缺失)

### 模式9：前端import隨機數在函數體內
```python
# ❌ 危險：函數內import random
def generate_mock():
    import random  # 每次調用都重新導入
    return random.random()

# ✅ 正確：import在模塊頂部
import random
random.seed(42)  # 確定性種子
```
**曾出現的文件**: `dashboard.py`

## 三、不可輕易改動的區域

### 🔒 絕對禁區（動了必出事）

| 區域 | 原因 |
|------|------|
| `state_persistence.py` 的 `save_state()` 和 `load_state()` | 序列化/反序列化必須嚴格對稱，改一個必改另一個 |
| `web_server.py` 的 `tick_engine()` 交易門控邏輯 | 閉市門控(`is_trading_session`)、價格過期保護、建倉邏輯 |
| `ml_model/transformer.py` 的 `StockTransformer` 類名和forward籤名 | 模型權重與類名綁定，改名會導致所有已訓練模型無法加載 |
| `live_trading/state_persistence.py` 的 `serialize_*` / `deserialize_*` | 改了序列化格式，歷史狀態文件全部失效 |
| `data/trading_state.json` | 正在運行的狀態文件，手動改會損壞數據 |

### ⚠️ 高風險區（需充分測試後改）

| 區域 | 注意事項 |
|------|---------|
| `live_trading/web_server.py` 全局變量 | 任何新全局變量=必須加`global`聲明，必須加入`_collect_globals_dict`或確認不需持久化 |
| `config/settings.py` 的 `ModelConfig` | 改了模型參數→舊checkpoint可能不兼容→需重新訓練 |
| `live_trading/templates/dashboard.html` | JS是單文件無編譯，改完必須瀏覽器硬刷新（Cmd+Shift+R）驗證 |
| `requirements.txt` | 加新依賴需確認兼容性，尤其是yfinance/werkzeug版本 |

### 📌 已知設計權衡（不是bug，不要修）

| 項目 | 說明 |
|------|------|
| `ml_model/data_loader.py` shuffle=True | batch級shuffle有輕微時序洩漏但有助於SGD，已評估為可接受 |
| `TimeSeriesTransformer` | encoder-decoder架構，死代碼但保留參考，不影響運行 |
| Flask開發伺服器 | 生產應切waitress/gunicorn，但`threaded=True`已緩解kqueue問題 |
| 前端輪詢1秒 | WebSocket已移除改為輪詢，1秒間隔可接受 |

---

## 四、工作規則（必須遵守）

### 每次修改代碼前
1. ✅ **先讀本文件** — 了解歷史錯誤和禁區
2. ✅ **先殺舊進程** — `screen -S trading -X quit` + `lsof -ti tcp:8080 | xargs kill -9`
3. ✅ **先讀相關日誌** — `work_logs/` 最新幾篇了解上下文
4. ✅ **先跑語法檢查** — `python3 -c "import ast; ast.parse(open('文件.py').read())"`

### 每次修改代碼後
1. ✅ **寫工作日誌** — 存入 `work_logs/YYYY-MM-DD_描述.md`
2. ✅ **全項目語法驗證** — 所有.py文件通過ast.parse()
3. ✅ **重啟服務** — `screen -dmS trading python3 -u live_trading/web_server.py`
4. ✅ **驗證功能** — `curl http://localhost:8080/api/status` 檢查數據正常
5. ✅ **檢查前端** — 瀏覽器打開 `http://localhost:8080` 硬刷新驗證
6. ✅ **更新CHANGELOG.md** — 記錄本次改動摘要

### 日誌規範
- 文件名: `YYYY-MM-DD_簡短描述.md`
- 內容必須包含: 修改文件列表、問題描述、修複方案、影響評估
- 如果有bug修復，標註嚴重程度: 🔴致命 / 🟡嚴重 / 🟢輕微

### 禁止事項
- ❌ 不要修改 `data/trading_state.json`（運行狀態）
- ❌ 不要刪除 `work_logs/` 中的歷史日誌
- ❌ 不要移除全局變量而不檢查所有引用
- ❌ 不要改變 `StockTransformer` 類名或模型保存格式
- ❌ 不要在未測試的情況下提交涉及交易邏輯的改動
- ❌ 不要忽略`AGENTS.md`中的項目規範

---

## 五、快速檢查清單（每次工作結束時逐項打勾）

```
[ ] 所有.py文件通過 ast.parse() 語法檢查
[ ] 舊進程已殺，新進程已啟動
[ ] curl /api/status 返回正常數據
[ ] 數據源顯示正確（非"緩存"即"v7"）
[ ] 瀏覽器硬刷新後可正常訪問
[ ] work_logs 已寫入本次工作日誌
[ ] CHANGELOG.md 已更新
[ ] 未觸碰GUARDRAILS.md中的"不可改動"區域（如觸碰則已充分測試）
```

---

> **記住**: 這個項目的交易邏輯是經過多輪審計才穩定下來的。每次改動都可能是新的bug來源。謹慎、記錄、驗證。
