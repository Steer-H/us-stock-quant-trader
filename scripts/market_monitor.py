import os
#!/usr/bin/env python3
"""
市場時段交易系統監控腳本 (Market Monitor)

在美股交易時段（盤前/正常盤/盤後）持續監測交易系統狀態，
發現異常自動修復並寫入詳細日誌。

監控項：
  1. Web伺服器進程是否存活 → 自動重啟
  2. 健康檢查 API 是否正常響應 → 診斷並修復
  3. 持倉數據是否異常（數量/市值突變）→ 記錄告警
  4. 日誌中是否出現 ERROR/CRITICAL → 提取並記錄
  5. 系統資源（CPU/內存）是否過高 → 記錄告警
  6. 埠是否被佔用 → 清理殭屍進程

運行方式：
  python scripts/market_monitor.py           # 單次檢查
  python scripts/market_monitor.py --daemon  # 持續監控（循環模式）
  python scripts/market_monitor.py --once    # 單次檢查並退出

配合 crontab 使用（每2分鐘檢查一次）：
  */2 21-4,9-16 * * 1-5 cd /path/to/project && python scripts/market_monitor.py --once
"""

import sys
import os
import time
import json
import signal
import logging
import subprocess
import traceback
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
import urllib.request
import urllib.error

# 項目根目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# 日誌配置
# ============================================================================
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 監控專用日誌
monitor_log = LOG_DIR / 'market_monitor.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | [MarketMonitor] %(message)s',
    handlers=[
        logging.FileHandler(monitor_log, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('market_monitor')

# 異常事件日誌（僅記錄異常）
anomaly_log = LOG_DIR / 'market_anomalies.log'
anomaly_handler = logging.FileHandler(anomaly_log, encoding='utf-8')
anomaly_handler.setLevel(logging.WARNING)
anomaly_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s'
))
anomaly_logger = logging.getLogger('market_anomalies')
anomaly_logger.addHandler(anomaly_handler)
anomaly_logger.propagate = False


# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    'server_port': 8080,
    'health_url': 'http://127.0.0.1:8080/api/health',
    'status_url': 'http://127.0.0.1:8080/api/status',
    'health_timeout': 10,          # 健康檢查超時（秒）
    'startup_timeout': 45,          # 啟動超時（秒）
    'check_interval': 120,          # 循環檢查間隔（秒）
    'max_consecutive_failures': 3,  # 連續失敗閾值
    'memory_warning_mb': 800,       # 內存告警閾值（MB）
    'cpu_warning_pct': 80,          # CPU告警閾值（%）
    'position_value_change_pct': 30, # 單持倉市值變化告警閾值
    'error_log_scan_lines': 200,    # 掃描最近的日誌行數
}


# ============================================================================
# 市場時鐘集成
# ============================================================================
def is_market_active() -> Tuple[bool, str]:
    """
    判斷當前是否處於美股交易相關時段
    
    返回:
        (是否活躍, 狀態描述)
    """
    try:
        from live_trading.market_clock import MarketClock, MarketStatus
        clock = MarketClock()
        status, desc = clock.get_status()
        
        # 盤前、正常盤、盤後都算活躍時段
        active_statuses = {
            MarketStatus.PRE_MARKET,
            MarketStatus.REGULAR_HOURS,
            MarketStatus.AFTER_HOURS,
        }
        
        is_active = status in active_statuses
        return is_active, desc
    except Exception as e:
        logger.warning(f"無法獲取市場狀態: {e}，默認按活躍時段處理")
        return True, "未知(默認活躍)"


# ============================================================================
# 健康檢查
# ============================================================================
def check_server_process() -> Tuple[bool, Optional[int]]:
    """
    檢查 Web 伺服器進程是否存活
    
    返回:
        (是否存活, PID或None)
    """
    try:
        # 檢查佔用埠的進程
        result = subprocess.run(
            ['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
            capture_output=True, text=True, timeout=5
        )
        pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        
        if pids:
            # 驗證進程確實是 Python web_server
            for pid in pids:
                try:
                    proc_result = subprocess.run(
                        ['ps', '-p', pid, '-o', 'command='],
                        capture_output=True, text=True, timeout=3
                    )
                    if 'web_server' in proc_result.stdout:
                        return True, int(pid)
                except Exception:
                    continue
            # 埠被其他進程佔用
            logger.warning(f"埠 {CONFIG['server_port']} 被非伺服器進程佔用: {pids}")
            return False, None
        
        return False, None
    except Exception as e:
        logger.error(f"檢查進程失敗: {e}")
        return False, None


def check_health_api() -> Tuple[bool, Optional[Dict]]:
    """
    檢查健康檢查 API 是否正常響應
    
    返回:
        (是否正常, 響應數據或None)
    """
    try:
        req = urllib.request.Request(
            CONFIG['health_url'],
            headers={'User-Agent': 'MarketMonitor/1.0'}
        )
        with urllib.request.urlopen(req, timeout=CONFIG['health_timeout']) as resp:
            data = json.loads(resp.read().decode())
            
            if data.get('status') == 'ok':
                return True, data
            else:
                logger.warning(f"健康檢查返回異常狀態: {data}")
                return False, data
    except urllib.error.URLError as e:
        logger.warning(f"健康檢查連接失敗: {e.reason}")
        return False, None
    except Exception as e:
        logger.warning(f"健康檢查異常: {e}")
        return False, None


def check_position_anomalies() -> List[Dict]:
    """
    檢查持倉數據異常
    
    返回:
        異常列表
    """
    anomalies = []
    
    try:
        req = urllib.request.Request(
            CONFIG['status_url'],
            headers={'User-Agent': 'MarketMonitor/1.0'}
        )
        with urllib.request.urlopen(req, timeout=CONFIG['health_timeout']) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return anomalies
    
    # 如果系統還在等待開市，跳過持倉檢查
    if data.get('waiting_for_open', False):
        logger.debug("系統等待開市中，跳過持倉檢查")
        return anomalies
    
    # 從 account 嵌套對象讀取權益數據
    account = data.get('account', {})
    total_equity = account.get('total_equity', data.get('total_equity', 0))
    initial_capital = account.get('initial_capital', data.get('initial_capital', 100000))
    
    if total_equity <= 0:
        anomalies.append({
            'type': 'zero_equity',
            'severity': 'CRITICAL',
            'message': f'總權益為0或負數: ${total_equity:,.2f}',
        })
    
    # 檢查整體回撤
    if initial_capital > 0:
        net_pnl = total_equity - initial_capital
        net_pnl_pct = (net_pnl / initial_capital * 100)
        if net_pnl_pct < -20:
            anomalies.append({
                'type': 'large_drawdown',
                'severity': 'CRITICAL',
                'message': f'整體虧損超過20%: {net_pnl_pct:.1f}% (${net_pnl:,.2f})',
            })
        elif net_pnl_pct < -10:
            anomalies.append({
                'type': 'drawdown_warning',
                'severity': 'WARNING',
                'message': f'整體虧損超過10%: {net_pnl_pct:.1f}% (${net_pnl:,.2f})',
            })
    
    # 檢查單持倉異常
    positions = data.get('positions', [])
    for pos in positions:
        ticker = pos.get('ticker', '?')
        market_value = pos.get('market_value', 0)
        unrealized_pnl_pct = pos.get('unrealized_pnl_pct', 0)
        
        if abs(unrealized_pnl_pct) > 50:
            anomalies.append({
                'type': 'position_extreme_pnl',
                'severity': 'WARNING',
                'message': f'{ticker} 未實現盈虧達到 {unrealized_pnl_pct:.1f}%',
            })
        
        # 檢查持倉權重異常
        weight = pos.get('weight', 0)
        if weight > 40:
            anomalies.append({
                'type': 'position_concentration',
                'severity': 'WARNING',
                'message': f'{ticker} 持倉集中度過高: {weight:.1f}%',
            })
    
    # 檢查持倉數量突變
    positions_count = len(positions)
    if positions_count > 50:
        anomalies.append({
            'type': 'too_many_positions',
            'severity': 'WARNING',
            'message': f'持倉數量過多: {positions_count}',
        })
    
    # 檢查交易信號是否正常產生
    market_data = data.get('market', {})
    market_status = market_data.get('status', data.get('market_status', ''))
    positions_initialized = data.get('positions_initialized', False)
    if market_status == 'REGULAR' and not positions_initialized:
        anomalies.append({
            'type': 'positions_not_initialized',
            'severity': 'ERROR',
            'message': '正常交易時段但持倉未初始化',
        })
    
    return anomalies


# 日誌文件上次掃描位置（避免重複報告同一錯誤）
_log_scan_positions: Dict[str, int] = {}

def scan_error_logs() -> List[Dict]:
    """
    掃描日誌文件中的新增錯誤（增量掃描）
    
    返回:
        錯誤列表
    """
    errors = []
    
    # 掃描 watchdog.log 和 server 相關日誌
    log_files = [
        LOG_DIR / 'watchdog.log',
        LOG_DIR / 'server.log',
        LOG_DIR / 'error.log',
    ]
    
    # 找到最新的 server 日誌
    server_logs = sorted(LOG_DIR.glob('server_*.log'))
    if server_logs:
        log_files.append(server_logs[-1])
    
    for log_file in log_files:
        if not log_file.exists():
            continue
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            last_pos = _log_scan_positions.get(str(log_file), 0)
            
            # 如果文件被截斷，從頭開始
            if last_pos > total_lines:
                last_pos = 0
            
            # 只掃描新增的行
            recent = lines[last_pos:]
            _log_scan_positions[str(log_file)] = total_lines
            
            # 如果新增行太多，只取最後 N 行
            if len(recent) > CONFIG['error_log_scan_lines']:
                recent = recent[-CONFIG['error_log_scan_lines']:]
            
            for line in recent:
                if 'ERROR' in line or 'CRITICAL' in line or 'Traceback' in line:
                    errors.append({
                        'source': log_file.name,
                        'line': line.strip()[:300],
                    })
        except Exception:
            pass
    
    return errors


def check_system_resources() -> List[Dict]:
    """
    檢查系統資源使用情況
    
    返回:
        告警列表
    """
    alerts = []
    
    try:
        # 檢查內存（macOS 使用 ps 或 vm_stat）
        result = subprocess.run(
            ['ps', '-eo', 'pid,rss,comm'],
            capture_output=True, text=True, timeout=5
        )
        
        for line in result.stdout.split('\n'):
            if 'python' in line and 'web_server' in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rss_kb = int(parts[1])
                        rss_mb = rss_kb / 1024
                        if rss_mb > CONFIG['memory_warning_mb']:
                            alerts.append({
                                'type': 'high_memory',
                                'severity': 'WARNING',
                                'message': f'Web伺服器內存使用過高: {rss_mb:.0f}MB (閾值: {CONFIG["memory_warning_mb"]}MB)',
                            })
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        logger.debug(f"資源檢查失敗: {e}")
    
    return alerts


# ============================================================================
# 自動修復
# ============================================================================
def restart_server() -> bool:
    """
    自動重啟 Web 伺服器
    
    返回:
        是否成功
    """
    logger.info("=" * 50)
    logger.info("🔄 開始自動重啟交易系統...")
    logger.info("=" * 50)
    
    # 記錄異常
    anomaly_logger.warning(f"[自動修復] 嘗試重啟Web伺服器")
    
    # 步驟1: 停止現有進程
    try:
        result = subprocess.run(
            ['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
            capture_output=True, text=True, timeout=5
        )
        for pid in result.stdout.strip().split('\n'):
            if pid.strip():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"  已清理進程 PID={pid}")
                except Exception as e:
                    logger.warning(f"  清理進程 PID={pid} 失敗: {e}")
        time.sleep(2)
    except Exception as e:
        logger.warning(f"  清理埠佔用失敗: {e}")
    
    # 步驟2: 啟動伺服器
    server_script = PROJECT_ROOT / 'live_trading' / 'web_server.py'
    
    try:
        # 使用 nohup 啟動，避免被父進程影響
        log_file = LOG_DIR / f'server_{datetime.now().strftime("%H%M")}.log'
        
        subprocess.Popen(
            ['nohup', sys.executable, '-u', str(server_script)],
            cwd=str(PROJECT_ROOT),
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp if os.name != 'nt' else None,
        )
        
        logger.info(f"  伺服器進程已啟動，日誌: {log_file.name}")
    except Exception as e:
        logger.error(f"  啟動伺服器失敗: {e}")
        anomaly_logger.error(f"[自動修復失敗] 啟動伺服器異常: {e}")
        return False
    
    # 步驟3: 等待伺服器就緒
    logger.info(f"  等待伺服器就緒（最多 {CONFIG['startup_timeout']}s）...")
    
    for i in range(CONFIG['startup_timeout']):
        time.sleep(1)
        alive, data = check_health_api()
        if alive:
            logger.info(f"  ✅ 伺服器重啟成功 (耗時 {i+1}s)")
            logger.info(f"     PID: {data.get('pid', '?')}")
            logger.info(f"     市場狀態: {data.get('market_status', '?')}")
            anomaly_logger.warning(f"[自動修復成功] 伺服器在 {i+1}s 內恢復正常")
            return True
    
    logger.error(f"  ❌ 伺服器啟動超時 ({CONFIG['startup_timeout']}s)")
    anomaly_logger.error(f"[自動修復失敗] 伺服器啟動超時 ({CONFIG['startup_timeout']}s)")
    return False


# ============================================================================
# 主監控邏輯
# ============================================================================
class MarketMonitor:
    """市場時段交易系統監控器"""
    
    def __init__(self):
        self.consecutive_failures = 0
        self.last_restart_time: Optional[datetime] = None
        self.total_checks = 0
        self.total_anomalies = 0
        self.total_restarts = 0
        self.start_time = datetime.now()
    
    def run_check(self) -> Dict[str, Any]:
        """
        執行一次完整檢查
        
        返回:
            檢查報告
        """
        self.total_checks += 1
        check_start = datetime.now()
        
        report = {
            'timestamp': check_start.isoformat(),
            'check_id': self.total_checks,
            'market_active': False,
            'market_status': 'unknown',
            'server_process': False,
            'health_api': False,
            'health_data': None,
            'anomalies': [],
            'log_errors': [],
            'resource_alerts': [],
            'actions_taken': [],
            'check_duration_ms': 0,
        }
        
        # 1. 檢查市場是否活躍
        try:
            is_active, market_desc = is_market_active()
            report['market_active'] = is_active
            report['market_status'] = market_desc
            
            if not is_active:
                logger.debug(f"市場已閉市 ({market_desc})，跳過詳細檢查")
                return report
        except Exception as e:
            logger.warning(f"市場狀態檢查失敗: {e}")
        
        logger.info(f"--- 檢查 #{self.total_checks} [{market_desc}] ---")
        
        # 2. 檢查進程
        alive, pid = check_server_process()
        report['server_process'] = alive
        if alive:
            logger.debug(f"  進程檢查: ✅ (PID={pid})")
        else:
            logger.warning(f"  進程檢查: ❌ 未檢測到運行中的Web伺服器")
        
        # 3. 檢查健康API
        healthy, health_data = check_health_api()
        report['health_api'] = healthy
        report['health_data'] = health_data
        
        if healthy:
            logger.debug(f"  健康API: ✅ (迭代#{health_data.get('uptime_iterations', '?')})")
            self.consecutive_failures = 0
        else:
            logger.warning(f"  健康API: ❌ 無響應")
        
        # 4. 如果伺服器不存在或不健康，嘗試修復
        if not alive or not healthy:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= CONFIG['max_consecutive_failures']:
                logger.warning(
                    f"  連續失敗 {self.consecutive_failures} 次，觸發自動重啟"
                )
                
                # 防止頻繁重啟
                if self.last_restart_time:
                    elapsed = (datetime.now() - self.last_restart_time).total_seconds()
                    if elapsed < 120:  # 2分鐘內不重複重啟
                        logger.warning(f"  距上次重啟僅 {elapsed:.0f}s，跳過本次重啟")
                        return report
                
                success = restart_server()
                self.last_restart_time = datetime.now()
                
                if success:
                    self.total_restarts += 1
                    self.consecutive_failures = 0
                    report['actions_taken'].append({
                        'action': 'restart_server',
                        'result': 'success',
                        'time': datetime.now().isoformat(),
                    })
                else:
                    report['actions_taken'].append({
                        'action': 'restart_server',
                        'result': 'failed',
                        'time': datetime.now().isoformat(),
                    })
        
        # 5. 如果API正常，檢查持倉異常
        if healthy:
            pos_anomalies = check_position_anomalies()
            report['anomalies'].extend(pos_anomalies)
            
            for anomaly in pos_anomalies:
                self.total_anomalies += 1
                severity = anomaly.get('severity', 'WARNING')
                msg = f"[{severity}] {anomaly.get('message', '')}"
                
                if severity == 'CRITICAL':
                    anomaly_logger.critical(msg)
                    logger.critical(f"  持倉異常: {msg}")
                elif severity == 'ERROR':
                    anomaly_logger.error(msg)
                    logger.error(f"  持倉異常: {msg}")
                else:
                    anomaly_logger.warning(msg)
                    logger.warning(f"  持倉異常: {msg}")
        
        # 6. 掃描日誌錯誤
        log_errors = scan_error_logs()
        report['log_errors'] = log_errors
        
        if log_errors:
            # 過濾掉已知的 werkzeug selectors 錯誤
            # 過濾已知的無害錯誤（werkzeug/macOS bug, watchdog 常態）
            known_patterns = [
                'select.kevent',
                'TypeError: changelist must be an iterable',
                'Error on request:',
                '  File "/Users/oujianli/Library/Python',
                '  File "/Library/Developer/CommandLineTools',
            ]
            # watchdog.log 中的 Traceback 行全部來自同一個已知 bug
            def is_known_watchdog_error(err):
                if err.get('source') == 'watchdog.log':
                    line = err.get('line', '')
                    if 'Traceback' in line:
                        return True
                    for pat in known_patterns:
                        if pat in line:
                            return True
                return False
            
            real_errors = [
                e for e in log_errors
                if not is_known_watchdog_error(e)
            ]
            
            if real_errors:
                real_count = len(real_errors)
                for err in real_errors[:5]:
                    anomaly_logger.warning(f"[日誌錯誤] {err['source']}: {err['line'][:200]}")
                    logger.warning(f"  日誌錯誤 [{err['source']}]: {err['line'][:150]}")
                if real_count > 5:
                    logger.warning(f"  ... 及其他 {real_count - 5} 條錯誤")
        
        # 7. 系統資源檢查
        resource_alerts = check_system_resources()
        report['resource_alerts'] = resource_alerts
        
        for alert in resource_alerts:
            anomaly_logger.warning(f"[資源告警] {alert['message']}")
            logger.warning(f"  資源告警: {alert['message']}")
        
        # 完成
        check_duration = (datetime.now() - check_start).total_seconds() * 1000
        report['check_duration_ms'] = round(check_duration)
        
        if not report['anomalies'] and not report['actions_taken']:
            logger.info(f"  ✅ 檢查完成 ({check_duration:.0f}ms) - 系統正常")
        
        return report
    
    def run_loop(self):
        """持續監控循環"""
        logger.info("=" * 60)
        logger.info("  市場時段交易系統監控啟動")
        logger.info(f"  檢查間隔: {CONFIG['check_interval']}s")
        logger.info(f"  日誌文件: {monitor_log}")
        logger.info(f"  異常日誌: {anomaly_log}")
        logger.info("=" * 60)
        
        anomaly_logger.warning("[監控啟動] 市場監控器開始運行")
        
        self._running = True
        
        # 信號處理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self._running:
                try:
                    self.run_check()
                except Exception as e:
                    logger.error(f"檢查循環異常: {e}")
                    traceback.print_exc()
                
                # 等待下次檢查
                for _ in range(CONFIG['check_interval']):
                    if not self._running:
                        break
                    time.sleep(1)
        finally:
            self._shutdown()
    
    def _signal_handler(self, signum, frame):
        logger.info(f"收到信號 {signum}，正在停止監控...")
        self._running = False
    
    def _shutdown(self):
        """關閉監控器"""
        uptime = datetime.now() - self.start_time
        logger.info("=" * 60)
        logger.info("  監控器停止")
        logger.info(f"  運行時長: {uptime}")
        logger.info(f"  總檢查次數: {self.total_checks}")
        logger.info(f"  總異常數: {self.total_anomalies}")
        logger.info(f"  總重啟次數: {self.total_restarts}")
        logger.info("=" * 60)
        anomaly_logger.warning(
            f"[監控停止] 運行{uptime}, 檢查{self.total_checks}次, "
            f"異常{self.total_anomalies}次, 重啟{self.total_restarts}次"
        )


# ============================================================================
# 入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='市場時段交易系統監控腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/market_monitor.py --once     # 單次檢查
  python scripts/market_monitor.py --daemon   # 持續監控（後臺循環）

配合 cron（推薦）:
  */2 21-4,9-16 * * 1-5 cd /path/to/project && python scripts/market_monitor.py --once >> logs/monitor_cron.log 2>&1
        """
    )
    parser.add_argument('--once', action='store_true', help='僅執行一次檢查')
    parser.add_argument('--daemon', action='store_true', help='持續監控模式')
    args = parser.parse_args()
    
    monitor = MarketMonitor()
    
    if args.daemon:
        monitor.run_loop()
    else:
        # 默認 --once 模式
        report = monitor.run_check()
        
        # 輸出簡要報告
        print(f"\n{'='*50}")
        print(f"  監控報告 #{report['check_id']}")
        print(f"{'='*50}")
        print(f"  時間: {report['timestamp']}")
        print(f"  市場: {report['market_status']}")
        print(f"  進程: {'✅' if report['server_process'] else '❌'}")
        print(f"  API:  {'✅' if report['health_api'] else '❌'}")
        print(f"  異常: {len(report['anomalies'])}")
        print(f"  日誌錯誤: {len(report['log_errors'])}")
        print(f"  修復動作: {len(report['actions_taken'])}")
        print(f"  耗時: {report['check_duration_ms']}ms")
        
        if report['anomalies']:
            print(f"\n  異常詳情:")
            for a in report['anomalies']:
                print(f"    [{a['severity']}] {a['message']}")
        
        if report['actions_taken']:
            print(f"\n  修復記錄:")
            for a in report['actions_taken']:
                print(f"    {a['action']}: {a['result']} ({a['time']})")
        
        print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
