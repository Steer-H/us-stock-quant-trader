#!/usr/bin/env python3
"""
Codex 自動化監控腳本

專為 Codex 自動化設計的美股量化交易系統監控。
在美股交易時段自動檢測系統健康狀態，發現異常生成結構化的狀態報告。

監控維度：
  1. Web 伺服器進程存活
  2. 健康檢查 API 響應
  3. 持倉數據異常（權益/回撤/集中度）
  4. 錯誤日誌增量掃描
  5. 系統資源（CPU/內存）
  6. 市場時鐘狀態

輸出格式：
  JSON 結構化狀態報告，方便 Codex 解析和展示。

用法：
  python scripts/codex_monitor.py              # 標準檢查
  python scripts/codex_monitor.py --notify     # 檢查並發送桌面通知
  python scripts/codex_monitor.py --json-only  # 僅輸出 JSON
"""

import sys
import os
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 精簡日誌
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [CodexMonitor] %(levelname)s: %(message)s',
    handlers=[logging.FileHandler(LOG_DIR / 'codex_monitor.log', encoding='utf-8')]
)
logger = logging.getLogger('codex_monitor')

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    'server_port': 8080,
    'health_url': 'http://127.0.0.1:8080/api/health',
    'status_url': 'http://127.0.0.1:8080/api/status',
    'request_timeout': 10,
    'startup_timeout': 45,
    'memory_warning_mb': 1024,
    'cpu_warning_pct': 80,
    'drawdown_critical_pct': -20,
    'drawdown_warning_pct': -10,
    'position_pnl_critical_pct': 50,
    'position_concentration_pct': 40,
}

# ============================================================================
# 市場時鐘檢查
# ============================================================================
def is_market_active() -> Tuple[bool, str]:
    """判斷當前是否處於美股交易相關時段"""
    try:
        from live_trading.market_clock import MarketClock, MarketStatus
        clock = MarketClock()
        status, desc = clock.get_status()
        active = status in {
            MarketStatus.PRE_MARKET,
            MarketStatus.REGULAR_HOURS,
            MarketStatus.AFTER_HOURS,
        }
        return active, desc
    except Exception as e:
        logger.warning(f"市場時鐘獲取失敗: {e}，默認按活躍處理")
        return True, "未知(默認活躍)"

# ============================================================================
# 各檢查項
# ============================================================================
def check_server_process() -> Dict[str, Any]:
    """檢查 Web 伺服器進程"""
    result = {'alive': False, 'pid': None, 'error': None}
    try:
        r = subprocess.run(
            ['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
            capture_output=True, text=True, timeout=5
        )
        pids = [p.strip() for p in r.stdout.strip().split('\n') if p.strip()]
        for pid in pids:
            try:
                pr = subprocess.run(
                    ['ps', '-p', pid, '-o', 'command='],
                    capture_output=True, text=True, timeout=3
                )
                if 'web_server' in pr.stdout:
                    result['alive'] = True
                    result['pid'] = int(pid)
                    return result
            except Exception:
                continue
        if pids:
            result['error'] = f'埠被非伺服器進程佔用: {pids}'
    except Exception as e:
        result['error'] = str(e)
    return result

def check_health_api() -> Dict[str, Any]:
    """檢查健康檢查 API"""
    import urllib.request
    import urllib.error
    result = {'ok': False, 'data': None, 'error': None, 'latency_ms': 0}
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            CONFIG['health_url'],
            headers={'User-Agent': 'CodexMonitor/1.0'}
        )
        with urllib.request.urlopen(req, timeout=CONFIG['request_timeout']) as resp:
            data = json.loads(resp.read().decode())
            result['ok'] = data.get('status') == 'ok'
            result['data'] = data
    except Exception as e:
        result['error'] = str(e)
    result['latency_ms'] = round((time.perf_counter() - start) * 1000)
    return result

def check_positions() -> List[Dict[str, Any]]:
    """檢查持倉異常"""
    import urllib.request
    anomalies = []
    try:
        req = urllib.request.Request(
            CONFIG['status_url'],
            headers={'User-Agent': 'CodexMonitor/1.0'}
        )
        with urllib.request.urlopen(req, timeout=CONFIG['request_timeout']) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return anomalies

    if data.get('waiting_for_open', False):
        return anomalies

    account = data.get('account', {})
    total_equity = account.get('total_equity', data.get('total_equity', 0))
    initial_capital = account.get('initial_capital', data.get('initial_capital', 100000))

    if total_equity <= 0:
        anomalies.append({
            'type': 'zero_equity', 'severity': 'CRITICAL',
            'message': f'總權益為0或負數: ${total_equity:,.2f}'
        })

    if initial_capital > 0:
        net_pnl_pct = (total_equity - initial_capital) / initial_capital * 100
        if net_pnl_pct < CONFIG['drawdown_critical_pct']:
            anomalies.append({
                'type': 'large_drawdown', 'severity': 'CRITICAL',
                'message': f'整體虧損超{abs(CONFIG["drawdown_critical_pct"])}%: {net_pnl_pct:.1f}%'
            })
        elif net_pnl_pct < CONFIG['drawdown_warning_pct']:
            anomalies.append({
                'type': 'drawdown_warning', 'severity': 'WARNING',
                'message': f'整體虧損超{abs(CONFIG["drawdown_warning_pct"])}%: {net_pnl_pct:.1f}%'
            })

    positions = data.get('positions', [])
    for pos in positions:
        ticker = pos.get('ticker', '?')
        unrealized = pos.get('unrealized_pnl_pct', 0)
        weight = pos.get('weight', 0)
        if abs(unrealized) > CONFIG['position_pnl_critical_pct']:
            anomalies.append({
                'type': 'position_extreme_pnl', 'severity': 'WARNING',
                'message': f'{ticker} 未實現盈虧 {unrealized:.1f}%'
            })
        if weight > CONFIG['position_concentration_pct']:
            anomalies.append({
                'type': 'position_concentration', 'severity': 'WARNING',
                'message': f'{ticker} 集中度 {weight:.1f}%'
            })

    if len(positions) > 50:
        anomalies.append({
            'type': 'too_many_positions', 'severity': 'WARNING',
            'message': f'持倉數量過多: {len(positions)}'
        })

    return anomalies

def scan_recent_errors() -> List[Dict[str, Any]]:
    """掃描最近的錯誤日誌（最近100行）"""
    errors = []
    log_files = [
        LOG_DIR / 'watchdog.log',
        LOG_DIR / 'server.log',
        LOG_DIR / 'error.log',
    ]
    server_logs = sorted(LOG_DIR.glob('server_*.log'))
    if server_logs:
        log_files.append(server_logs[-1])

    for lf in log_files:
        if not lf.exists():
            continue
        try:
            with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            recent = lines[-200:]
            for line in recent:
                if 'ERROR' in line or 'CRITICAL' in line or 'Traceback' in line:
                    errors.append({
                        'source': lf.name,
                        'line': line.strip()[:250]
                    })
        except Exception:
            pass
    return errors

def check_system_resources() -> List[Dict[str, Any]]:
    """檢查系統資源"""
    alerts = []
    try:
        import psutil
        mem_mb = psutil.virtual_memory().used / (1024 * 1024)
        cpu_pct = psutil.cpu_percent(interval=1)
        if mem_mb > CONFIG['memory_warning_mb']:
            alerts.append({
                'type': 'high_memory', 'severity': 'WARNING',
                'message': f'內存使用 {mem_mb:.0f}MB'
            })
        if cpu_pct > CONFIG['cpu_warning_pct']:
            alerts.append({
                'type': 'high_cpu', 'severity': 'WARNING',
                'message': f'CPU使用率 {cpu_pct:.1f}%'
            })
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"資源檢查失敗: {e}")
    return alerts

# ============================================================================
# 自動修復
# ============================================================================
def attempt_restart() -> Dict[str, Any]:
    """嘗試重啟伺服器"""
    result = {'action': 'restart_server', 'success': False, 'message': ''}
    try:
        subprocess.run(
            ['screen', '-S', 'trading_dashboard', '-X', 'quit'],
            capture_output=True, timeout=5
        )
        time.sleep(2)
        subprocess.run(
            ['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
            capture_output=True, timeout=3
        )
        subprocess.run(
            f'lsof -ti tcp:{CONFIG["server_port"]} | xargs kill -9',
            shell=True, capture_output=True, timeout=5
        )
        time.sleep(1)
        subprocess.Popen(
            ['screen', '-dmS', 'trading_dashboard',
             'python3', '-u', 'live_trading/web_server.py'],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(5)
        alive = check_server_process()
        if alive['alive']:
            result['success'] = True
            result['message'] = f"伺服器已自動重啟 (PID: {alive['pid']})"
        else:
            result['message'] = '重啟後伺服器仍未響應'
    except Exception as e:
        result['message'] = f'重啟失敗: {e}'
    return result

# ============================================================================
# 桌面通知
# ============================================================================
def send_desktop_notification(title: str, message: str, sound: bool = True):
    """發送 macOS 桌面通知"""
    try:
        script = f'display notification "{message}" with title "{title}"'
        if sound:
            script += ' sound name "default"'
        subprocess.run(['osascript', '-e', script], timeout=3)
    except Exception as e:
        logger.warning(f"桌面通知發送失敗: {e}")

# ============================================================================
# 主邏輯
# ============================================================================
def run_full_check() -> Dict[str, Any]:
    """執行完整檢查，返回結構化報告"""
    check_start = datetime.now()
    market_active, market_desc = is_market_active()

    report = {
        'timestamp': check_start.isoformat(),
        'market_active': market_active,
        'market_status': market_desc,
        'checks': {},
        'anomalies': [],
        'actions': [],
        'overall_status': 'HEALTHY',
    }

    # 1. 進程檢查
    proc = check_server_process()
    report['checks']['server_process'] = proc
    if not proc['alive']:
        report['anomalies'].append({
            'type': 'server_down', 'severity': 'CRITICAL',
            'message': f'Web伺服器未運行' + (f': {proc["error"]}' if proc.get('error') else '')
        })

    # 2. 健康檢查 API
    health = check_health_api()
    report['checks']['health_api'] = health
    if not health['ok']:
        report['anomalies'].append({
            'type': 'health_api_fail', 'severity': 'CRITICAL' if not proc['alive'] else 'WARNING',
            'message': f'健康檢查失敗: {health.get("error", "狀態異常")}'
        })

    # 3. 持倉異常
    pos_anomalies = check_positions()
    report['checks']['position_anomalies'] = pos_anomalies
    report['anomalies'].extend(pos_anomalies)

    # 4. 錯誤日誌
    log_errors = scan_recent_errors()
    report['checks']['log_errors'] = log_errors
    num_errors = len(log_errors)
    if num_errors > 0:
        report['anomalies'].append({
            'type': 'log_errors_found', 'severity': 'WARNING',
            'message': f'發現 {num_errors} 條錯誤日誌'
        })

    # 5. 系統資源
    resource_alerts = check_system_resources()
    report['checks']['resource_alerts'] = resource_alerts
    report['anomalies'].extend(resource_alerts)

    # 自動修復
    if not proc['alive']:
        if market_active:
            restart_result = attempt_restart()
            report['actions'].append(restart_result)
            if restart_result['success']:
                health_retry = check_health_api()
                if health_retry['ok']:
                    report['anomalies'] = [
                        a for a in report['anomalies']
                        if a['type'] not in ('server_down', 'health_api_fail')
                    ]
                    report['anomalies'].append({
                        'type': 'auto_recovered', 'severity': 'INFO',
                        'message': '伺服器已自動恢復'
                    })
        else:
            report['actions'].append({
                'action': 'skip_restart', 'message': '非交易時段，跳過自動重啟'
            })

    # 判斷整體狀態
    criticals = [a for a in report['anomalies'] if a['severity'] == 'CRITICAL']
    warnings = [a for a in report['anomalies'] if a['severity'] == 'WARNING']
    if criticals:
        report['overall_status'] = 'CRITICAL'
    elif warnings:
        report['overall_status'] = 'WARNING'

    report['check_duration_ms'] = round(
        (datetime.now() - check_start).total_seconds() * 1000
    )

    return report

def format_text_report(report: Dict[str, Any]) -> str:
    """將報告格式化為可讀文本"""
    lines = []
    lines.append("=" * 55)
    lines.append(f"  📊 量化交易系統監控報告")
    lines.append("=" * 55)
    lines.append(f"  時間: {report['timestamp']}")
    lines.append(f"  市場: {report['market_status']} {'🟢' if report['market_active'] else '⚫'}")

    status_icon = {'HEALTHY': '✅', 'WARNING': '⚠️', 'CRITICAL': '🚨'}
    lines.append(f"  狀態: {status_icon.get(report['overall_status'], '❓')} {report['overall_status']}")

    proc = report['checks']['server_process']
    lines.append(f"  進程: {'✅ 運行中' if proc['alive'] else '❌ 未運行'}")

    health = report['checks']['health_api']
    lines.append(f"  API:  {'✅ 正常' if health['ok'] else '❌ 異常'} ({health['latency_ms']}ms)")

    anomalies = report['anomalies']
    errors = report['checks']['log_errors']
    lines.append(f"  異常: {len(anomalies)} | 日誌錯誤: {len(errors)} | 耗時: {report['check_duration_ms']}ms")

    if anomalies:
        lines.append(f"\n  --- 異常詳情 ---")
        for a in anomalies:
            icon = {'CRITICAL': '🚨', 'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️'}
            lines.append(f"  {icon.get(a['severity'], '•')} [{a['severity']}] {a['message']}")

    actions = report.get('actions', [])
    if actions:
        lines.append(f"\n  --- 修復動作 ---")
        for a in actions:
            status = '✅' if a.get('success', True) else '❌'
            lines.append(f"  {status} {a.get('action', a.get('message', ''))}")

    lines.append("=" * 55)
    return '\n'.join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Codex 自動化交易系統監控')
    parser.add_argument('--notify', action='store_true', help='發現異常時發送桌面通知')
    parser.add_argument('--json-only', action='store_true', help='僅輸出 JSON 報告')
    parser.add_argument('--always-notify', action='store_true', help='無論是否有異常都發送通知')
    args = parser.parse_args()

    report = run_full_check()

    if args.json_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_text_report(report))

    # 桌面通知
    should_notify = args.always_notify or (
        args.notify and report['overall_status'] != 'HEALTHY'
    )
    if should_notify:
        anomaly_count = len(report['anomalies'])
        if anomaly_count > 0:
            criticals = [a for a in report['anomalies'] if a['severity'] == 'CRITICAL']
            if criticals:
                send_desktop_notification(
                    '🚨 量化交易系統異常',
                    f'{critical[0]["message"]}（共{anomaly_count}項異常）'
                )
            else:
                send_desktop_notification(
                    '⚠️ 量化交易系統告警',
                    f'發現{anomaly_count}項異常'
                )
        elif args.always_notify:
            send_desktop_notification(
                '✅ 量化交易系統正常',
                f'所有檢查通過 ({report["check_duration_ms"]}ms)'
            )

    # 退出碼：有嚴重異常時非0
    if report['overall_status'] == 'CRITICAL':
        sys.exit(2)
    elif report['overall_status'] == 'WARNING':
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
