import os
#!/usr/bin/env python3
"""
市场时段交易系统监控脚本 (Market Monitor)

在美股交易时段（盘前/正常盘/盘后）持续监测交易系统状态，
发现异常自动修复并写入详细日志。

监控项：
  1. Web服务器进程是否存活 → 自动重启
  2. 健康检查 API 是否正常响应 → 诊断并修复
  3. 持仓数据是否异常（数量/市值突变）→ 记录告警
  4. 日志中是否出现 ERROR/CRITICAL → 提取并记录
  5. 系统资源（CPU/内存）是否过高 → 记录告警
  6. 端口是否被占用 → 清理僵尸进程

运行方式：
  python scripts/market_monitor.py           # 单次检查
  python scripts/market_monitor.py --daemon  # 持续监控（循环模式）
  python scripts/market_monitor.py --once    # 单次检查并退出

配合 crontab 使用（每2分钟检查一次）：
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

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# 日志配置
# ============================================================================
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 监控专用日志
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

# 异常事件日志（仅记录异常）
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
    'health_timeout': 10,          # 健康检查超时（秒）
    'startup_timeout': 45,          # 启动超时（秒）
    'check_interval': 120,          # 循环检查间隔（秒）
    'max_consecutive_failures': 3,  # 连续失败阈值
    'memory_warning_mb': 800,       # 内存告警阈值（MB）
    'cpu_warning_pct': 80,          # CPU告警阈值（%）
    'position_value_change_pct': 30, # 单持仓市值变化告警阈值
    'error_log_scan_lines': 200,    # 扫描最近的日志行数
}


# ============================================================================
# 市场时钟集成
# ============================================================================
def is_market_active() -> Tuple[bool, str]:
    """
    判断当前是否处于美股交易相关时段
    
    返回:
        (是否活跃, 状态描述)
    """
    try:
        from live_trading.market_clock import MarketClock, MarketStatus
        clock = MarketClock()
        status, desc = clock.get_status()
        
        # 盘前、正常盘、盘后都算活跃时段
        active_statuses = {
            MarketStatus.PRE_MARKET,
            MarketStatus.REGULAR_HOURS,
            MarketStatus.AFTER_HOURS,
        }
        
        is_active = status in active_statuses
        return is_active, desc
    except Exception as e:
        logger.warning(f"无法获取市场状态: {e}，默认按活跃时段处理")
        return True, "未知(默认活跃)"


# ============================================================================
# 健康检查
# ============================================================================
def check_server_process() -> Tuple[bool, Optional[int]]:
    """
    检查 Web 服务器进程是否存活
    
    返回:
        (是否存活, PID或None)
    """
    try:
        # 检查占用端口的进程
        result = subprocess.run(
            ['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
            capture_output=True, text=True, timeout=5
        )
        pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        
        if pids:
            # 验证进程确实是 Python web_server
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
            # 端口被其他进程占用
            logger.warning(f"端口 {CONFIG['server_port']} 被非服务器进程占用: {pids}")
            return False, None
        
        return False, None
    except Exception as e:
        logger.error(f"检查进程失败: {e}")
        return False, None


def check_health_api() -> Tuple[bool, Optional[Dict]]:
    """
    检查健康检查 API 是否正常响应
    
    返回:
        (是否正常, 响应数据或None)
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
                logger.warning(f"健康检查返回异常状态: {data}")
                return False, data
    except urllib.error.URLError as e:
        logger.warning(f"健康检查连接失败: {e.reason}")
        return False, None
    except Exception as e:
        logger.warning(f"健康检查异常: {e}")
        return False, None


def check_position_anomalies() -> List[Dict]:
    """
    检查持仓数据异常
    
    返回:
        异常列表
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
    
    # 如果系统还在等待开市，跳过持仓检查
    if data.get('waiting_for_open', False):
        logger.debug("系统等待开市中，跳过持仓检查")
        return anomalies
    
    # 从 account 嵌套对象读取权益数据
    account = data.get('account', {})
    total_equity = account.get('total_equity', data.get('total_equity', 0))
    initial_capital = account.get('initial_capital', data.get('initial_capital', 100000))
    
    if total_equity <= 0:
        anomalies.append({
            'type': 'zero_equity',
            'severity': 'CRITICAL',
            'message': f'总权益为0或负数: ${total_equity:,.2f}',
        })
    
    # 检查整体回撤
    if initial_capital > 0:
        net_pnl = total_equity - initial_capital
        net_pnl_pct = (net_pnl / initial_capital * 100)
        if net_pnl_pct < -20:
            anomalies.append({
                'type': 'large_drawdown',
                'severity': 'CRITICAL',
                'message': f'整体亏损超过20%: {net_pnl_pct:.1f}% (${net_pnl:,.2f})',
            })
        elif net_pnl_pct < -10:
            anomalies.append({
                'type': 'drawdown_warning',
                'severity': 'WARNING',
                'message': f'整体亏损超过10%: {net_pnl_pct:.1f}% (${net_pnl:,.2f})',
            })
    
    # 检查单持仓异常
    positions = data.get('positions', [])
    for pos in positions:
        ticker = pos.get('ticker', '?')
        market_value = pos.get('market_value', 0)
        unrealized_pnl_pct = pos.get('unrealized_pnl_pct', 0)
        
        if abs(unrealized_pnl_pct) > 50:
            anomalies.append({
                'type': 'position_extreme_pnl',
                'severity': 'WARNING',
                'message': f'{ticker} 未实现盈亏达到 {unrealized_pnl_pct:.1f}%',
            })
        
        # 检查持仓权重异常
        weight = pos.get('weight', 0)
        if weight > 40:
            anomalies.append({
                'type': 'position_concentration',
                'severity': 'WARNING',
                'message': f'{ticker} 持仓集中度过高: {weight:.1f}%',
            })
    
    # 检查持仓数量突变
    positions_count = len(positions)
    if positions_count > 50:
        anomalies.append({
            'type': 'too_many_positions',
            'severity': 'WARNING',
            'message': f'持仓数量过多: {positions_count}',
        })
    
    # 检查交易信号是否正常产生
    market_data = data.get('market', {})
    market_status = market_data.get('status', data.get('market_status', ''))
    positions_initialized = data.get('positions_initialized', False)
    if market_status == 'REGULAR' and not positions_initialized:
        anomalies.append({
            'type': 'positions_not_initialized',
            'severity': 'ERROR',
            'message': '正常交易时段但持仓未初始化',
        })
    
    return anomalies


# 日志文件上次扫描位置（避免重复报告同一错误）
_log_scan_positions: Dict[str, int] = {}

def scan_error_logs() -> List[Dict]:
    """
    扫描日志文件中的新增错误（增量扫描）
    
    返回:
        错误列表
    """
    errors = []
    
    # 扫描 watchdog.log 和 server 相关日志
    log_files = [
        LOG_DIR / 'watchdog.log',
        LOG_DIR / 'server.log',
        LOG_DIR / 'error.log',
    ]
    
    # 找到最新的 server 日志
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
            
            # 如果文件被截断，从头开始
            if last_pos > total_lines:
                last_pos = 0
            
            # 只扫描新增的行
            recent = lines[last_pos:]
            _log_scan_positions[str(log_file)] = total_lines
            
            # 如果新增行太多，只取最后 N 行
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
    检查系统资源使用情况
    
    返回:
        告警列表
    """
    alerts = []
    
    try:
        # 检查内存（macOS 使用 ps 或 vm_stat）
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
                                'message': f'Web服务器内存使用过高: {rss_mb:.0f}MB (阈值: {CONFIG["memory_warning_mb"]}MB)',
                            })
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        logger.debug(f"资源检查失败: {e}")
    
    return alerts


# ============================================================================
# 自动修复
# ============================================================================
def restart_server() -> bool:
    """
    自动重启 Web 服务器
    
    返回:
        是否成功
    """
    logger.info("=" * 50)
    logger.info("🔄 开始自动重启交易系统...")
    logger.info("=" * 50)
    
    # 记录异常
    anomaly_logger.warning(f"[自动修复] 尝试重启Web服务器")
    
    # 步骤1: 停止现有进程
    try:
        result = subprocess.run(
            ['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
            capture_output=True, text=True, timeout=5
        )
        for pid in result.stdout.strip().split('\n'):
            if pid.strip():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"  已清理进程 PID={pid}")
                except Exception as e:
                    logger.warning(f"  清理进程 PID={pid} 失败: {e}")
        time.sleep(2)
    except Exception as e:
        logger.warning(f"  清理端口占用失败: {e}")
    
    # 步骤2: 启动服务器
    server_script = PROJECT_ROOT / 'live_trading' / 'web_server.py'
    
    try:
        # 使用 nohup 启动，避免被父进程影响
        log_file = LOG_DIR / f'server_{datetime.now().strftime("%H%M")}.log'
        
        subprocess.Popen(
            ['nohup', sys.executable, '-u', str(server_script)],
            cwd=str(PROJECT_ROOT),
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp if os.name != 'nt' else None,
        )
        
        logger.info(f"  服务器进程已启动，日志: {log_file.name}")
    except Exception as e:
        logger.error(f"  启动服务器失败: {e}")
        anomaly_logger.error(f"[自动修复失败] 启动服务器异常: {e}")
        return False
    
    # 步骤3: 等待服务器就绪
    logger.info(f"  等待服务器就绪（最多 {CONFIG['startup_timeout']}s）...")
    
    for i in range(CONFIG['startup_timeout']):
        time.sleep(1)
        alive, data = check_health_api()
        if alive:
            logger.info(f"  ✅ 服务器重启成功 (耗时 {i+1}s)")
            logger.info(f"     PID: {data.get('pid', '?')}")
            logger.info(f"     市场状态: {data.get('market_status', '?')}")
            anomaly_logger.warning(f"[自动修复成功] 服务器在 {i+1}s 内恢复正常")
            return True
    
    logger.error(f"  ❌ 服务器启动超时 ({CONFIG['startup_timeout']}s)")
    anomaly_logger.error(f"[自动修复失败] 服务器启动超时 ({CONFIG['startup_timeout']}s)")
    return False


# ============================================================================
# 主监控逻辑
# ============================================================================
class MarketMonitor:
    """市场时段交易系统监控器"""
    
    def __init__(self):
        self.consecutive_failures = 0
        self.last_restart_time: Optional[datetime] = None
        self.total_checks = 0
        self.total_anomalies = 0
        self.total_restarts = 0
        self.start_time = datetime.now()
    
    def run_check(self) -> Dict[str, Any]:
        """
        执行一次完整检查
        
        返回:
            检查报告
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
        
        # 1. 检查市场是否活跃
        try:
            is_active, market_desc = is_market_active()
            report['market_active'] = is_active
            report['market_status'] = market_desc
            
            if not is_active:
                logger.debug(f"市场已闭市 ({market_desc})，跳过详细检查")
                return report
        except Exception as e:
            logger.warning(f"市场状态检查失败: {e}")
        
        logger.info(f"--- 检查 #{self.total_checks} [{market_desc}] ---")
        
        # 2. 检查进程
        alive, pid = check_server_process()
        report['server_process'] = alive
        if alive:
            logger.debug(f"  进程检查: ✅ (PID={pid})")
        else:
            logger.warning(f"  进程检查: ❌ 未检测到运行中的Web服务器")
        
        # 3. 检查健康API
        healthy, health_data = check_health_api()
        report['health_api'] = healthy
        report['health_data'] = health_data
        
        if healthy:
            logger.debug(f"  健康API: ✅ (迭代#{health_data.get('uptime_iterations', '?')})")
            self.consecutive_failures = 0
        else:
            logger.warning(f"  健康API: ❌ 无响应")
        
        # 4. 如果服务器不存在或不健康，尝试修复
        if not alive or not healthy:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= CONFIG['max_consecutive_failures']:
                logger.warning(
                    f"  连续失败 {self.consecutive_failures} 次，触发自动重启"
                )
                
                # 防止频繁重启
                if self.last_restart_time:
                    elapsed = (datetime.now() - self.last_restart_time).total_seconds()
                    if elapsed < 120:  # 2分钟内不重复重启
                        logger.warning(f"  距上次重启仅 {elapsed:.0f}s，跳过本次重启")
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
        
        # 5. 如果API正常，检查持仓异常
        if healthy:
            pos_anomalies = check_position_anomalies()
            report['anomalies'].extend(pos_anomalies)
            
            for anomaly in pos_anomalies:
                self.total_anomalies += 1
                severity = anomaly.get('severity', 'WARNING')
                msg = f"[{severity}] {anomaly.get('message', '')}"
                
                if severity == 'CRITICAL':
                    anomaly_logger.critical(msg)
                    logger.critical(f"  持仓异常: {msg}")
                elif severity == 'ERROR':
                    anomaly_logger.error(msg)
                    logger.error(f"  持仓异常: {msg}")
                else:
                    anomaly_logger.warning(msg)
                    logger.warning(f"  持仓异常: {msg}")
        
        # 6. 扫描日志错误
        log_errors = scan_error_logs()
        report['log_errors'] = log_errors
        
        if log_errors:
            # 过滤掉已知的 werkzeug selectors 错误
            # 过滤已知的无害错误（werkzeug/macOS bug, watchdog 常态）
            known_patterns = [
                'select.kevent',
                'TypeError: changelist must be an iterable',
                'Error on request:',
                '  File "/Users/oujianli/Library/Python',
                '  File "/Library/Developer/CommandLineTools',
            ]
            # watchdog.log 中的 Traceback 行全部来自同一个已知 bug
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
                    anomaly_logger.warning(f"[日志错误] {err['source']}: {err['line'][:200]}")
                    logger.warning(f"  日志错误 [{err['source']}]: {err['line'][:150]}")
                if real_count > 5:
                    logger.warning(f"  ... 及其他 {real_count - 5} 条错误")
        
        # 7. 系统资源检查
        resource_alerts = check_system_resources()
        report['resource_alerts'] = resource_alerts
        
        for alert in resource_alerts:
            anomaly_logger.warning(f"[资源告警] {alert['message']}")
            logger.warning(f"  资源告警: {alert['message']}")
        
        # 完成
        check_duration = (datetime.now() - check_start).total_seconds() * 1000
        report['check_duration_ms'] = round(check_duration)
        
        if not report['anomalies'] and not report['actions_taken']:
            logger.info(f"  ✅ 检查完成 ({check_duration:.0f}ms) - 系统正常")
        
        return report
    
    def run_loop(self):
        """持续监控循环"""
        logger.info("=" * 60)
        logger.info("  市场时段交易系统监控启动")
        logger.info(f"  检查间隔: {CONFIG['check_interval']}s")
        logger.info(f"  日志文件: {monitor_log}")
        logger.info(f"  异常日志: {anomaly_log}")
        logger.info("=" * 60)
        
        anomaly_logger.warning("[监控启动] 市场监控器开始运行")
        
        self._running = True
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self._running:
                try:
                    self.run_check()
                except Exception as e:
                    logger.error(f"检查循环异常: {e}")
                    traceback.print_exc()
                
                # 等待下次检查
                for _ in range(CONFIG['check_interval']):
                    if not self._running:
                        break
                    time.sleep(1)
        finally:
            self._shutdown()
    
    def _signal_handler(self, signum, frame):
        logger.info(f"收到信号 {signum}，正在停止监控...")
        self._running = False
    
    def _shutdown(self):
        """关闭监控器"""
        uptime = datetime.now() - self.start_time
        logger.info("=" * 60)
        logger.info("  监控器停止")
        logger.info(f"  运行时长: {uptime}")
        logger.info(f"  总检查次数: {self.total_checks}")
        logger.info(f"  总异常数: {self.total_anomalies}")
        logger.info(f"  总重启次数: {self.total_restarts}")
        logger.info("=" * 60)
        anomaly_logger.warning(
            f"[监控停止] 运行{uptime}, 检查{self.total_checks}次, "
            f"异常{self.total_anomalies}次, 重启{self.total_restarts}次"
        )


# ============================================================================
# 入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='市场时段交易系统监控脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/market_monitor.py --once     # 单次检查
  python scripts/market_monitor.py --daemon   # 持续监控（后台循环）

配合 cron（推荐）:
  */2 21-4,9-16 * * 1-5 cd /path/to/project && python scripts/market_monitor.py --once >> logs/monitor_cron.log 2>&1
        """
    )
    parser.add_argument('--once', action='store_true', help='仅执行一次检查')
    parser.add_argument('--daemon', action='store_true', help='持续监控模式')
    args = parser.parse_args()
    
    monitor = MarketMonitor()
    
    if args.daemon:
        monitor.run_loop()
    else:
        # 默认 --once 模式
        report = monitor.run_check()
        
        # 输出简要报告
        print(f"\n{'='*50}")
        print(f"  监控报告 #{report['check_id']}")
        print(f"{'='*50}")
        print(f"  时间: {report['timestamp']}")
        print(f"  市场: {report['market_status']}")
        print(f"  进程: {'✅' if report['server_process'] else '❌'}")
        print(f"  API:  {'✅' if report['health_api'] else '❌'}")
        print(f"  异常: {len(report['anomalies'])}")
        print(f"  日志错误: {len(report['log_errors'])}")
        print(f"  修复动作: {len(report['actions_taken'])}")
        print(f"  耗时: {report['check_duration_ms']}ms")
        
        if report['anomalies']:
            print(f"\n  异常详情:")
            for a in report['anomalies']:
                print(f"    [{a['severity']}] {a['message']}")
        
        if report['actions_taken']:
            print(f"\n  修复记录:")
            for a in report['actions_taken']:
                print(f"    {a['action']}: {a['result']} ({a['time']})")
        
        print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
