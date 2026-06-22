#!/usr/bin/env python3
"""
交易系统守护进程 (Watchdog)

功能：
- 监控 Web 服务器进程是否存活
- 定期健康检查（HTTP GET /api/health）
- 崩溃自动重启（带指数退避）
- 内存/CPU监控告警
- 完整的事件日志
- 优雅关闭信号处理

启动方式:
    python live_trading/watchdog.py
    或
    python main.py watchdog
"""

import sys
import os
import time
import signal
import subprocess
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import urllib.request
import urllib.error
import subprocess as sp

# 配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | [Watchdog] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'watchdog.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('watchdog')

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    'server_host': '127.0.0.1',
    'server_port': 8080,
    'health_url': 'http://127.0.0.1:8080/api/health',
    'check_interval': 5,          # 健康检查间隔（秒）
    'startup_timeout': 30,        # 启动超时（秒）
    'max_restarts': 20,           # 1小时内最大重启次数
    'restart_window': 3600,       # 重启计数窗口（秒）
    'backoff_base': 2,            # 退避基数（秒）
    'max_backoff': 300,           # 最大退避（秒）
    'memory_limit_mb': 500,       # 内存告警阈值（MB）
    'alert_cooldown': 600,        # 同类型告警冷却（秒）
}


class Watchdog:
    """
    进程守护者
    
    持续监控 Web 服务器，确保它始终运行。
    
    工作流程:
    1. 启动 Web 服务器子进程
    2. 每 5 秒发送健康检查请求
    3. 如果服务器无响应 → 重启
    4. 如果崩溃 → 记录日志并重启
    5. 如果 1 小时内重启超过 20 次 → 停止并告警
    6. 捕获 SIGTERM/SIGINT → 优雅关闭
    """
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[datetime] = None
        self.total_restarts: int = 0
        self.restart_times: list = []  # 最近重启时间戳
        self.last_alerts: Dict[str, float] = {}
        self._running: bool = False
        self._current_backoff: int = 0
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理：优雅关闭"""
        logger.info(f"收到信号 {signum}，正在关闭...")
        self._running = False
    
    def start_server(self) -> bool:
        """
        启动 Web 服务器子进程
        
        返回:
            是否启动成功
        """
        if self.process and self.process.poll() is None:
            logger.warning("服务器已在运行，先停止旧进程")
            self.stop_server()
        
        # 清理占用端口的残留进程
        try:
            r = sp.run(['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
                       capture_output=True, text=True, timeout=5)
            for pid in r.stdout.strip().split('\n'):
                if pid.strip():
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"清理端口占用 PID={pid}")
            time.sleep(1)
        except Exception:
                logger.debug(f"Non-critical error in watchdog.py: {e}", exc_info=True)
        
        logger.info("正在启动 Web 服务器...")
        
        server_script = PROJECT_ROOT / 'live_trading' / 'web_server.py'
        
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # 创建新的进程组，避免被父进程信号影响
                preexec_fn=os.setpgrp if os.name != 'nt' else None,
            )
            
            self.start_time = datetime.now()
            
            # 等待服务器启动
            logger.info(f"等待服务器启动（最多 {CONFIG['startup_timeout']}s）...")
            
            for i in range(CONFIG['startup_timeout']):
                time.sleep(1)
                if self.health_check():
                    logger.info(f"✅ 服务器启动成功 (耗时 {i+1}s)")
                    logger.info(f"   地址: http://localhost:{CONFIG['server_port']}")
                    logger.info(f"   进程 PID: {self.process.pid}")
                    self._current_backoff = 0
                    return True
            
            # 超时：检查是否有错误输出
            stderr = self.process.stderr.read() if self.process.stderr else ''
            logger.error(f"服务器启动超时 ({CONFIG['startup_timeout']}s)")
            if stderr:
                logger.error(f"错误输出: {stderr[-500:]}")
            
            return False
            
        except Exception as e:
            logger.error(f"启动服务器失败: {e}")
            return False
    
    def stop_server(self) -> None:
        """停止服务器子进程"""
        if not self.process:
            return
        
        logger.info(f"正在停止服务器 (PID: {self.process.pid})...")
        
        try:
            # 先发送 SIGTERM
            self.process.terminate()
            
            # 等待 5 秒
            try:
                self.process.wait(timeout=5)
                logger.info("服务器已正常停止")
            except subprocess.TimeoutExpired:
                # 强制杀死
                logger.warning("服务器未响应 SIGTERM，强制杀死")
                self.process.kill()
                self.process.wait(timeout=3)
                logger.info("服务器已强制停止")
        except Exception as e:
            logger.error(f"停止服务器时出错: {e}")
        
        self.process = None
    
    def health_check(self) -> bool:
        """
        健康检查
        
        向 /api/health 发送 GET 请求，
        验证服务器是否正常响应。
        
        返回:
            是否健康
        """
        try:
            req = urllib.request.Request(
                CONFIG['health_url'],
                headers={'User-Agent': 'TradingWatchdog/1.0'}
            )
            
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    return data.get('status') == 'ok'
                
        except urllib.error.URLError as e:
            logger.debug(f"健康检查失败: {e.reason}")
        except Exception as e:
            logger.debug(f"健康检查异常: {e}")
        
        return False
    
    def check_memory(self) -> Optional[float]:
        """
        检查服务器进程内存使用
        
        返回:
            内存使用量（MB），或 None
        """
        if not self.process or self.process.poll() is not None:
            return None
        
        try:
            import psutil
            proc = psutil.Process(self.process.pid)
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            return mem_mb
        except ImportError:
            return None
        except Exception:
            return None
    
    def alert(self, alert_type: str, message: str) -> None:
        """
        发送告警（带去重）
        
        同一类型告警在冷却期内不会重复发送。
        
        参数:
            alert_type: 告警类型
            message: 告警消息
        """
        now = time.time()
        last = self.last_alerts.get(alert_type, 0)
        
        if now - last < CONFIG['alert_cooldown']:
            return  # 冷却中
        
        self.last_alerts[alert_type] = now
        
        # 写入告警日志
        alert_log = LOG_DIR / 'watchdog_alerts.log'
        with open(alert_log, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] [{alert_type}] {message}\n")
        
        logger.warning(f"🚨 告警 [{alert_type}]: {message}")
    
    def can_restart(self) -> bool:
        """
        检查是否可以重启（防止无限重启循环）
        
        规则：1小时内最多重启 20 次。
        
        返回:
            是否可以重启
        """
        now = time.time()
        window_start = now - CONFIG['restart_window']
        
        # 清理过期记录
        self.restart_times = [t for t in self.restart_times if t > window_start]
        
        if len(self.restart_times) >= CONFIG['max_restarts']:
            logger.critical(
                f"❌ 1小时内重启 {len(self.restart_times)} 次，超过上限 "
                f"{CONFIG['max_restarts']}，停止自动恢复！"
            )
            self.alert('MAX_RESTARTS', 
                       f'1小时内重启{len(self.restart_times)}次，已停止自动恢复，请手动检查')
            return False
        
        return True
    
    def get_backoff_delay(self) -> int:
        """
        计算指数退避延迟
        
        每次重启失败后，等待时间翻倍，
        从 2 秒开始，最高 300 秒。
        
        返回:
            等待秒数
        """
        delay = CONFIG['backoff_base'] * (2 ** min(self._current_backoff, 7))
        delay = min(delay, CONFIG['max_backoff'])
        return delay
    
    def run(self) -> None:
        """
        主守护循环
        
        持续监控并自动恢复服务器。
        """
        logger.info("=" * 60)
        logger.info("  交易系统守护进程 (Watchdog) 启动")
        logger.info(f"  监控地址: {CONFIG['health_url']}")
        logger.info(f"  检查间隔: {CONFIG['check_interval']}s")
        logger.info(f"  崩溃日志: {LOG_DIR / 'watchdog.log'}")
        logger.info("=" * 60)
        
        self._running = True
        
        # 首次启动服务器
        if not self.start_server():
            logger.error("首次启动失败")
            if not self.can_restart():
                return
            self._current_backoff += 1
        
        # 主循环
        while self._running:
            try:
                time.sleep(CONFIG['check_interval'])
                
                # 检查进程是否存活
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.returncode
                    
                    # 读取错误输出
                    stderr = ''
                    if self.process.stderr:
                        stderr = self.process.stderr.read()[-500:]
                    
                    logger.error(f"❌ 服务器进程退出 (exit code: {exit_code})")
                    if stderr:
                        logger.error(f"错误输出: {stderr}")
                    
                    self.total_restarts += 1
                    self.restart_times.append(time.time())
                    
                    # 检查是否可以重启
                    if not self.can_restart():
                        break
                    
                    self.alert('SERVER_CRASH', 
                               f'服务器崩溃(exit={exit_code})，第{self.total_restarts}次重启')
                    
                    # 退避等待
                    delay = self.get_backoff_delay()
                    logger.info(f"等待 {delay}s 后重启...")
                    time.sleep(delay)
                    
                    # 重启
                    if self.start_server():
                        self._current_backoff = 0
                    else:
                        self._current_backoff += 1
                        self.alert('RESTART_FAILED', f'第{self.total_restarts}次重启失败')
                    
                    continue
                
                # 健康检查
                if not self.health_check():
                    logger.warning("⚠️ 健康检查失败，服务器可能无响应")
                    
                    # 连续 3 次失败才重启（避免偶发网络波动）
                    consecutive_failures = 0
                    for _ in range(3):
                        time.sleep(2)
                        if self.health_check():
                            consecutive_failures = 0
                            break
                        consecutive_failures += 1
                    
                    if consecutive_failures >= 3:
                        logger.error("连续 3 次健康检查失败，准备重启")
                        self.stop_server()
                        
                        if self.can_restart():
                            self.total_restarts += 1
                            self.restart_times.append(time.time())
                            self.alert('HEALTH_FAIL', '连续健康检查失败，触发重启')
                            
                            if self.start_server():
                                self._current_backoff = 0
                            else:
                                self._current_backoff += 1
                    continue
                
                # 内存检查
                mem = self.check_memory()
                if mem and mem > CONFIG['memory_limit_mb']:
                    logger.warning(f"⚠️ 内存使用偏高: {mem:.0f}MB (阈值 {CONFIG['memory_limit_mb']}MB)")
                    self.alert('HIGH_MEMORY', f'内存使用 {mem:.0f}MB')
                
                # 健康时打印心跳
                uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
                logger.info(
                    f"💓 心跳正常 | 运行时间: {str(uptime).split('.')[0]} | "
                    f"重启次数: {self.total_restarts} | 内存: {mem:.0f}MB" if mem else ""
                )
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"守护循环异常: {e}", exc_info=True)
        
        # 清理
        self.stop_server()
        logger.info("守护进程已退出")
    
    def get_status(self) -> Dict[str, Any]:
        """获取守护进程状态"""
        return {
            'running': self._running,
            'server_alive': self.process is not None and self.process.poll() is None,
            'server_pid': self.process.pid if self.process else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'total_restarts': self.total_restarts,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
        }


# ============================================================================
# 入口
# ============================================================================
if __name__ == '__main__':
    wd = Watchdog()
    wd.run()
