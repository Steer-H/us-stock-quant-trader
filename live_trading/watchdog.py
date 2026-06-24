#!/usr/bin/env python3
"""
交易系統守護進程 (Watchdog)

功能：
- 監控 Web 伺服器進程是否存活
- 定期健康檢查（HTTP GET /api/health）
- 崩潰自動重啟（帶指數退避）
- 內存/CPU監控告警
- 完整的事件日誌
- 優雅關閉信號處理

啟動方式:
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

# 日誌配置
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
    'check_interval': 5,          # 健康檢查間隔（秒）
    'startup_timeout': 30,        # 啟動超時（秒）
    'max_restarts': 20,           # 1小時內最大重啟次數
    'restart_window': 3600,       # 重啟計數窗口（秒）
    'backoff_base': 2,            # 退避基數（秒）
    'max_backoff': 300,           # 最大退避（秒）
    'memory_limit_mb': 500,       # 內存告警閾值（MB）
    'alert_cooldown': 600,        # 同類型告警冷卻（秒）
}


class Watchdog:
    """
    進程守護者
    
    持續監控 Web 伺服器，確保它始終運行。
    
    工作流程:
    1. 啟動 Web 伺服器子進程
    2. 每 5 秒發送健康檢查請求
    3. 如果伺服器無響應 → 重啟
    4. 如果崩潰 → 記錄日誌並重啟
    5. 如果 1 小時內重啟超過 20 次 → 停止並告警
    6. 捕獲 SIGTERM/SIGINT → 優雅關閉
    """
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[datetime] = None
        self.total_restarts: int = 0
        self.restart_times: list = []  # 最近重啟時間戳
        self.last_alerts: Dict[str, float] = {}
        self._running: bool = False
        self._current_backoff: int = 0
        
        # 註冊信號處理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信號處理：優雅關閉"""
        logger.info(f"收到信號 {signum}，正在關閉...")
        self._running = False
    
    def start_server(self) -> bool:
        """
        啟動 Web 伺服器子進程
        
        返回:
            是否啟動成功
        """
        if self.process and self.process.poll() is None:
            logger.warning("伺服器已在運行，先停止舊進程")
            self.stop_server()
        
        # 清理佔用埠的殘留進程
        try:
            r = sp.run(['lsof', '-ti', f'tcp:{CONFIG["server_port"]}'],
                       capture_output=True, text=True, timeout=5)
            for pid in r.stdout.strip().split('\n'):
                if pid.strip():
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"清理埠佔用 PID={pid}")
            time.sleep(1)
        except Exception:
                logger.debug(f"Non-critical error in watchdog.py: {e}", exc_info=True)
        
        logger.info("正在啟動 Web 伺服器...")
        
        server_script = PROJECT_ROOT / 'live_trading' / 'web_server.py'
        
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # 創建新的進程組，避免被父進程信號影響
                preexec_fn=os.setpgrp if os.name != 'nt' else None,
            )
            
            self.start_time = datetime.now()
            
            # 等待伺服器啟動
            logger.info(f"等待伺服器啟動（最多 {CONFIG['startup_timeout']}s）...")
            
            for i in range(CONFIG['startup_timeout']):
                time.sleep(1)
                if self.health_check():
                    logger.info(f"✅ 伺服器啟動成功 (耗時 {i+1}s)")
                    logger.info(f"   地址: http://localhost:{CONFIG['server_port']}")
                    logger.info(f"   進程 PID: {self.process.pid}")
                    self._current_backoff = 0
                    return True
            
            # 超時：檢查是否有錯誤輸出
            stderr = self.process.stderr.read() if self.process.stderr else ''
            logger.error(f"伺服器啟動超時 ({CONFIG['startup_timeout']}s)")
            if stderr:
                logger.error(f"錯誤輸出: {stderr[-500:]}")
            
            return False
            
        except Exception as e:
            logger.error(f"啟動伺服器失敗: {e}")
            return False
    
    def stop_server(self) -> None:
        """停止伺服器子進程"""
        if not self.process:
            return
        
        logger.info(f"正在停止伺服器 (PID: {self.process.pid})...")
        
        try:
            # 先發送 SIGTERM
            self.process.terminate()
            
            # 等待 5 秒
            try:
                self.process.wait(timeout=5)
                logger.info("伺服器已正常停止")
            except subprocess.TimeoutExpired:
                # 強制殺死
                logger.warning("伺服器未響應 SIGTERM，強制殺死")
                self.process.kill()
                self.process.wait(timeout=3)
                logger.info("伺服器已強制停止")
        except Exception as e:
            logger.error(f"停止伺服器時出錯: {e}")
        
        self.process = None
    
    def health_check(self) -> bool:
        """
        健康檢查
        
        向 /api/health 發送 GET 請求，
        驗證伺服器是否正常響應。
        
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
            logger.debug(f"健康檢查失敗: {e.reason}")
        except Exception as e:
            logger.debug(f"健康檢查異常: {e}")
        
        return False
    
    def check_memory(self) -> Optional[float]:
        """
        檢查伺服器進程內存使用
        
        返回:
            內存使用量（MB），或 None
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
        發送告警（帶去重）
        
        同一類型告警在冷卻期內不會重複發送。
        
        參數:
            alert_type: 告警類型
            message: 告警消息
        """
        now = time.time()
        last = self.last_alerts.get(alert_type, 0)
        
        if now - last < CONFIG['alert_cooldown']:
            return  # 冷卻中
        
        self.last_alerts[alert_type] = now
        
        # 寫入告警日誌
        alert_log = LOG_DIR / 'watchdog_alerts.log'
        with open(alert_log, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] [{alert_type}] {message}\n")
        
        logger.warning(f"🚨 告警 [{alert_type}]: {message}")
    
    def can_restart(self) -> bool:
        """
        檢查是否可以重啟（防止無限重啟循環）
        
        規則：1小時內最多重啟 20 次。
        
        返回:
            是否可以重啟
        """
        now = time.time()
        window_start = now - CONFIG['restart_window']
        
        # 清理過期記錄
        self.restart_times = [t for t in self.restart_times if t > window_start]
        
        if len(self.restart_times) >= CONFIG['max_restarts']:
            logger.critical(
                f"❌ 1小時內重啟 {len(self.restart_times)} 次，超過上限 "
                f"{CONFIG['max_restarts']}，停止自動恢復！"
            )
            self.alert('MAX_RESTARTS', 
                       f'1小時內重啟{len(self.restart_times)}次，已停止自動恢復，請手動檢查')
            return False
        
        return True
    
    def get_backoff_delay(self) -> int:
        """
        計算指數退避延遲
        
        每次重啟失敗後，等待時間翻倍，
        從 2 秒開始，最高 300 秒。
        
        返回:
            等待秒數
        """
        delay = CONFIG['backoff_base'] * (2 ** min(self._current_backoff, 7))
        delay = min(delay, CONFIG['max_backoff'])
        return delay
    
    def run(self) -> None:
        """
        主守護循環
        
        持續監控並自動恢復伺服器。
        """
        logger.info("=" * 60)
        logger.info("  交易系統守護進程 (Watchdog) 啟動")
        logger.info(f"  監控地址: {CONFIG['health_url']}")
        logger.info(f"  檢查間隔: {CONFIG['check_interval']}s")
        logger.info(f"  崩潰日誌: {LOG_DIR / 'watchdog.log'}")
        logger.info("=" * 60)
        
        self._running = True
        
        # 首次啟動伺服器
        if not self.start_server():
            logger.error("首次啟動失敗")
            if not self.can_restart():
                return
            self._current_backoff += 1
        
        # 主循環
        while self._running:
            try:
                time.sleep(CONFIG['check_interval'])
                
                # 檢查進程是否存活
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.returncode
                    
                    # 讀取錯誤輸出
                    stderr = ''
                    if self.process.stderr:
                        stderr = self.process.stderr.read()[-500:]
                    
                    logger.error(f"❌ 伺服器進程退出 (exit code: {exit_code})")
                    if stderr:
                        logger.error(f"錯誤輸出: {stderr}")
                    
                    self.total_restarts += 1
                    self.restart_times.append(time.time())
                    
                    # 檢查是否可以重啟
                    if not self.can_restart():
                        break
                    
                    self.alert('SERVER_CRASH', 
                               f'伺服器崩潰(exit={exit_code})，第{self.total_restarts}次重啟')
                    
                    # 退避等待
                    delay = self.get_backoff_delay()
                    logger.info(f"等待 {delay}s 後重啟...")
                    time.sleep(delay)
                    
                    # 重啟
                    if self.start_server():
                        self._current_backoff = 0
                    else:
                        self._current_backoff += 1
                        self.alert('RESTART_FAILED', f'第{self.total_restarts}次重啟失敗')
                    
                    continue
                
                # 健康檢查
                if not self.health_check():
                    logger.warning("⚠️ 健康檢查失敗，伺服器可能無響應")
                    
                    # 連續 3 次失敗才重啟（避免偶發網絡波動）
                    consecutive_failures = 0
                    for _ in range(3):
                        time.sleep(2)
                        if self.health_check():
                            consecutive_failures = 0
                            break
                        consecutive_failures += 1
                    
                    if consecutive_failures >= 3:
                        logger.error("連續 3 次健康檢查失敗，準備重啟")
                        self.stop_server()
                        
                        if self.can_restart():
                            self.total_restarts += 1
                            self.restart_times.append(time.time())
                            self.alert('HEALTH_FAIL', '連續健康檢查失敗，觸發重啟')
                            
                            if self.start_server():
                                self._current_backoff = 0
                            else:
                                self._current_backoff += 1
                    continue
                
                # 內存檢查
                mem = self.check_memory()
                if mem and mem > CONFIG['memory_limit_mb']:
                    logger.warning(f"⚠️ 內存使用偏高: {mem:.0f}MB (閾值 {CONFIG['memory_limit_mb']}MB)")
                    self.alert('HIGH_MEMORY', f'內存使用 {mem:.0f}MB')
                
                # 健康時列印心跳
                uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
                logger.info(
                    f"💓 心跳正常 | 運行時間: {str(uptime).split('.')[0]} | "
                    f"重啟次數: {self.total_restarts} | 內存: {mem:.0f}MB" if mem else ""
                )
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"守護循環異常: {e}", exc_info=True)
        
        # 清理
        self.stop_server()
        logger.info("守護進程已退出")
    
    def get_status(self) -> Dict[str, Any]:
        """獲取守護進程狀態"""
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
