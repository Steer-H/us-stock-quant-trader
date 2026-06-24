"""System resource monitoring."""

import logging
import threading
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

import numpy as np
import psutil  # 用於系統資源監控

from utils.helpers import safe_divide

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """健康狀態"""
    HEALTHY = 'HEALTHY'
    DEGRADED = 'DEGRADED'    # 性能下降
    UNHEALTHY = 'UNHEALTHY'  # 異常
    OFFLINE = 'OFFLINE'      # 離線


@dataclass
class HealthCheck:
    """單次健康檢查結果"""
    component: str
    status: HealthStatus
    message: str = ''
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ''
    latency_ms: float = 0.0


class SystemMonitor:
    """
    系統監控器
    
    持續監控系統資源使用、API連接狀態和交易行為。
    支持自定義檢查項和告警回調。
    """
    
    def __init__(self, check_interval_sec: int = 30):
        """
        參數:
            check_interval_sec: 檢查間隔（秒）
        """
        self.check_interval = check_interval_sec
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        
        # 健康狀態歷史
        self.health_history: deque = deque(maxlen=1000)
        
        # 告警去重（避免重複告警）
        self._last_alerts: Dict[str, float] = {}
        self._alert_cooldown: float = 300  # 5分鐘內不重複告警
        
        # API連接狀態
        self.api_connections: Dict[str, bool] = {
            'broker_api': False,
            'market_data': False,
        }
    
    def start(self) -> None:
        """啟動後臺監控線程"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name='system_monitor'
        )
        self._monitor_thread.start()
        logger.info("系統監控已啟動")
    
    def stop(self) -> None:
        """停止監控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("系統監控已停止")
    
    def add_callback(self, callback: Callable[[HealthCheck], None]) -> None:
        """
        添加告警回調
        
        參數:
            callback: 接收HealthCheck的回調函數
        """
        self._callbacks.append(callback)
    
    def _monitor_loop(self) -> None:
        """監控主循環"""
        while self._running:
            try:
                checks = self.run_all_checks()
                
                for check in checks:
                    if check.status != HealthStatus.HEALTHY:
                        self._handle_unhealthy(check)
                    
                    self.health_history.append(check)
                
            except Exception as e:
                logger.error(f"監控循環異常: {e}")
            
            time.sleep(self.check_interval)
    
    def run_all_checks(self) -> List[HealthCheck]:
        """執行全部健康檢查"""
        checks = []
        
        # 系統資源檢查
        checks.append(self._check_system_resources())
        
        # API連接檢查
        checks.append(self._check_api_connections())
        
        # 數據延遲檢查（需要外部注入數據）
        checks.append(self._check_data_freshness())
        
        return checks
    
    def _check_system_resources(self) -> HealthCheck:
        """
        檢查系統資源使用
        
        監控指標：
        - CPU使用率
        - 內存使用率
        - 磁碟可用空間
        """
        start = time.perf_counter()
        
        try:
            cpu_pct = psutil.cpu_percent(interval=1)
            mem_pct = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            metrics = {
                'cpu_percent': cpu_pct,
                'memory_percent': mem_pct,
                'disk_percent': disk_usage,
            }
            
            # 判斷健康狀態
            if cpu_pct > 90 or mem_pct > 90:
                status = HealthStatus.DEGRADED
                message = f"高資源使用: CPU={cpu_pct}%, MEM={mem_pct}%"
            elif cpu_pct > 70 or mem_pct > 80:
                status = HealthStatus.DEGRADED
                message = f"資源使用偏高: CPU={cpu_pct}%, MEM={mem_pct}%"
            else:
                status = HealthStatus.HEALTHY
                message = "系統資源正常"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"資源檢查失敗: {e}"
            metrics = {}
        
        return HealthCheck(
            component='system_resources',
            status=status,
            message=message,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            latency_ms=(time.perf_counter() - start) * 1000
        )
    
    def _check_api_connections(self) -> HealthCheck:
        """檢查API連接狀態"""
        all_connected = all(self.api_connections.values())
        disconnected = [k for k, v in self.api_connections.items() if not v]
        
        if all_connected:
            return HealthCheck(
                component='api_connections',
                status=HealthStatus.HEALTHY,
                message='所有API連接正常',
                timestamp=datetime.now().isoformat()
            )
        else:
            return HealthCheck(
                component='api_connections',
                status=HealthStatus.DEGRADED if len(disconnected) <= 1 else HealthStatus.UNHEALTHY,
                message=f'以下API斷開: {disconnected}',
                timestamp=datetime.now().isoformat()
            )
    
    def _check_data_freshness(self) -> HealthCheck:
        """
        檢查數據新鮮度
        
        如果最新數據時間戳超過閾值，說明數據流可能中斷。
        """
        # 此檢查需要外部注入最新數據時間戳
        # 簡化實現：返回健康狀態
        
        return HealthCheck(
            component='data_freshness',
            status=HealthStatus.HEALTHY,
            message='數據流正常（示例）',
            timestamp=datetime.now().isoformat()
        )
    
    def _handle_unhealthy(self, check: HealthCheck) -> None:
        """處理不健康狀態（觸發告警）"""
        alert_key = f"{check.component}:{check.status.value}"
        now = time.time()
        
        # 檢查告警冷卻期
        if alert_key in self._last_alerts:
            if now - self._last_alerts[alert_key] < self._alert_cooldown:
                return  # 冷卻期內，不重複告警
        
        self._last_alerts[alert_key] = now
        
        # 觸發回調
        for callback in self._callbacks:
            try:
                callback(check)
            except Exception as e:
                logger.error(f"告警回調異常: {e}")
        
        # 記錄告警
        from config.logging_config import LogManager
        risk_logger = LogManager.get_risk_logger()
        risk_logger.warning(f"[{check.component}] {check.status.value}: {check.message}")
    
    def update_api_status(self, name: str, connected: bool) -> None:
        """更新API連接狀態"""
        self.api_connections[name] = connected
    
    def get_status_summary(self) -> Dict[str, Any]:
        """獲取當前狀態摘要"""
        recent_checks = list(self.health_history)[-50:]  # 最近50次檢查
        
        unhealthy_count = sum(
            1 for c in recent_checks if c.status != HealthStatus.HEALTHY
        )
        
        return {
            'overall_status': HealthStatus.HEALTHY.value if unhealthy_count == 0 else 
                            HealthStatus.DEGRADED.value if unhealthy_count < 5 else 
                            HealthStatus.UNHEALTHY.value,
            'recent_checks': len(recent_checks),
            'unhealthy_count': unhealthy_count,
            'api_connections': dict(self.api_connections),
            'last_check_time': recent_checks[-1].timestamp if recent_checks else '',
        }
