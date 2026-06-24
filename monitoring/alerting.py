"""Alerting and notification."""

import logging
import smtplib
import json
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警級別"""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


class AlertChannel(Enum):
    """告警通道"""
    CONSOLE = 'console'
    TELEGRAM = 'telegram'
    EMAIL = 'email'
    SMS = 'sms'


@dataclass
class Alert:
    """告警對象"""
    level: AlertLevel
    title: str
    message: str
    source: str = ''           # 來源模塊
    timestamp: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent_channels: List[str] = field(default_factory=list)


class AlertManager:
    """
    告警管理器
    
    集中管理所有告警的發送、去重和升級。
    
    配置示例:
        am = AlertManager()
        am.add_channel(AlertChannel.TELEGRAM, telegram_bot_token='xxx', chat_id='yyy')
        am.send_alert(AlertLevel.CRITICAL, '交易異常', '連續5次訂單被拒絕')
    """
    
    def __init__(self, cooldown_seconds: int = 300):
        """
        參數:
            cooldown_seconds: 相同告警冷卻時間（秒）
        """
        self.channels: Dict[AlertChannel, Dict] = {}
        self._alert_history: List[Alert] = []
        self._sent_alerts: Dict[str, float] = {}  # {alert_key: last_sent_time}
        self.cooldown = cooldown_seconds
        
        # 默認添加控制臺通道
        self.add_console_channel()
    
    def add_console_channel(self) -> None:
        """添加控制臺告警通道"""
        self.channels[AlertChannel.CONSOLE] = {
            'enabled': True,
            'min_level': AlertLevel.INFO,
        }
    
    def add_telegram_channel(self, bot_token: str, chat_id: str,
                             min_level: AlertLevel = AlertLevel.WARNING) -> None:
        """
        添加Telegram告警通道
        
        參數:
            bot_token: Telegram Bot Token
            chat_id: 目標Chat ID
            min_level: 該通道的最低告警級別
        """
        self.channels[AlertChannel.TELEGRAM] = {
            'enabled': True,
            'min_level': min_level,
            'bot_token': bot_token,
            'chat_id': chat_id,
        }
        logger.info("Telegram告警通道已添加")
    
    def add_email_channel(self, smtp_host: str, smtp_port: int,
                          sender: str, password: str,
                          recipients: List[str],
                          min_level: AlertLevel = AlertLevel.CRITICAL) -> None:
        """
        添加Email告警通道
        
        參數:
            smtp_host: SMTP伺服器地址
            smtp_port: SMTP埠
            sender: 發件人郵箱
            password: 發件人密碼/授權碼
            recipients: 收件人列表
            min_level: 最低告警級別
        """
        self.channels[AlertChannel.EMAIL] = {
            'enabled': True,
            'min_level': min_level,
            'smtp_host': smtp_host,
            'smtp_port': smtp_port,
            'sender': sender,
            'password': password,
            'recipients': recipients,
        }
        logger.info("Email告警通道已添加")
    
    def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str = '',
        metadata: Optional[Dict] = None,
        force: bool = False
    ) -> bool:
        """
        發送告警
        
        參數:
            level: 告警級別
            title: 告警標題
            message: 告警詳細信息
            source: 來源模塊
            metadata: 附加元數據
            force: 是否無視冷卻期強制發送
        
        返回:
            是否成功發送到至少一個通道
        """
        # 檢查冷卻期
        alert_key = f"{level.value}:{title}:{source}"
        now = datetime.now()
        
        if not force and alert_key in self._sent_alerts:
            if (now - datetime.fromtimestamp(self._sent_alerts[alert_key])).seconds < self.cooldown:
                logger.debug(f"告警冷卻中: {title}")
                return False
        
        # 創建告警對象
        alert = Alert(
            level=level,
            title=title,
            message=message,
            source=source,
            timestamp=now.isoformat(),
            metadata=metadata or {}
        )
        
        sent_any = False
        
        # 發送到各通道
        for channel, config in self.channels.items():
            if not config.get('enabled', False):
                continue
            
            min_level = config.get('min_level', AlertLevel.WARNING)
            if level.value < min_level.value:
                continue  # 低於通道最低級別要求
            
            try:
                if channel == AlertChannel.CONSOLE:
                    self._send_console(alert)
                elif channel == AlertChannel.TELEGRAM:
                    self._send_telegram(alert, config)
                elif channel == AlertChannel.EMAIL:
                    self._send_email(alert, config)
                
                alert.sent_channels.append(channel.value)
                sent_any = True
                
            except Exception as e:
                logger.error(f"告警通道 {channel.value} 發送失敗: {e}")
        
        self._alert_history.append(alert)
        self._sent_alerts[alert_key] = now.timestamp()
        
        return sent_any
    
    def _send_console(self, alert: Alert) -> None:
        """發送到控制臺"""
        prefixes = {
            AlertLevel.INFO: '📢',
            AlertLevel.WARNING: '⚠️',
            AlertLevel.CRITICAL: '🚨',
            AlertLevel.EMERGENCY: '🔥',
        }
        prefix = prefixes.get(alert.level, '')
        
        log_msg = f"\n{'='*60}\n{prefix} [{alert.level.name}] {alert.title}\n{'='*60}\n"
        log_msg += f"時間: {alert.timestamp}\n"
        log_msg += f"來源: {alert.source}\n"
        log_msg += f"詳情: {alert.message}\n"
        
        if alert.metadata:
            log_msg += f"元數據: {json.dumps(alert.metadata, indent=2)}\n"
        
        log_msg += f"{'='*60}\n"
        
        if alert.level.value >= AlertLevel.CRITICAL.value:
            logger.critical(log_msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def _send_telegram(self, alert: Alert, config: Dict) -> None:
        """
        通過Telegram Bot發送告警
        
        使用Telegram Bot API的sendMessage接口。
        """
        import requests
        
        bot_token = config['bot_token']
        chat_id = config['chat_id']
        
        text = f"*[{alert.level.name}] {alert.title}*\n\n"
        text += f"📅 `{alert.timestamp}`\n"
        text += f"📡 來源: `{alert.source}`\n\n"
        text += f"{alert.message}\n"
        
        if alert.metadata:
            text += f"\n```json\n{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}\n```"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        try:
            resp = requests.post(url, json={
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'Markdown',
            }, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Telegram發送失敗: {e}")
    
    def _send_email(self, alert: Alert, config: Dict) -> None:
        """
        通過Email發送告警
        """
        msg = MIMEMultipart()
        msg['From'] = config['sender']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = f"[量化交易-{alert.level.name}] {alert.title}"
        
        body = f"""
        <html>
        <body>
        <h2>[{alert.level.name}] {alert.title}</h2>
        <p><b>時間:</b> {alert.timestamp}</p>
        <p><b>來源:</b> {alert.source}</p>
        <hr/>
        <p>{alert.message}</p>
        <pre>{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}</pre>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        try:
            with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['sender'], config['password'])
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Email發送失敗: {e}")
    
    def get_recent_alerts(self, n: int = 20) -> List[Alert]:
        """獲取最近的告警記錄"""
        return self._alert_history[-n:]
    
    def clear_history(self) -> None:
        """清空告警歷史"""
        self._alert_history.clear()


# 全局告警管理器實例
alert_manager = AlertManager()
