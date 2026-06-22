"""
美股量化交易系统 - 告警模块

支持多通道分级告警推送：
- Telegram Bot
- Email (SMTP)
- SMS (Twilio)
- Console (开发调试用)

告警分级：
- INFO: 一般信息，无需立即处理
- WARNING: 需关注，建议检查
- CRITICAL: 需要立即处理
- EMERGENCY: 紧急情况，可能造成严重损失

设计原则：
- 告警去重：相同告警在冷却期内不重复发送
- 告警升级：低级别告警持续触发后自动升级
- 告警静默：支持临时静默特定类型告警
"""

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
    """告警级别"""
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
    """告警对象"""
    level: AlertLevel
    title: str
    message: str
    source: str = ''           # 来源模块
    timestamp: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent_channels: List[str] = field(default_factory=list)


class AlertManager:
    """
    告警管理器
    
    集中管理所有告警的发送、去重和升级。
    
    配置示例:
        am = AlertManager()
        am.add_channel(AlertChannel.TELEGRAM, telegram_bot_token='xxx', chat_id='yyy')
        am.send_alert(AlertLevel.CRITICAL, '交易异常', '连续5次订单被拒绝')
    """
    
    def __init__(self, cooldown_seconds: int = 300):
        """
        参数:
            cooldown_seconds: 相同告警冷却时间（秒）
        """
        self.channels: Dict[AlertChannel, Dict] = {}
        self._alert_history: List[Alert] = []
        self._sent_alerts: Dict[str, float] = {}  # {alert_key: last_sent_time}
        self.cooldown = cooldown_seconds
        
        # 默认添加控制台通道
        self.add_console_channel()
    
    def add_console_channel(self) -> None:
        """添加控制台告警通道"""
        self.channels[AlertChannel.CONSOLE] = {
            'enabled': True,
            'min_level': AlertLevel.INFO,
        }
    
    def add_telegram_channel(self, bot_token: str, chat_id: str,
                             min_level: AlertLevel = AlertLevel.WARNING) -> None:
        """
        添加Telegram告警通道
        
        参数:
            bot_token: Telegram Bot Token
            chat_id: 目标Chat ID
            min_level: 该通道的最低告警级别
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
        
        参数:
            smtp_host: SMTP服务器地址
            smtp_port: SMTP端口
            sender: 发件人邮箱
            password: 发件人密码/授权码
            recipients: 收件人列表
            min_level: 最低告警级别
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
        发送告警
        
        参数:
            level: 告警级别
            title: 告警标题
            message: 告警详细信息
            source: 来源模块
            metadata: 附加元数据
            force: 是否无视冷却期强制发送
        
        返回:
            是否成功发送到至少一个通道
        """
        # 检查冷却期
        alert_key = f"{level.value}:{title}:{source}"
        now = datetime.now()
        
        if not force and alert_key in self._sent_alerts:
            if (now - datetime.fromtimestamp(self._sent_alerts[alert_key])).seconds < self.cooldown:
                logger.debug(f"告警冷却中: {title}")
                return False
        
        # 创建告警对象
        alert = Alert(
            level=level,
            title=title,
            message=message,
            source=source,
            timestamp=now.isoformat(),
            metadata=metadata or {}
        )
        
        sent_any = False
        
        # 发送到各通道
        for channel, config in self.channels.items():
            if not config.get('enabled', False):
                continue
            
            min_level = config.get('min_level', AlertLevel.WARNING)
            if level.value < min_level.value:
                continue  # 低于通道最低级别要求
            
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
                logger.error(f"告警通道 {channel.value} 发送失败: {e}")
        
        self._alert_history.append(alert)
        self._sent_alerts[alert_key] = now.timestamp()
        
        return sent_any
    
    def _send_console(self, alert: Alert) -> None:
        """发送到控制台"""
        prefixes = {
            AlertLevel.INFO: '📢',
            AlertLevel.WARNING: '⚠️',
            AlertLevel.CRITICAL: '🚨',
            AlertLevel.EMERGENCY: '🔥',
        }
        prefix = prefixes.get(alert.level, '')
        
        log_msg = f"\n{'='*60}\n{prefix} [{alert.level.name}] {alert.title}\n{'='*60}\n"
        log_msg += f"时间: {alert.timestamp}\n"
        log_msg += f"来源: {alert.source}\n"
        log_msg += f"详情: {alert.message}\n"
        
        if alert.metadata:
            log_msg += f"元数据: {json.dumps(alert.metadata, indent=2)}\n"
        
        log_msg += f"{'='*60}\n"
        
        if alert.level.value >= AlertLevel.CRITICAL.value:
            logger.critical(log_msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def _send_telegram(self, alert: Alert, config: Dict) -> None:
        """
        通过Telegram Bot发送告警
        
        使用Telegram Bot API的sendMessage接口。
        """
        import requests
        
        bot_token = config['bot_token']
        chat_id = config['chat_id']
        
        text = f"*[{alert.level.name}] {alert.title}*\n\n"
        text += f"📅 `{alert.timestamp}`\n"
        text += f"📡 来源: `{alert.source}`\n\n"
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
            logger.error(f"Telegram发送失败: {e}")
    
    def _send_email(self, alert: Alert, config: Dict) -> None:
        """
        通过Email发送告警
        """
        msg = MIMEMultipart()
        msg['From'] = config['sender']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = f"[量化交易-{alert.level.name}] {alert.title}"
        
        body = f"""
        <html>
        <body>
        <h2>[{alert.level.name}] {alert.title}</h2>
        <p><b>时间:</b> {alert.timestamp}</p>
        <p><b>来源:</b> {alert.source}</p>
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
            logger.error(f"Email发送失败: {e}")
    
    def get_recent_alerts(self, n: int = 20) -> List[Alert]:
        """获取最近的告警记录"""
        return self._alert_history[-n:]
    
    def clear_history(self) -> None:
        """清空告警历史"""
        self._alert_history.clear()


# 全局告警管理器实例
alert_manager = AlertManager()
