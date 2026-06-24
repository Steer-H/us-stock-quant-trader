#!/usr/bin/env python3
"""
桌面通知工具

通過 macOS osascript 發送原生桌面通知。
支持異常告警、定期狀態摘要、以及靜音時段控制。

用法：
  python scripts/desktop_notify.py "標題" "消息內容"
  python scripts/desktop_notify.py --sound "交易異常" "連續3次訂單被拒絕"
  python scripts/desktop_notify.py --quiet-hours 22-8 "標題" "非緊急消息"
"""

import sys
import subprocess
import argparse
from datetime import datetime
from typing import Optional


def is_quiet_hours(quiet_range: Optional[str] = None) -> bool:
    """檢查當前是否在靜音時段"""
    if not quiet_range:
        return False
    try:
        start_h, end_h = map(int, quiet_range.split('-'))
        now_h = datetime.now().hour
        if start_h > end_h:
            # 跨天，如 22-8
            return now_h >= start_h or now_h < end_h
        else:
            return start_h <= now_h < end_h
    except Exception:
        return False


def send_notification(title: str, message: str, *,
                      sound: bool = True,
                      subtitle: str = '',
                      quiet_hours: Optional[str] = None) -> bool:
    """
    發送 macOS 桌面通知

    參數:
        title: 通知標題
        message: 通知正文
        sound: 是否播放聲音
        subtitle: 副標題
        quiet_hours: 靜音時段，如 "22-8"

    返回:
        是否成功發送
    """
    if is_quiet_hours(quiet_hours):
        return False

    # 清理文本中的特殊字符
    title = title.replace('"', "'").replace('\\', '')
    message = message.replace('"', "'").replace('\\', '')
    subtitle = subtitle.replace('"', "'").replace('\\', '')

    script_parts = [f'display notification "{message}" with title "{title}"']
    if subtitle:
        script_parts.append(f'subtitle "{subtitle}"')
    if sound:
        script_parts.append('sound name "default"')

    script = ' '.join(script_parts)

    try:
        subprocess.run(
            ['osascript', '-e', script],
            timeout=3, capture_output=True
        )
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='macOS 桌面通知工具')
    parser.add_argument('title', help='通知標題')
    parser.add_argument('message', help='通知內容')
    parser.add_argument('--no-sound', action='store_true', help='靜音通知')
    parser.add_argument('--subtitle', default='', help='通知副標題')
    parser.add_argument('--quiet-hours', default=None,
                        help='靜音時段，如 22-8（22點到次日8點不通知）')
    args = parser.parse_args()

    sent = send_notification(
        title=args.title,
        message=args.message,
        sound=not args.no_sound,
        subtitle=args.subtitle,
        quiet_hours=args.quiet_hours,
    )

    if sent:
        print(f"✅ 通知已發送: {args.title}")
    else:
        print("⏸️  通知已抑制（靜音時段或發送失敗）")


if __name__ == '__main__':
    main()
