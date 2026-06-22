#!/usr/bin/env python3
"""
交易系统守护进程 (Python实现)

确保持续运行：
- 启动 web_server.py 子进程
- 监控进程状态
- 崩溃自动重启
- 日志记录
"""

import sys, os, time, signal, subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / 'daemon.log'
RESTART_LOG = LOG_DIR / 'server_restart.log'

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def restart_log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(RESTART_LOG, 'a') as f:
        f.write(f"[{ts}] {msg}\n")

def main():
    log("===== 守护进程启动 =====")
    log(f"工作目录: {PROJECT_ROOT}")
    
    restart_count = 0
    window_start = time.time()
    
    # 忽略 SIGINT，不传递给子进程
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    while True:
        # 每小时重置计数器
        if time.time() - window_start > 3600:
            restart_count = 0
            window_start = time.time()
        
        log(f"启动 Web 服务器 (重启 #{restart_count})")
        restart_log(f"启动 Web 服务器 (重启 #{restart_count})")
        
        # 启动 web_server 作为子进程（新进程组，避免信号传播）
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / 'live_trading' / 'web_server.py')],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,
        )
        
        log(f"Web 服务器 PID: {proc.pid}")
        
        # 等待子进程退出
        try:
            exit_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            break
        
        restart_count += 1
        log(f"服务器退出 (code: {exit_code}), 5秒后重启...")
        restart_log(f"服务器退出 (code: {exit_code}), 重启 #{restart_count}, 5s后...")
        
        # 崩溃太频繁则停止
        if restart_count > 20:
            log("❌ 1小时内崩溃超过20次，停止自动恢复！")
            restart_log("❌ 1小时内崩溃超过20次，停止自动恢复！")
            break
        
        time.sleep(5)
    
    log("守护进程退出")

if __name__ == '__main__':
    main()
