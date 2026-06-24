#!/usr/bin/env python3
"""
美股量化交易系統 - 主入口

═══════════════════════════════════════════════════════════════
  US Stock Quantitative Trading System
  基於 Transformer 深度學習的智能量化交易系統
═══════════════════════════════════════════════════════════════

系統架構概覽：

  ┌──────────────────────────────────────────────────────┐
  │                    main.py (入口)                     │
  ├──────────────────────────────────────────────────────┤
  │  config/     │  全局配置（系統/數據源/模型/交易）      │
  │  utils/      │  工具模塊（常量/異常/工具函數/日誌）   │
  ├──────────────────────────────────────────────────────┤
  │  data_pipeline/  │  數據管道（採集/清洗/指標/存儲）   │
  │  crawler/        │  爬蟲模塊（S&P 500歷史數據）       │
  │  ml_model/       │  Transformer ML模型（訓練/調參）   │
  ├──────────────────────────────────────────────────────┤
  │  backtesting/    │  回測引擎（撮合/績效/券商模擬）     │
  │  execution/      │  OMS訂單執行（路由/算法單）         │
  │  risk/           │  風險管理（事前/事中/事後）         │
  ├──────────────────────────────────────────────────────┤
  │  monitoring/     │  監控告警（系統/交易/通知）         │
  │  compliance/     │  合規審查（洗售/做空/PDT）          │
  └──────────────────────────────────────────────────────┘

使用方式:
    python main.py crawl          # 爬取S&P 500歷史數據
    python main.py train          # 訓練Transformer模型
    python main.py tune           # 超參數自動調優
    python main.py backtest       # 運行回測
    python main.py live           # 啟動在線模擬交易（真實行情）
  python main.py web            # 啟動Web儀錶盤 (localhost:5000)
  python main.py demo           # 啟動離線演示模式

依賴安裝:
    pip install -r requirements.txt
"""

import sys
import argparse
import logging
from pathlib import Path

# 確保項目根目錄在 Python 路徑中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    system_config, data_source_config, model_config, trading_config
)
from config.logging_config import setup_logging


def setup_environment():
    """初始化運行環境"""
    # 配置日誌系統
    setup_logging(system_config)
    logger = logging.getLogger(__name__)
    logger.info("美股量化交易系統啟動")
    logger.info(f"運行模式: {system_config.mode}")
    logger.info(f"日誌級別: {system_config.log_level}")
    return logger


def cmd_crawl(args):
    """
    爬取S&P 500歷史數據（2010-2025）
    
    從Yahoo Finance批量下載S&P 500成分股的日線OHLCV數據，
    自動清洗、計算技術指標，並保存到本地。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任務: 爬取 S&P 500 歷史數據")
    logger.info("=" * 60)
    
    from crawler import SP500Crawler
    
    start = getattr(args, 'start', '2010-01-01')
    end = getattr(args, 'end', '2025-12-31')
    workers = getattr(args, 'workers', 3)
    refresh = getattr(args, 'refresh', False)
    
    crawler = SP500Crawler(data_source_config)
    
    if refresh:
        logger.info("強制刷新S&P 500成分股列表...")
    
    summary = crawler.crawl_sp500_historical(
        start_date=start,
        end_date=end,
        max_workers=workers
    )
    
    logger.info(f"爬取完成! 詳情請查看上方報告")
    return summary


def cmd_train(args):
    """
    訓練Transformer模型
    
    從本地加載已爬取的數據，使用Transformer模型進行訓練。
    訓練完成後自動評估精度並保存模型。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任務: 訓練 Transformer 模型")
    logger.info("=" * 60)
    
    from ml_model import ModelTrainer, prepare_data
    
    # 準備數據
    logger.info("正在加載訓練數據...")
    tickers = getattr(args, 'tickers', None)
    if tickers:
        tickers = tickers.split(',')
    
    train_loader, val_loader, test_loader, scaler = prepare_data(
        tickers=tickers,
        config=model_config
    )
    
    # 創建訓練器
    trainer = ModelTrainer(model_config)
    
    # 訓練
    epochs = getattr(args, 'epochs', model_config.epochs)
    metrics = trainer.train(
        train_loader, val_loader,
        epochs=epochs
    )
    
    # 評估
    logger.info("\n正在評估模型...")
    result = trainer.evaluate(test_loader)
    
    # 保存模型
    if not getattr(args, 'no_save', False):
        model_path = trainer.save_model("transformer_stock_latest")
        logger.info(f"模型已保存: {model_path}")
    
    return result


def cmd_tune(args):
    """
    超參數自動調優
    
    當模型精度不達標時（方向準確率<55%或RMSE>閾值），
    自動搜索最優超參數組合。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任務: 超參數自動調優")
    logger.info("=" * 60)
    
    from ml_model import ModelTrainer, HyperparameterTuner, prepare_data
    
    # 準備數據
    logger.info("正在加載訓練數據...")
    train_loader, val_loader, test_loader, scaler = prepare_data()
    
    # 創建調優器
    tuner = HyperparameterTuner(model_config)
    
    # 執行自動調優
    optimized_config = tuner.auto_tune(train_loader, val_loader, test_loader)
    
    # 使用最優配置重新訓練
    logger.info("\n使用最優配置重新訓練...")
    trainer = ModelTrainer(optimized_config)
    trainer.train(train_loader, val_loader)
    
    # 最終評估
    final_result = trainer.evaluate(test_loader)
    
    # 保存最優模型
    trainer.save_model("transformer_stock_optimized")
    
    logger.info(f"調優完成! 最優配置: lr={optimized_config.learning_rate:.2e}")
    
    return optimized_config


def cmd_backtest(args):
    """
    運行回測
    
    使用訓練好的模型生成交易信號，在歷史數據上進行回測。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任務: 運行回測")
    logger.info("=" * 60)
    
    from backtesting import BacktestEngine
    from data_pipeline.storage import ParquetStorage
    
    # 加載已處理的數據
    storage = ParquetStorage()
    available = storage.list_keys()
    
    ticker = getattr(args, 'ticker', 'AAPL')
    feature_key = f"{ticker}_features"
    
    if feature_key not in available:
        logger.error(f"未找到 {ticker} 的特徵數據，請先運行 crawl 命令")
        logger.info(f"可用數據: {list(set(k.replace('_features', '') for k in available if '_features' in k))[:20]}")
        return None
    
    # 加載數據
    df = storage.load(feature_key)
    logger.info(f"已加載 {ticker}: {len(df)} 行")
    
    # 創建回測引擎
    engine = BacktestEngine(trading_config)
    engine.load_data(ticker, df)
    
    # 設置簡單的ML策略
    def ml_strategy(engine, tickers, current_date, current_rows, account):
        """基於ML預測的簡單策略"""
        orders = []
        
        for ticker in tickers:
            row = current_rows.get(ticker)
            if row is None:
                continue
            
            # 簡化示例：始終買入（實際應使用模型預測）
            if ticker not in account.positions:
                qty = int(account.cash * 0.1 / row['close'])
                if qty > 0:
                    orders.append(engine._create_order(
                        ticker, 'BUY', qty,
                        reason=f"ML策略信號"
                    ))
        
        return orders
    
    engine.set_strategy(ml_strategy)
    
    # 運行回測
    start = getattr(args, 'start', None)
    end = getattr(args, 'end', None)
    
    result = engine.run(start_date=start, end_date=end)
    
    # 列印結果
    logger.info(f"\n回測完成!")
    logger.info(f"  總收益:   {result.total_return:.2%}")
    logger.info(f"  年化收益: {result.annual_return:.2%}")
    logger.info(f"  最大回撤: {result.max_drawdown:.2%}")
    logger.info(f"  夏普比率: {result.sharpe_ratio:.3f}")
    logger.info(f"  交易次數: {result.total_trades}")
    
    return result


def cmd_live(args):
    """
    啟動在線模擬交易系統
    
    同步真實時間，每分鐘刷新行情和預測，
    展示完整的持倉、盈虧、模型準確率和納指對比。
    初始資金：$100,000（模擬資金，使用真實行情數據）
    """
    logger = logging.getLogger(__name__)
    
    from live_trading import launch_dashboard
    
    tickers = getattr(args, 'tickers', None)
    if tickers:
        tickers = [t.strip().upper() for t in tickers.split(',')]
    
    capital = getattr(args, 'capital', 100000.0)
    
    logger.info("=" * 60)
    logger.info("  在線模擬交易系統啟動")
    logger.info(f"  初始資金: ${capital:,.2f}")
    logger.info("=" * 60)
    
    print(f"\n{'='*60}")
    print(f"  美股量化交易系統 - 在線模擬交易")
    print(f"  初始資金: ${capital:,.2f}")
    print(f"  追蹤股票: {tickers if tickers else 'AAPL,GOOGL,MSFT,AMZN,NVDA,META,TSLA,NFLX'}") 
    print(f"{'='*60}\n")
    
    launch_dashboard(
        tickers=tickers,
        initial_capital=capital,
        mode='live'
    )




def cmd_watchdog(args):
    """
    啟動守護進程（Watchdog）
    
    自動監控 Web 伺服器，崩潰後自動重啟。
    """
    from live_trading.watchdog import Watchdog
    wd = Watchdog()
    wd.run()


def cmd_web(args):
    """
    啟動 Web 儀錶盤
    
    在 localhost:5000 啟動網頁版交易面板，
    瀏覽器訪問即可實時觀察交易狀態。
    """
    logger = logging.getLogger(__name__)
    
    from live_trading.web_server import start_server
    
    port = getattr(args, 'port', 5000)
    
    print(f"\n{'='*60}")
    print(f"  美股量化交易系統 - Web儀錶盤")
    print(f"  打開瀏覽器訪問: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    start_server(port=port, debug=False)


def cmd_demo(args):
    """
    離線演示模式
    
    使用模擬數據顯示完整功能，無需網絡連接。
    適合測試和查看系統界面。
    """
    logger = logging.getLogger(__name__)
    
    from live_trading import launch_dashboard
    
    tickers = getattr(args, 'tickers', None)
    if tickers:
        tickers = [t.strip().upper() for t in tickers.split(',')]
    
    logger.info("啟動離線演示模式...")
    
    launch_dashboard(
        tickers=tickers,
        initial_capital=100000.0,
        mode='demo'
    )


def cmd_monitor(args):
    """啟動監控面板"""
    logger = logging.getLogger(__name__)
    logger.info("啟動監控面板...")
    
    from monitoring import SystemMonitor, AlertManager, AlertLevel
    
    monitor = SystemMonitor()
    alert_mgr = AlertManager()
    
    # 添加告警回調
    monitor.add_callback(lambda check: alert_mgr.send_alert(
        AlertLevel.WARNING,
        f"系統告警: {check.component}",
        check.message,
        source='monitoring'
    ))
    
    monitor.start()
    logger.info("監控已啟動，按 Ctrl+C 停止")
    
    try:
        while True:
            import time
            time.sleep(5)
            status = monitor.get_status_summary()
            logger.info(f"系統狀態: {status['overall_status']}")
    except KeyboardInterrupt:
        monitor.stop()
        logger.info("監控已停止")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='美股量化交易系統 - 基於Transformer的智能交易系統',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py crawl              # 爬取S&P 500歷史數據
  python main.py train              # 訓練Transformer模型
  python main.py tune               # 超參數調優
  python main.py backtest -t AAPL   # 在AAPL上回測
  python main.py monitor            # 啟動監控面板
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # crawl 子命令
    crawl_parser = subparsers.add_parser('crawl', help='爬取S&P 500歷史數據')
    crawl_parser.add_argument('--start', default='2010-01-01', help='起始日期')
    crawl_parser.add_argument('--end', default='2025-12-31', help='結束日期')
    crawl_parser.add_argument('--workers', type=int, default=3, help='並發數')
    crawl_parser.add_argument('--refresh', action='store_true', help='強制刷新成分股列表')
    
    # train 子命令
    train_parser = subparsers.add_parser('train', help='訓練Transformer模型')
    train_parser.add_argument('--tickers', help='股票代碼列表（逗號分隔）')
    train_parser.add_argument('--epochs', type=int, help='訓練輪數')
    train_parser.add_argument('--no-save', action='store_true', help='不保存模型')
    
    # tune 子命令
    subparsers.add_parser('tune', help='超參數自動調優')
    
    # backtest 子命令
    backtest_parser = subparsers.add_parser('backtest', help='運行回測')
    backtest_parser.add_argument('-t', '--ticker', default='AAPL', help='股票代碼')
    backtest_parser.add_argument('--start', help='回測起始日期')
    backtest_parser.add_argument('--end', help='回測結束日期')
    
    # live 子命令
    live_parser = subparsers.add_parser('live', help='啟動在線模擬交易（真實行情）')
    live_parser.add_argument('--tickers', help='追蹤股票（逗號分隔）')
    live_parser.add_argument('--capital', type=float, default=100000.0, help='初始資金')
    
    # monitor 子命令
    subparsers.add_parser('monitor', help='啟動監控面板')
    # demo 子命令
    # web 子命令
    
    # watchdog 子命令
    watchdog_parser = subparsers.add_parser('watchdog', help='啟動守護進程（自動監控+重啟）')

    web_parser = subparsers.add_parser("web", help="啟動Web儀錶盤")
    web_parser.add_argument("--port", type=int, default=8080, help="Web服務埠")

    # demo 子命令
    demo_parser = subparsers.add_parser('demo', help='離線演示模式')
    demo_parser.add_argument('--tickers', help='追蹤股票（逗號分隔）')

    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 初始化環境
    setup_environment()
    
    # 分發命令
    commands = {
        'crawl': cmd_crawl,
        'train': cmd_train,
        'tune': cmd_tune,
        'backtest': cmd_backtest,
        'live': cmd_live,
        'watchdog': cmd_watchdog,
        'web': cmd_web,
        'demo': cmd_demo,
        'monitor': cmd_monitor,
    }
    
    func = commands.get(args.command)
    if func:
        func(args)
    else:
        logger = logging.getLogger(__name__)
        logger.error(f"未知命令: {args.command}")


if __name__ == '__main__':
    main()
