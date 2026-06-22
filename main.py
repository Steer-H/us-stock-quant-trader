#!/usr/bin/env python3
"""
美股量化交易系统 - 主入口

═══════════════════════════════════════════════════════════════
  US Stock Quantitative Trading System
  基于 Transformer 深度学习的智能量化交易系统
═══════════════════════════════════════════════════════════════

系统架构概览：

  ┌──────────────────────────────────────────────────────┐
  │                    main.py (入口)                     │
  ├──────────────────────────────────────────────────────┤
  │  config/     │  全局配置（系统/数据源/模型/交易）      │
  │  utils/      │  工具模块（常量/异常/工具函数/日志）   │
  ├──────────────────────────────────────────────────────┤
  │  data_pipeline/  │  数据管道（采集/清洗/指标/存储）   │
  │  crawler/        │  爬虫模块（S&P 500历史数据）       │
  │  ml_model/       │  Transformer ML模型（训练/调参）   │
  ├──────────────────────────────────────────────────────┤
  │  backtesting/    │  回测引擎（撮合/绩效/券商模拟）     │
  │  execution/      │  OMS订单执行（路由/算法单）         │
  │  risk/           │  风险管理（事前/事中/事后）         │
  ├──────────────────────────────────────────────────────┤
  │  monitoring/     │  监控告警（系统/交易/通知）         │
  │  compliance/     │  合规审查（洗售/做空/PDT）          │
  └──────────────────────────────────────────────────────┘

使用方式:
    python main.py crawl          # 爬取S&P 500历史数据
    python main.py train          # 训练Transformer模型
    python main.py tune           # 超参数自动调优
    python main.py backtest       # 运行回测
    python main.py live           # 启动在线模拟交易（真实行情）
  python main.py web            # 启动Web仪表盘 (localhost:5000)
  python main.py demo           # 启动离线演示模式

依赖安装:
    pip install -r requirements.txt
"""

import sys
import argparse
import logging
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    system_config, data_source_config, model_config, trading_config
)
from config.logging_config import setup_logging


def setup_environment():
    """初始化运行环境"""
    # 配置日志系统
    setup_logging(system_config)
    logger = logging.getLogger(__name__)
    logger.info("美股量化交易系统启动")
    logger.info(f"运行模式: {system_config.mode}")
    logger.info(f"日志级别: {system_config.log_level}")
    return logger


def cmd_crawl(args):
    """
    爬取S&P 500历史数据（2010-2025）
    
    从Yahoo Finance批量下载S&P 500成分股的日线OHLCV数据，
    自动清洗、计算技术指标，并保存到本地。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任务: 爬取 S&P 500 历史数据")
    logger.info("=" * 60)
    
    from crawler import SP500Crawler
    
    start = getattr(args, 'start', '2010-01-01')
    end = getattr(args, 'end', '2025-12-31')
    workers = getattr(args, 'workers', 3)
    refresh = getattr(args, 'refresh', False)
    
    crawler = SP500Crawler(data_source_config)
    
    if refresh:
        logger.info("强制刷新S&P 500成分股列表...")
    
    summary = crawler.crawl_sp500_historical(
        start_date=start,
        end_date=end,
        max_workers=workers
    )
    
    logger.info(f"爬取完成! 详情请查看上方报告")
    return summary


def cmd_train(args):
    """
    训练Transformer模型
    
    从本地加载已爬取的数据，使用Transformer模型进行训练。
    训练完成后自动评估精度并保存模型。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任务: 训练 Transformer 模型")
    logger.info("=" * 60)
    
    from ml_model import ModelTrainer, prepare_data
    
    # 准备数据
    logger.info("正在加载训练数据...")
    tickers = getattr(args, 'tickers', None)
    if tickers:
        tickers = tickers.split(',')
    
    train_loader, val_loader, test_loader, scaler = prepare_data(
        tickers=tickers,
        config=model_config
    )
    
    # 创建训练器
    trainer = ModelTrainer(model_config)
    
    # 训练
    epochs = getattr(args, 'epochs', model_config.epochs)
    metrics = trainer.train(
        train_loader, val_loader,
        epochs=epochs
    )
    
    # 评估
    logger.info("\n正在评估模型...")
    result = trainer.evaluate(test_loader)
    
    # 保存模型
    if not getattr(args, 'no_save', False):
        model_path = trainer.save_model("transformer_stock_latest")
        logger.info(f"模型已保存: {model_path}")
    
    return result


def cmd_tune(args):
    """
    超参数自动调优
    
    当模型精度不达标时（方向准确率<55%或RMSE>阈值），
    自动搜索最优超参数组合。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任务: 超参数自动调优")
    logger.info("=" * 60)
    
    from ml_model import ModelTrainer, HyperparameterTuner, prepare_data
    
    # 准备数据
    logger.info("正在加载训练数据...")
    train_loader, val_loader, test_loader, scaler = prepare_data()
    
    # 创建调优器
    tuner = HyperparameterTuner(model_config)
    
    # 执行自动调优
    optimized_config = tuner.auto_tune(train_loader, val_loader, test_loader)
    
    # 使用最优配置重新训练
    logger.info("\n使用最优配置重新训练...")
    trainer = ModelTrainer(optimized_config)
    trainer.train(train_loader, val_loader)
    
    # 最终评估
    final_result = trainer.evaluate(test_loader)
    
    # 保存最优模型
    trainer.save_model("transformer_stock_optimized")
    
    logger.info(f"调优完成! 最优配置: lr={optimized_config.learning_rate:.2e}")
    
    return optimized_config


def cmd_backtest(args):
    """
    运行回测
    
    使用训练好的模型生成交易信号，在历史数据上进行回测。
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  任务: 运行回测")
    logger.info("=" * 60)
    
    from backtesting import BacktestEngine
    from data_pipeline.storage import ParquetStorage
    
    # 加载已处理的数据
    storage = ParquetStorage()
    available = storage.list_keys()
    
    ticker = getattr(args, 'ticker', 'AAPL')
    feature_key = f"{ticker}_features"
    
    if feature_key not in available:
        logger.error(f"未找到 {ticker} 的特征数据，请先运行 crawl 命令")
        logger.info(f"可用数据: {list(set(k.replace('_features', '') for k in available if '_features' in k))[:20]}")
        return None
    
    # 加载数据
    df = storage.load(feature_key)
    logger.info(f"已加载 {ticker}: {len(df)} 行")
    
    # 创建回测引擎
    engine = BacktestEngine(trading_config)
    engine.load_data(ticker, df)
    
    # 设置简单的ML策略
    def ml_strategy(engine, tickers, current_date, current_rows, account):
        """基于ML预测的简单策略"""
        orders = []
        
        for ticker in tickers:
            row = current_rows.get(ticker)
            if row is None:
                continue
            
            # 简化示例：始终买入（实际应使用模型预测）
            if ticker not in account.positions:
                qty = int(account.cash * 0.1 / row['close'])
                if qty > 0:
                    orders.append(engine._create_order(
                        ticker, 'BUY', qty,
                        reason=f"ML策略信号"
                    ))
        
        return orders
    
    engine.set_strategy(ml_strategy)
    
    # 运行回测
    start = getattr(args, 'start', None)
    end = getattr(args, 'end', None)
    
    result = engine.run(start_date=start, end_date=end)
    
    # 打印结果
    logger.info(f"\n回测完成!")
    logger.info(f"  总收益:   {result.total_return:.2%}")
    logger.info(f"  年化收益: {result.annual_return:.2%}")
    logger.info(f"  最大回撤: {result.max_drawdown:.2%}")
    logger.info(f"  夏普比率: {result.sharpe_ratio:.3f}")
    logger.info(f"  交易次数: {result.total_trades}")
    
    return result


def cmd_live(args):
    """
    启动在线模拟交易系统
    
    同步真实时间，每分钟刷新行情和预测，
    展示完整的持仓、盈亏、模型准确率和纳指对比。
    初始资金：$100,000（模拟资金，使用真实行情数据）
    """
    logger = logging.getLogger(__name__)
    
    from live_trading import launch_dashboard
    
    tickers = getattr(args, 'tickers', None)
    if tickers:
        tickers = [t.strip().upper() for t in tickers.split(',')]
    
    capital = getattr(args, 'capital', 100000.0)
    
    logger.info("=" * 60)
    logger.info("  在线模拟交易系统启动")
    logger.info(f"  初始资金: ${capital:,.2f}")
    logger.info("=" * 60)
    
    print(f"\n{'='*60}")
    print(f"  美股量化交易系统 - 在线模拟交易")
    print(f"  初始资金: ${capital:,.2f}")
    print(f"  追踪股票: {tickers if tickers else 'AAPL,GOOGL,MSFT,AMZN,NVDA,META,TSLA,NFLX'}") 
    print(f"{'='*60}\n")
    
    launch_dashboard(
        tickers=tickers,
        initial_capital=capital,
        mode='live'
    )




def cmd_watchdog(args):
    """
    启动守护进程（Watchdog）
    
    自动监控 Web 服务器，崩溃后自动重启。
    """
    from live_trading.watchdog import Watchdog
    wd = Watchdog()
    wd.run()


def cmd_web(args):
    """
    启动 Web 仪表盘
    
    在 localhost:5000 启动网页版交易面板，
    浏览器访问即可实时观察交易状态。
    """
    logger = logging.getLogger(__name__)
    
    from live_trading.web_server import start_server
    
    port = getattr(args, 'port', 5000)
    
    print(f"\n{'='*60}")
    print(f"  美股量化交易系统 - Web仪表盘")
    print(f"  打开浏览器访问: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    start_server(port=port, debug=False)


def cmd_demo(args):
    """
    离线演示模式
    
    使用模拟数据显示完整功能，无需网络连接。
    适合测试和查看系统界面。
    """
    logger = logging.getLogger(__name__)
    
    from live_trading import launch_dashboard
    
    tickers = getattr(args, 'tickers', None)
    if tickers:
        tickers = [t.strip().upper() for t in tickers.split(',')]
    
    logger.info("启动离线演示模式...")
    
    launch_dashboard(
        tickers=tickers,
        initial_capital=100000.0,
        mode='demo'
    )


def cmd_monitor(args):
    """启动监控面板"""
    logger = logging.getLogger(__name__)
    logger.info("启动监控面板...")
    
    from monitoring import SystemMonitor, AlertManager, AlertLevel
    
    monitor = SystemMonitor()
    alert_mgr = AlertManager()
    
    # 添加告警回调
    monitor.add_callback(lambda check: alert_mgr.send_alert(
        AlertLevel.WARNING,
        f"系统告警: {check.component}",
        check.message,
        source='monitoring'
    ))
    
    monitor.start()
    logger.info("监控已启动，按 Ctrl+C 停止")
    
    try:
        while True:
            import time
            time.sleep(5)
            status = monitor.get_status_summary()
            logger.info(f"系统状态: {status['overall_status']}")
    except KeyboardInterrupt:
        monitor.stop()
        logger.info("监控已停止")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='美股量化交易系统 - 基于Transformer的智能交易系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py crawl              # 爬取S&P 500历史数据
  python main.py train              # 训练Transformer模型
  python main.py tune               # 超参数调优
  python main.py backtest -t AAPL   # 在AAPL上回测
  python main.py monitor            # 启动监控面板
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # crawl 子命令
    crawl_parser = subparsers.add_parser('crawl', help='爬取S&P 500历史数据')
    crawl_parser.add_argument('--start', default='2010-01-01', help='起始日期')
    crawl_parser.add_argument('--end', default='2025-12-31', help='结束日期')
    crawl_parser.add_argument('--workers', type=int, default=3, help='并发数')
    crawl_parser.add_argument('--refresh', action='store_true', help='强制刷新成分股列表')
    
    # train 子命令
    train_parser = subparsers.add_parser('train', help='训练Transformer模型')
    train_parser.add_argument('--tickers', help='股票代码列表（逗号分隔）')
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    train_parser.add_argument('--no-save', action='store_true', help='不保存模型')
    
    # tune 子命令
    subparsers.add_parser('tune', help='超参数自动调优')
    
    # backtest 子命令
    backtest_parser = subparsers.add_parser('backtest', help='运行回测')
    backtest_parser.add_argument('-t', '--ticker', default='AAPL', help='股票代码')
    backtest_parser.add_argument('--start', help='回测起始日期')
    backtest_parser.add_argument('--end', help='回测结束日期')
    
    # live 子命令
    live_parser = subparsers.add_parser('live', help='启动在线模拟交易（真实行情）')
    live_parser.add_argument('--tickers', help='追踪股票（逗号分隔）')
    live_parser.add_argument('--capital', type=float, default=100000.0, help='初始资金')
    
    # monitor 子命令
    subparsers.add_parser('monitor', help='启动监控面板')
    # demo 子命令
    # web 子命令
    
    # watchdog 子命令
    watchdog_parser = subparsers.add_parser('watchdog', help='启动守护进程（自动监控+重启）')

    web_parser = subparsers.add_parser("web", help="启动Web仪表盘")
    web_parser.add_argument("--port", type=int, default=8080, help="Web服务端口")

    # demo 子命令
    demo_parser = subparsers.add_parser('demo', help='离线演示模式')
    demo_parser.add_argument('--tickers', help='追踪股票（逗号分隔）')

    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 初始化环境
    setup_environment()
    
    # 分发命令
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
