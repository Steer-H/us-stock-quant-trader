"""
美股量化交易系统 - 订单与执行管理系统 (OMS)

负责把策略信号以最优方式送进市场：

核心组件：
1. OrderManager: 订单全生命周期管理
2. ExecutionRouter: 智能路由（Reg NMS最佳执行）
3. SmartOrderAlgo: 算法单（TWAP/VWAP/Iceberg）

订单生命周期：
策略信号 → 风控检查 → 订单创建 → 路由选择 → 提交交易所 → 
成交回报 → 状态更新 → 持仓更新 → 绩效记录

设计原则：
- 状态机模型：订单在各状态间严格迁移
- 幂等性：重复的订单操作不会产生副作用
- 审计完整：每步操作记录完整日志
"""

import logging
import uuid
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

import numpy as np

from config.settings import TradingConfig
from utils.constants import MarketHours
from utils.exceptions import (
    OrderRejectedError, OrderExecutionError, BrokerConnectionError
)

logger = logging.getLogger(__name__)


# ============================================================================
# 订单状态机
# ============================================================================
class OrderState(Enum):
    """订单生命周期状态"""
    CREATED = 'CREATED'           # 已创建，待风控审核
    RISK_APPROVED = 'APPROVED'    # 风控通过
    RISK_REJECTED = 'REJECTED'    # 风控拒绝
    PENDING_SUBMIT = 'PENDING'    # 待提交交易所
    SUBMITTED = 'SUBMITTED'       # 已提交
    PARTIALLY_FILLED = 'PARTIAL'  # 部分成交
    FILLED = 'FILLED'             # 全部成交
    CANCELLED = 'CANCELLED'       # 已取消
    EXPIRED = 'EXPIRED'           # 已过期
    ERROR = 'ERROR'               # 异常状态


# 合法的状态转换
VALID_TRANSITIONS = {
    OrderState.CREATED:          {OrderState.RISK_APPROVED, OrderState.RISK_REJECTED},
    OrderState.RISK_APPROVED:    {OrderState.PENDING_SUBMIT},
    OrderState.PENDING_SUBMIT:   {OrderState.SUBMITTED, OrderState.CANCELLED},
    OrderState.SUBMITTED:        {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED},
    OrderState.PARTIALLY_FILLED: {OrderState.FILLED, OrderState.CANCELLED},
    OrderState.RISK_REJECTED:    set(),  # 终态
    OrderState.FILLED:           set(),  # 终态
    OrderState.CANCELLED:        set(),  # 终态
    OrderState.EXPIRED:          set(),  # 终态
    OrderState.ERROR:            set(),  # 终态
}


@dataclass
class ManagedOrder:
    """
    托管订单对象
    
    包含完整的订单信息和状态追踪。
    """
    order_id: str                          # 唯一订单ID
    client_order_id: str                   # 客户订单ID（用于幂等）
    ticker: str
    side: str                              # BUY/SELL/SELL_SHORT
    order_type: str                        # MKT/LMT/STP
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'
    
    # 状态追踪
    state: OrderState = OrderState.CREATED
    state_history: List[tuple] = field(default_factory=list)
    
    # 成交信息
    filled_quantity: int = 0
    filled_avg_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    
    # 时间戳
    created_at: str = ''
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    
    # 元数据
    strategy_name: str = ''
    signal_reason: str = ''
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_terminal(self) -> bool:
        """是否为终态"""
        return self.state in {
            OrderState.FILLED, OrderState.CANCELLED,
            OrderState.EXPIRED, OrderState.ERROR,
            OrderState.RISK_REJECTED
        }
    
    @property
    def remaining_quantity(self) -> int:
        """剩余未成交数量"""
        return self.quantity - self.filled_quantity
    
    def transition_to(self, new_state: OrderState) -> bool:
        """
        状态转换
        
        参数:
            new_state: 目标状态
        
        返回:
            是否成功转换
        """
        if new_state not in VALID_TRANSITIONS.get(self.state, set()):
            logger.error(
                f"非法状态转换: {self.order_id} {self.state.value} → {new_state.value}"
            )
            return False
        
        self.state_history.append((self.state, new_state, datetime.now().isoformat()))
        self.state = new_state
        return True


# ============================================================================
# 订单管理器
# ============================================================================
class OrderManager:
    """
    订单全生命周期管理器
    
    职责：
    - 订单创建与ID生成
    - 状态追踪与转换
    - 订单查询与过滤
    - 成交汇总
    - 与券商API的交互封装
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.orders: Dict[str, ManagedOrder] = {}          # 所有订单
        self.active_orders: Dict[str, ManagedOrder] = {}   # 活跃订单（非终态）
        self.order_history: List[ManagedOrder] = []        # 历史订单（终态）
        self._id_counter: int = 0
        
        # 与券商API的接口（实际使用时替换为真实连接）
        self.broker_api: Optional[Any] = None
    
    def create_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = 'LMT',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = 'DAY',
        strategy_name: str = '',
        signal_reason: str = ''
    ) -> ManagedOrder:
        """
        创建新订单
        
        生成唯一订单ID和客户订单ID（用于幂等性）。
        
        参数:
            ticker: 股票代码
            side: BUY/SELL/SELL_SHORT
            quantity: 股数
            order_type: MKT/LMT/STP
            limit_price: 限价
            stop_price: 止损价
            time_in_force: DAY/GTC/IOC
            strategy_name: 策略名称
            signal_reason: 信号原因
        
        返回:
            ManagedOrder对象
        """
        self._id_counter += 1
        
        order_id = f"ORD{datetime.now().strftime('%Y%m%d')}{self._id_counter:06d}"
        client_id = str(uuid.uuid4())[:8]  # 短UUID用于幂等
        
        order = ManagedOrder(
            order_id=order_id,
            client_order_id=client_id,
            ticker=ticker.upper(),
            side=side.upper(),
            order_type=order_type.upper(),
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            created_at=datetime.now().isoformat(),
            strategy_name=strategy_name,
            signal_reason=signal_reason
        )
        
        self.orders[order_id] = order
        self.active_orders[order_id] = order
        
        logger.info(
            f"订单创建: {order_id} {side} {quantity} {ticker} "
            f"@{limit_price if limit_price else 'MKT'} [{strategy_name}]"
        )
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消订单
        
        只能取消尚未全部成交的订单。
        
        参数:
            order_id: 订单ID
        
        返回:
            是否成功取消
        """
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"订单不存在: {order_id}")
            return False
        
        if order.is_terminal:
            logger.warning(f"订单已是终态: {order_id} ({order.state.value})")
            return False
        
        if order.transition_to(OrderState.CANCELLED):
            order.cancelled_at = datetime.now().isoformat()
            self.active_orders.pop(order_id, None)
            self.order_history.append(order)
            logger.info(f"订单已取消: {order_id}")
            return True
        
        return False
    
    def on_fill(self, order_id: str, quantity: int, price: float) -> bool:
        """
        处理成交回报
        
        参数:
            order_id: 订单ID
            quantity: 成交数量
            price: 成交价格
        
        返回:
            是否成功处理
        """
        order = self.orders.get(order_id)
        if not order:
            return False
        
        # 更新成交信息
        order.filled_quantity += quantity
        
        # 更新平均成交价
        total_value = order.filled_avg_price * (order.filled_quantity - quantity) + quantity * price
        order.filled_avg_price = total_value / order.filled_quantity if order.filled_quantity > 0 else 0
        
        order.fills.append({
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat()
        })
        
        # 判断成交状态
        if order.filled_quantity >= order.quantity:
            order.transition_to(OrderState.FILLED)
            order.filled_at = datetime.now().isoformat()
            self.active_orders.pop(order_id, None)
            self.order_history.append(order)
        else:
            order.transition_to(OrderState.PARTIALLY_FILLED)
        
        logger.info(
            f"成交: {order_id} {quantity}股 @ ${price:.2f}, "
            f"累计 {order.filled_quantity}/{order.quantity}"
        )
        
        return True
    
    def on_reject(self, order_id: str, reason: str) -> None:
        """
        处理订单拒绝
        
        参数:
            order_id: 订单ID
            reason: 拒绝原因
        """
        order = self.orders.get(order_id)
        if order and not order.is_terminal:
            if order.transition_to(OrderState.RISK_REJECTED):
                self.active_orders.pop(order_id, None)
                self.order_history.append(order)
                logger.warning(f"订单被拒绝: {order_id} - {reason}")
            else:
                logger.warning(f"订单拒绝失败(状态转换无效): {order_id}, 当前状态: {order.state}")
        elif order:
            logger.warning(f"订单拒绝跳过(已是终态): {order_id}, 状态: {order.state}")
    
    def get_orders_by_ticker(self, ticker: str) -> List[ManagedOrder]:
        """按股票代码查询订单"""
        return [o for o in self.orders.values() if o.ticker == ticker.upper()]
    
    def get_active_orders(self) -> List[ManagedOrder]:
        """获取所有活跃订单"""
        return list(self.active_orders.values())
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """获取当前持仓汇总"""
        # 从成交记录汇总当前持仓（简化实现）
        positions = defaultdict(lambda: {'quantity': 0, 'total_cost': 0.0})
        
        for order in self.order_history:
            if order.state == OrderState.FILLED:
                ticker = order.ticker
                if order.side == 'BUY':
                    positions[ticker]['quantity'] += order.filled_quantity
                    positions[ticker]['total_cost'] += order.filled_quantity * order.filled_avg_price
                elif order.side == 'SELL':
                    positions[ticker]['quantity'] -= order.filled_quantity
                    positions[ticker]['total_cost'] -= order.filled_quantity * order.filled_avg_price
        
        # 过滤零持仓
        return {k: dict(v) for k, v in positions.items() if v['quantity'] != 0}


# ============================================================================
# 智能执行路由
# ============================================================================
class ExecutionRouter:
    """
    智能订单路由 (Smart Order Router)
    
    根据Reg NMS最佳执行义务，选择最优执行场所。
    
    路由考虑因素：
    - 报价质量（NBBO - National Best Bid and Offer）
    - 流动性（各交易所的订单簿深度）
    - 费用（maker/taker费率，回扣）
    - 速度（延迟）
    - 成交概率（历史成交率）
    """
    
    # 交易所代码和费率（简化）
    EXCHANGE_FEES = {
        'NYSE':   {'maker': -0.0010, 'taker': 0.0030},  # maker有回扣
        'NASDAQ': {'maker': -0.0015, 'taker': 0.0030},
        'ARCA':   {'maker': -0.0020, 'taker': 0.0030},
        'BATS':   {'maker': -0.0020, 'taker': 0.0029},
        'IEX':    {'maker': 0.0000,  'taker': 0.0009},  # 低延迟但无回扣
    }
    
    def __init__(self):
        self.exchange_priority: List[str] = ['ARCA', 'NASDAQ', 'NYSE', 'BATS', 'IEX']
    
    def route_order(
        self, ticker: str, side: str, quantity: int,
        urgency: str = 'normal'
    ) -> List[Dict]:
        """
        为订单选择最优路由
        
        根据流动性需求和紧急程度，将大订单拆分到多个交易所。
        
        参数:
            ticker: 股票代码
            side: BUY/SELL
            quantity: 总数量
            urgency: 'low'/'normal'/'high'
        
        返回:
            路由分配列表 [{exchange, quantity, reason}]
        """
        routes = []
        remaining = quantity
        
        if urgency == 'high':
            # 高紧急：优先速度，使用IEX（低延迟）
            routes.append({'exchange': 'IEX', 'quantity': remaining, 'reason': '低延迟'})
            return routes
        
        # 正常情况：按费用优势分配
        # 优先使用maker费率有回扣的交易所
        priority = ['ARCA', 'NASDAQ'] if side == 'BUY' else ['ARCA', 'BATS']
        
        for exchange in priority:
            if remaining <= 0:
                break
            # 每个交易所分配一部分（简化：平分）
            alloc = max(remaining // len(priority), 100)  # 至少100股
            alloc = min(alloc, remaining)
            routes.append({
                'exchange': exchange,
                'quantity': alloc,
                'reason': f'maker回扣: ${abs(self.EXCHANGE_FEES[exchange]["maker"]):.4f}/股'
            })
            remaining -= alloc
        
        if remaining > 0:
            routes.append({'exchange': 'NYSE', 'quantity': remaining, 'reason': '剩余'})
        
        return routes


# ============================================================================
# 算法单
# ============================================================================
class SmartOrderAlgo:
    """
    智能算法单
    
    支持高级执行算法：
    - TWAP: 时间加权平均价格
    - VWAP: 成交量加权平均价格
    - Iceberg: 冰山订单（只显示部分数量）
    """
    
    @staticmethod
    def twap_schedule(
        total_quantity: int,
        duration_minutes: int,
        interval_minutes: int = 5
    ) -> List[Dict]:
        """
        TWAP订单拆分
        
        将大订单拆分为多个等时间间隔的小订单，
        以接近时间加权平均价格。
        
        参数:
            total_quantity: 总股数
            duration_minutes: 执行总时长（分钟）
            interval_minutes: 下单间隔（分钟）
        
        返回:
            时间表 [{time_offset_min, quantity}]
        """
        n_intervals = max(1, duration_minutes // interval_minutes)
        base_qty = total_quantity // n_intervals
        remainder = total_quantity % n_intervals
        
        schedule = []
        for i in range(n_intervals):
            qty = base_qty + (1 if i < remainder else 0)
            if qty > 0:
                schedule.append({
                    'time_offset_min': i * interval_minutes,
                    'quantity': qty
                })
        
        logger.info(
            f"TWAP: {total_quantity}股, "
            f"{n_intervals}个间隔, 每{interval_minutes}分钟"
        )
        
        return schedule
    
    @staticmethod
    def vwap_schedule(
        total_quantity: int,
        volume_profile: Dict[int, float]
    ) -> Dict[int, int]:
        """
        VWAP订单拆分
        
        根据历史成交量分布，将订单按比例分配到不同时段，
        以接近成交量加权平均价格。
        
        参数:
            total_quantity: 总股数
            volume_profile: 成交量分布 {分钟: 比例}
                           如 {10: 0.05, 11: 0.08, ...}
        
        返回:
            分配方案 {分钟: 股数}
        """
        schedule = {}
        for minute, ratio in volume_profile.items():
            qty = int(total_quantity * ratio)
            if qty > 0:
                schedule[minute] = qty
        
        # 处理舍入误差
        allocated = sum(schedule.values())
        if allocated < total_quantity:
            # 将剩余分配到最后时段
            last_key = max(schedule.keys()) if schedule else 0
            schedule[last_key] = schedule.get(last_key, 0) + (total_quantity - allocated)
        
        return schedule
    
    @staticmethod
    def iceberg_params(
        total_quantity: int,
        display_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        冰山订单参数
        
        只显示订单总量的一小部分（如10%），
        隐藏真实意图，减少市场冲击。
        
        参数:
            total_quantity: 总股数
            display_ratio: 显示比例
        
        返回:
            冰山参数
        """
        display_qty = max(100, int(total_quantity * display_ratio))
        display_qty = min(display_qty, total_quantity)
        
        return {
            'total_quantity': total_quantity,
            'display_quantity': display_qty,
            'hidden_quantity': total_quantity - display_qty,
            'display_ratio': display_ratio,
        }
