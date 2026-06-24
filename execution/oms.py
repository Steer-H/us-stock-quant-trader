"""Order management system."""

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
# 訂單狀態機
# ============================================================================
class OrderState(Enum):
    """訂單生命周期狀態"""
    CREATED = 'CREATED'           # 已創建，待風控審核
    RISK_APPROVED = 'APPROVED'    # 風控通過
    RISK_REJECTED = 'REJECTED'    # 風控拒絕
    PENDING_SUBMIT = 'PENDING'    # 待提交交易所
    SUBMITTED = 'SUBMITTED'       # 已提交
    PARTIALLY_FILLED = 'PARTIAL'  # 部分成交
    FILLED = 'FILLED'             # 全部成交
    CANCELLED = 'CANCELLED'       # 已取消
    EXPIRED = 'EXPIRED'           # 已過期
    ERROR = 'ERROR'               # 異常狀態


# 合法的狀態轉換
VALID_TRANSITIONS = {
    OrderState.CREATED:          {OrderState.RISK_APPROVED, OrderState.RISK_REJECTED},
    OrderState.RISK_APPROVED:    {OrderState.PENDING_SUBMIT},
    OrderState.PENDING_SUBMIT:   {OrderState.SUBMITTED, OrderState.CANCELLED},
    OrderState.SUBMITTED:        {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED},
    OrderState.PARTIALLY_FILLED: {OrderState.FILLED, OrderState.CANCELLED},
    OrderState.RISK_REJECTED:    set(),  # 終態
    OrderState.FILLED:           set(),  # 終態
    OrderState.CANCELLED:        set(),  # 終態
    OrderState.EXPIRED:          set(),  # 終態
    OrderState.ERROR:            set(),  # 終態
}


@dataclass
class ManagedOrder:
    """
    託管訂單對象
    
    包含完整的訂單信息和狀態追蹤。
    """
    order_id: str                          # 唯一訂單ID
    client_order_id: str                   # 客戶訂單ID（用於冪等）
    ticker: str
    side: str                              # BUY/SELL/SELL_SHORT
    order_type: str                        # MKT/LMT/STP
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'
    
    # 狀態追蹤
    state: OrderState = OrderState.CREATED
    state_history: List[tuple] = field(default_factory=list)
    
    # 成交信息
    filled_quantity: int = 0
    filled_avg_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    
    # 時間戳
    created_at: str = ''
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    
    # 元數據
    strategy_name: str = ''
    signal_reason: str = ''
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_terminal(self) -> bool:
        """是否為終態"""
        return self.state in {
            OrderState.FILLED, OrderState.CANCELLED,
            OrderState.EXPIRED, OrderState.ERROR,
            OrderState.RISK_REJECTED
        }
    
    @property
    def remaining_quantity(self) -> int:
        """剩餘未成交數量"""
        return self.quantity - self.filled_quantity
    
    def transition_to(self, new_state: OrderState) -> bool:
        """
        狀態轉換
        
        參數:
            new_state: 目標狀態
        
        返回:
            是否成功轉換
        """
        if new_state not in VALID_TRANSITIONS.get(self.state, set()):
            logger.error(
                f"非法狀態轉換: {self.order_id} {self.state.value} → {new_state.value}"
            )
            return False
        
        self.state_history.append((self.state, new_state, datetime.now().isoformat()))
        self.state = new_state
        return True


# ============================================================================
# 訂單管理器
# ============================================================================
class OrderManager:
    """
    訂單全生命周期管理器
    
    職責：
    - 訂單創建與ID生成
    - 狀態追蹤與轉換
    - 訂單查詢與過濾
    - 成交匯總
    - 與券商API的交互封裝
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.orders: Dict[str, ManagedOrder] = {}          # 所有訂單
        self.active_orders: Dict[str, ManagedOrder] = {}   # 活躍訂單（非終態）
        self.order_history: List[ManagedOrder] = []        # 歷史訂單（終態）
        self._id_counter: int = 0
        
        # 與券商API的接口（實際使用時替換為真實連接）
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
        創建新訂單
        
        生成唯一訂單ID和客戶訂單ID（用於冪等性）。
        
        參數:
            ticker: 股票代碼
            side: BUY/SELL/SELL_SHORT
            quantity: 股數
            order_type: MKT/LMT/STP
            limit_price: 限價
            stop_price: 止損價
            time_in_force: DAY/GTC/IOC
            strategy_name: 策略名稱
            signal_reason: 信號原因
        
        返回:
            ManagedOrder對象
        """
        self._id_counter += 1
        
        order_id = f"ORD{datetime.now().strftime('%Y%m%d')}{self._id_counter:06d}"
        client_id = str(uuid.uuid4())[:8]  # 短UUID用於冪等
        
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
            f"訂單創建: {order_id} {side} {quantity} {ticker} "
            f"@{limit_price if limit_price else 'MKT'} [{strategy_name}]"
        )
        
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        取消訂單
        
        只能取消尚未全部成交的訂單。
        
        參數:
            order_id: 訂單ID
        
        返回:
            是否成功取消
        """
        order = self.orders.get(order_id)
        if not order:
            logger.warning(f"訂單不存在: {order_id}")
            return False
        
        if order.is_terminal:
            logger.warning(f"訂單已是終態: {order_id} ({order.state.value})")
            return False
        
        if order.transition_to(OrderState.CANCELLED):
            order.cancelled_at = datetime.now().isoformat()
            self.active_orders.pop(order_id, None)
            self.order_history.append(order)
            logger.info(f"訂單已取消: {order_id}")
            return True
        
        return False
    
    def on_fill(self, order_id: str, quantity: int, price: float) -> bool:
        """
        處理成交回報
        
        參數:
            order_id: 訂單ID
            quantity: 成交數量
            price: 成交價格
        
        返回:
            是否成功處理
        """
        order = self.orders.get(order_id)
        if not order:
            return False
        
        # 更新成交信息
        order.filled_quantity += quantity
        
        # 更新平均成交價
        total_value = order.filled_avg_price * (order.filled_quantity - quantity) + quantity * price
        order.filled_avg_price = total_value / order.filled_quantity if order.filled_quantity > 0 else 0
        
        order.fills.append({
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().isoformat()
        })
        
        # 判斷成交狀態
        if order.filled_quantity >= order.quantity:
            order.transition_to(OrderState.FILLED)
            order.filled_at = datetime.now().isoformat()
            self.active_orders.pop(order_id, None)
            self.order_history.append(order)
        else:
            order.transition_to(OrderState.PARTIALLY_FILLED)
        
        logger.info(
            f"成交: {order_id} {quantity}股 @ ${price:.2f}, "
            f"累計 {order.filled_quantity}/{order.quantity}"
        )
        
        return True
    
    def on_reject(self, order_id: str, reason: str) -> None:
        """
        處理訂單拒絕
        
        參數:
            order_id: 訂單ID
            reason: 拒絕原因
        """
        order = self.orders.get(order_id)
        if order and not order.is_terminal:
            if order.transition_to(OrderState.RISK_REJECTED):
                self.active_orders.pop(order_id, None)
                self.order_history.append(order)
                logger.warning(f"訂單被拒絕: {order_id} - {reason}")
            else:
                logger.warning(f"訂單拒絕失敗(狀態轉換無效): {order_id}, 當前狀態: {order.state}")
        elif order:
            logger.warning(f"訂單拒絕跳過(已是終態): {order_id}, 狀態: {order.state}")
    
    def get_orders_by_ticker(self, ticker: str) -> List[ManagedOrder]:
        """按股票代碼查詢訂單"""
        return [o for o in self.orders.values() if o.ticker == ticker.upper()]
    
    def get_active_orders(self) -> List[ManagedOrder]:
        """獲取所有活躍訂單"""
        return list(self.active_orders.values())
    
    def get_position_summary(self) -> Dict[str, Dict]:
        """獲取當前持倉匯總"""
        # 從成交記錄匯總當前持倉（簡化實現）
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
        
        # 過濾零持倉
        return {k: dict(v) for k, v in positions.items() if v['quantity'] != 0}


# ============================================================================
# 智能執行路由
# ============================================================================
class ExecutionRouter:
    """
    智能訂單路由 (Smart Order Router)
    
    根據Reg NMS最佳執行義務，選擇最優執行場所。
    
    路由考慮因素：
    - 報價質量（NBBO - National Best Bid and Offer）
    - 流動性（各交易所的訂單簿深度）
    - 費用（maker/taker費率，回扣）
    - 速度（延遲）
    - 成交概率（歷史成交率）
    """
    
    # 交易所代碼和費率（簡化）
    EXCHANGE_FEES = {
        'NYSE':   {'maker': -0.0010, 'taker': 0.0030},  # maker有回扣
        'NASDAQ': {'maker': -0.0015, 'taker': 0.0030},
        'ARCA':   {'maker': -0.0020, 'taker': 0.0030},
        'BATS':   {'maker': -0.0020, 'taker': 0.0029},
        'IEX':    {'maker': 0.0000,  'taker': 0.0009},  # 低延遲但無回扣
    }
    
    def __init__(self):
        self.exchange_priority: List[str] = ['ARCA', 'NASDAQ', 'NYSE', 'BATS', 'IEX']
    
    def route_order(
        self, ticker: str, side: str, quantity: int,
        urgency: str = 'normal'
    ) -> List[Dict]:
        """
        為訂單選擇最優路由
        
        根據流動性需求和緊急程度，將大訂單拆分到多個交易所。
        
        參數:
            ticker: 股票代碼
            side: BUY/SELL
            quantity: 總數量
            urgency: 'low'/'normal'/'high'
        
        返回:
            路由分配列表 [{exchange, quantity, reason}]
        """
        routes = []
        remaining = quantity
        
        if urgency == 'high':
            # 高緊急：優先速度，使用IEX（低延遲）
            routes.append({'exchange': 'IEX', 'quantity': remaining, 'reason': '低延遲'})
            return routes
        
        # 正常情況：按費用優勢分配
        # 優先使用maker費率有回扣的交易所
        priority = ['ARCA', 'NASDAQ'] if side == 'BUY' else ['ARCA', 'BATS']
        
        for exchange in priority:
            if remaining <= 0:
                break
            # 每個交易所分配一部分（簡化：平分）
            alloc = max(remaining // len(priority), 100)  # 至少100股
            alloc = min(alloc, remaining)
            routes.append({
                'exchange': exchange,
                'quantity': alloc,
                'reason': f'maker回扣: ${abs(self.EXCHANGE_FEES[exchange]["maker"]):.4f}/股'
            })
            remaining -= alloc
        
        if remaining > 0:
            routes.append({'exchange': 'NYSE', 'quantity': remaining, 'reason': '剩餘'})
        
        return routes


# ============================================================================
# 算法單
# ============================================================================
class SmartOrderAlgo:
    """
    智能算法單
    
    支持高級執行算法：
    - TWAP: 時間加權平均價格
    - VWAP: 成交量加權平均價格
    - Iceberg: 冰山訂單（只顯示部分數量）
    """
    
    @staticmethod
    def twap_schedule(
        total_quantity: int,
        duration_minutes: int,
        interval_minutes: int = 5
    ) -> List[Dict]:
        """
        TWAP訂單拆分
        
        將大訂單拆分為多個等時間間隔的小訂單，
        以接近時間加權平均價格。
        
        參數:
            total_quantity: 總股數
            duration_minutes: 執行總時長（分鐘）
            interval_minutes: 下單間隔（分鐘）
        
        返回:
            時間表 [{time_offset_min, quantity}]
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
            f"{n_intervals}個間隔, 每{interval_minutes}分鐘"
        )
        
        return schedule
    
    @staticmethod
    def vwap_schedule(
        total_quantity: int,
        volume_profile: Dict[int, float]
    ) -> Dict[int, int]:
        """
        VWAP訂單拆分
        
        根據歷史成交量分布，將訂單按比例分配到不同時段，
        以接近成交量加權平均價格。
        
        參數:
            total_quantity: 總股數
            volume_profile: 成交量分布 {分鐘: 比例}
                           如 {10: 0.05, 11: 0.08, ...}
        
        返回:
            分配方案 {分鐘: 股數}
        """
        schedule = {}
        for minute, ratio in volume_profile.items():
            qty = int(total_quantity * ratio)
            if qty > 0:
                schedule[minute] = qty
        
        # 處理捨入誤差
        allocated = sum(schedule.values())
        if allocated < total_quantity:
            # 將剩餘分配到最後時段
            last_key = max(schedule.keys()) if schedule else 0
            schedule[last_key] = schedule.get(last_key, 0) + (total_quantity - allocated)
        
        return schedule
    
    @staticmethod
    def iceberg_params(
        total_quantity: int,
        display_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        冰山訂單參數
        
        只顯示訂單總量的一小部分（如10%），
        隱藏真實意圖，減少市場衝擊。
        
        參數:
            total_quantity: 總股數
            display_ratio: 顯示比例
        
        返回:
            冰山參數
        """
        display_qty = max(100, int(total_quantity * display_ratio))
        display_qty = min(display_qty, total_quantity)
        
        return {
            'total_quantity': total_quantity,
            'display_quantity': display_qty,
            'hidden_quantity': total_quantity - display_qty,
            'display_ratio': display_ratio,
        }
