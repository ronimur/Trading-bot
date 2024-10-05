from enum import Enum

class StrategySignal(Enum):
    ENTER_LONG = 2
    ENTER_SHORT = 1
    DO_NOTHING = 0
    CLOSE_SHORT = -1
    CLOSE_LONG = -2
    BUY = 3
    SELL = -3
    TAKEPROFIT = 4
    STOPLOSS = -4

class PositionType(Enum):
    LONG = 1
    SHORT = -1

class ActionType(Enum):
    BUY = 1
    SELL = -1

class GapSignal(Enum):
    Gap_N_Go = 1
    Close_Gap = -1

class Position():
    def __init__(self, qty: float, price: float, takeProfit : float, stopLoss : float, type: PositionType, gapSignal: GapSignal):
        self.qty = qty
        self.price = price
        self.takeProfit = takeProfit
        self.stopLoss = stopLoss
        self.type = type
        self.gapSignal = gapSignal