import pandas as pd
import numpy as np

from models import ActionType, PositionType, Position, StrategySignal
from strategies import *
from evaluation import evaluate_strategy         
    
def calc_realistic_price(row: pd.Series ,action_type: ActionType, close_col : str, open_col : str, slippage_factor=np.inf):
    slippage_rate = ((row[close_col] - row[open_col]) / row[open_col]) / slippage_factor
    slippage_price = row[open_col] + row[open_col] * slippage_rate
    
    if action_type == ActionType.BUY:
        return max(slippage_price, row[open_col])
    else:
        return min(slippage_price, row[open_col])   

def backtest(data: pd.DataFrame, strategy: BaseStrategy, starting_balance: int, slippage_factor: float=30.0, commission: float=3.0, close_col : str ='Close', open_col : str ='Open', hp : dict = None) -> pd.DataFrame:

    if hp is None:
        print('No hyperparameters were passed')
        print('Exiting...')
        exit(1)

    def enter_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position_type: PositionType) -> Position:
        if position_type == PositionType.LONG:
            buy_price = calc_realistic_price(row, ActionType.BUY, close_col, open_col, slippage_factor=slippage_factor)
            qty_to_buy = strategy.calc_qty(buy_price, curr_balance, ActionType.BUY)
            position = Position(qty=qty_to_buy, price=buy_price, takeProfit=data.loc[index, 'takeProfit'], stopLoss=data.loc[index, 'stopLoss'], type=position_type, gapSignal=data.loc[index, 'type of gap strategy'])
            data.loc[index, 'qty'] = curr_qty + qty_to_buy
            commission_price = (abs(commission*qty_to_buy) if abs(qty_to_buy) > 100 else 1.0) if commission > 0 else 0.0
            data.loc[index, 'balance'] = curr_balance - qty_to_buy * buy_price - commission_price
        
        elif position_type == PositionType.SHORT:
            sell_price = calc_realistic_price(row, ActionType.SELL, close_col, open_col, slippage_factor=slippage_factor)
            qty_to_sell = strategy.calc_qty(sell_price, curr_balance, ActionType.SELL)
            position = Position(qty=qty_to_sell, price=sell_price, takeProfit=data.loc[index, 'takeProfit'], stopLoss=data.loc[index, 'stopLoss'], type=position_type, gapSignal=data.loc[index, 'type of gap strategy'])
            data.loc[index, 'qty'] = curr_qty - qty_to_sell
            commission_price = (abs(commission*qty_to_sell) if abs(qty_to_sell) > 100 else 1.0) if commission > 0 else 0.0
            data.loc[index, 'balance'] = curr_balance + qty_to_sell * sell_price - commission_price

        
        return position

    def close_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position: Position, tp_sl: bool = False, end_of_data: bool=False, exit_signal: str = '') -> None:
        if position.type == PositionType.LONG:
            if tp_sl:
                sell_price = position.takeProfit if row['high'] >= position.takeProfit else position.stopLoss
                exit_signal = 'TakeProfit' if row['high'] >= position.takeProfit else 'StopLoss'
            elif end_of_data:
                sell_price = row[close_col]
                exit_signal = 'End of data'
            else:
                sell_price = calc_realistic_price(row, ActionType.SELL, close_col, open_col, slippage_factor=slippage_factor)
            data.loc[index, 'qty'] = curr_qty - position.qty
            commission_price = (abs(commission*position.qty) if abs(position.qty) > 100 else 1.0) if commission > 0 else 0.0
            data.loc[index, 'balance'] = curr_balance + position.qty * sell_price - commission_price
            data.loc[index, 'close_position_price'] = sell_price 
            data.loc[index, 'profit'] = (sell_price-position.price)*position.qty 
            data.loc[index, 'profit or loss trade'] = 'Profit' if data.loc[index, 'profit'] > 0 else 'Loss'
            data.loc[index, 'exit_signal'] = exit_signal
        
        elif position.type == PositionType.SHORT:
            if tp_sl:
                buy_price = position.takeProfit if row['low'] <= position.takeProfit else position.stopLoss
                exit_signal = 'TakeProfit' if row['low'] <= position.takeProfit else 'StopLoss'
            elif end_of_data:
                buy_price = row[close_col]
                exit_signal = 'End of data'
            else:
                buy_price = calc_realistic_price(row, ActionType.BUY, close_col, open_col, slippage_factor=slippage_factor)
            data.loc[index, 'qty'] = curr_qty + position.qty
            commission_price = (abs(commission*position.qty) if abs(position.qty) > 100 else 1.0) if commission > 0 else 0.0
            data.loc[index, 'balance'] = curr_balance - position.qty * buy_price - commission_price
            data.loc[index, 'close_position_price'] = buy_price
            data.loc[index, 'profit'] = (position.price-buy_price)*position.qty
            data.loc[index, 'profit or loss trade'] = 'Profit' if data.loc[index, 'profit'] > 0 else 'Loss'
            data.loc[index, 'exit_signal'] = exit_signal


    def handleStopLossTakeProfit(data: pd.DataFrame, index: int, curr_qty: float, curr_balance: float, position: Position, sl_tp_res: Tuple) -> Tuple[float, float, StrategySignal]:
        sl_tp_qty, sl_tp_price, sl_tp_action = sl_tp_res
        strategyCloseAction = None
        positionEnterPrice = position.price
        if sl_tp_action == ActionType.BUY:
            commission_price = (abs(commission*sl_tp_qty) if abs(sl_tp_qty) > 100 else 1.0) if commission > 0 else 0.0
            curr_balance = curr_balance - (sl_tp_qty * sl_tp_price) - commission_price
            curr_qty = curr_qty + sl_tp_qty
            strategyCloseAction = StrategySignal.TAKEPROFIT if positionEnterPrice > sl_tp_price else StrategySignal.STOPLOSS
            data.loc[index-1,'profit'] = (positionEnterPrice-sl_tp_price)*sl_tp_qty
            data.loc[index-1, 'profit or loss trade'] = 'Profit' if data.loc[index-1, 'profit'] > 0 else 'Loss'
            data.loc[index-1, 'exit_signal'] = 'TakeProfit' if positionEnterPrice > sl_tp_price else 'StopLoss'

        elif sl_tp_action == ActionType.SELL:
            commission_price = (abs(commission*sl_tp_qty) if abs(sl_tp_qty) > 100 else 1.0) if commission > 0 else 0.0
            curr_balance = curr_balance + (sl_tp_qty * sl_tp_price) - commission_price
            curr_qty = curr_qty - sl_tp_qty
            strategyCloseAction = StrategySignal.TAKEPROFIT if positionEnterPrice < sl_tp_price else StrategySignal.STOPLOSS
            data.loc[index-1,'profit'] = (sl_tp_price-positionEnterPrice)*sl_tp_qty
            data.loc[index-1, 'profit or loss trade'] = 'Profit' if data.loc[index-1, 'profit'] > 0 else 'Loss'
            data.loc[index-1, 'exit_signal'] = 'TakeProfit' if positionEnterPrice < sl_tp_price else 'StopLoss'

        data.loc[index-1,'close_position_price'] = sl_tp_price
        data.loc[index-1, 'qty'] = curr_qty
        data.loc[index-1, 'balance'] = curr_balance
        return curr_qty, curr_balance, strategyCloseAction
    
    # initialize df 
    data['qty'] = 0.0
    data['balance'] = 0.0
    data['open_position_price'] = 0.0
    data['close_position_price'] = 0.0 
    data['position_indicator'] = False

    # Calculate strategy signal
    strategy.calc_signal(data, hp)
    
    # Loop through the data to calculate portfolio value
    position: Position = None
    data.reset_index(inplace=True)
    num_trading_candles = data.shape[0]

    for index, row in data.iterrows():
        curr_qty = data.loc[index - 1, 'qty'] if index > 0 else 0
        curr_balance = data.loc[index - 1, 'balance'] if index > 0 else starting_balance
        keep_same_state = True
        
        # handle stop loss and take profit -> pay attention that this is not the strategy stopLoss/TakeProfit but the user percentage limits (will be shown differntly in the visual graph)
        if position is not None:
            sl_tp_res = strategy.check_sl_tp(data.iloc[index - 1], position)
            if sl_tp_res is not None:
                curr_qty, curr_balance, strategyCloseAction = handleStopLossTakeProfit(data, index, curr_qty, curr_balance, position, sl_tp_res)
                data.loc[index-1, 'position close order'] = strategyCloseAction
                position = None 
        
        # Close position at end of data
        if index + 1 == num_trading_candles and position is not None: 
            close_position(data, index, row, curr_qty, curr_balance, position, end_of_data=True)
            data.loc[index, 'position close order'] = StrategySignal.CLOSE_LONG if position.type == PositionType.LONG else StrategySignal.CLOSE_SHORT

        # Handle enter long signal
        elif row['strategy_signal'] == StrategySignal.ENTER_LONG and position is None:
            position = enter_position(data, index, row, curr_qty, curr_balance, PositionType.LONG)
            data.loc[index, 'open_position_price'] = position.price 
            data.loc[index, 'position open order'] = StrategySignal.ENTER_LONG
            data.loc[index, 'position_indicator'] = True
            keep_same_state = False
            curr_qty = data.loc[index, 'qty']
            curr_balance = data.loc[index, 'balance']
        
        # Handle enter short signal  
        elif row['strategy_signal'] == StrategySignal.ENTER_SHORT and position is None:
            position = enter_position(data, index, row, curr_qty, curr_balance, PositionType.SHORT)
            data.loc[index, 'open_position_price'] = position.price
            data.loc[index, 'position open order'] = StrategySignal.ENTER_SHORT
            data.loc[index, 'position_indicator'] = True
            keep_same_state = False
            curr_qty = data.loc[index, 'qty']
            curr_balance = data.loc[index, 'balance']

        # check if there is an exit signal on 'Gap&Go_exit_signals'
        elif position is not None and position.gapSignal == GapSignal.Gap_N_Go:
            if (position.type == PositionType.LONG and row['Gap&Go_exit_signals'] == StrategySignal.SELL) or (position.type == PositionType.SHORT and row['Gap&Go_exit_signals'] == StrategySignal.BUY):
                close_position(data, index, row, curr_qty, curr_balance, position, exit_signal='UTbot')
                data.loc[index, 'position close order'] = StrategySignal.CLOSE_LONG if position.type == PositionType.LONG else StrategySignal.CLOSE_SHORT
                data.loc[index, 'position_indicator'] = True
                position = None
                keep_same_state = False

        # check if there is an exit signal on 'CloseGap_exit_signals'
        elif position is not None and position.gapSignal == GapSignal.Close_Gap:
            if row['CloseGap_exit_signals'] == True:
                close_position(data, index, row, curr_qty, curr_balance, position, exit_signal='z_score')
                data.loc[index, 'position close order'] = StrategySignal.CLOSE_LONG if position.type == PositionType.LONG else StrategySignal.CLOSE_SHORT
                data.loc[index, 'position_indicator'] = True
                data.loc[index, 'exit signal'] = 'z_score'
                position = None
                keep_same_state = False

        
        # Handle close position based on take profit or stop loss pre-defined prices
        if position is not None and ((position.type == PositionType.LONG and row['high'] >= position.takeProfit) 
                                       or
                                         (position.type == PositionType.SHORT and row['low'] <= position.takeProfit)):
            close_position(data, index, row, curr_qty, curr_balance, position, tp_sl=True)
            data.loc[index, 'position close order'] = StrategySignal.CLOSE_LONG if position.type == PositionType.LONG else StrategySignal.CLOSE_SHORT
            data.loc[index, 'position_indicator'] = True
            position = None
            keep_same_state = False

        elif position is not None and ((position.type == PositionType.LONG and row['low'] <= position.stopLoss)
                                       or
                                         (position.type == PositionType.SHORT and row['high'] >= position.stopLoss)):
            close_position(data, index, row, curr_qty, curr_balance, position, tp_sl=True)
            data.loc[index, 'position close order'] = StrategySignal.CLOSE_LONG if position.type == PositionType.LONG else StrategySignal.CLOSE_SHORT
            data.loc[index, 'position_indicator'] = True
            position = None
            keep_same_state = False
                    
        if keep_same_state:
            data.loc[index, 'qty'] = curr_qty
            data.loc[index, 'balance'] = curr_balance
            data.loc[index, 'open_position_price'] = data.loc[index, 'close']
            data.loc[index, 'close_position_price'] = data.loc[index, 'close']
            if position is not None:
                data.loc[index, 'position_indicator'] = True

        
    
    # Calculate portfolio value
    data['portfolio_value'] = data['close'] * data['qty'] + data['balance']
    return data

