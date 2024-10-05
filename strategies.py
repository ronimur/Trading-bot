import pandas as pd
import numpy as np
from datetime import datetime
from models import *
from abc import ABC, abstractmethod
from typing import Tuple
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class BaseStrategy(ABC):
    def __init__(self, sl_rate: float=None, tp_rate: float=None) -> None:
        super().__init__()
        if(sl_rate < 0 or tp_rate < 0 or sl_rate > 1 or tp_rate > 1):
            raise ValueError("sl_rate and tp_rate must be between 0 and 1")
        self.sl_rate = sl_rate
        self.tp_rate = tp_rate
    
    @abstractmethod
    def calc_signal(self, data: pd.DataFrame, hp: dict = None) -> pd.DataFrame:
        pass

    def calc_qty(self, real_price: float, balance: float, action: ActionType, **kwargs) -> int:
        if action == ActionType.BUY:
            qty = (int)(balance / real_price)
        
        elif action == ActionType.SELL:
            qty = (int)(balance / real_price)
        
        return qty    
    
    def check_sl_tp(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        sl_res = self.is_stop_loss(row, position)
        if sl_res is not None:
            return sl_res
        
        tp_res = self.is_take_profit(row, position)
        if tp_res is not None:
            return tp_res
    
    def is_stop_loss(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the stop-loss level.
        
        Returns:
            Tuple[float, float, ActionType] or None: If stop-loss is triggered, returns a tuple containing quantity and stop-loss price and action type, otherwise returns None.
        """
        if self.sl_rate is not None:
            long_stop_loss_price = position.price * (1 - self.sl_rate)
            if position.type == PositionType.LONG and row['low'] <= long_stop_loss_price:
                return position.qty, long_stop_loss_price, ActionType.SELL
            
            short_stop_loss_price = position.price * (1 + self.sl_rate)
            if position.type == PositionType.SHORT and row['high'] >= short_stop_loss_price:
                return position.qty, short_stop_loss_price, ActionType.BUY
    
    def is_take_profit(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the take-profit level.

        Returns:
            Tuple[float, float, ActionType] or None: If take-profit is triggered, returns a tuple containing quantity and take-profit price and action type, otherwise returns None.
        """
        if self.tp_rate is not None:
            long_take_profit_price = position.price * (1 + self.tp_rate)
            if position.type == PositionType.LONG and row['high'] >= long_take_profit_price:
                return position.qty, long_take_profit_price, ActionType.SELL
            
            short_take_profit_price = position.price * (1 - self.tp_rate)
            if position.type == PositionType.SHORT and row['low'] <= short_take_profit_price:
                return position.qty, short_take_profit_price, ActionType.BUY
    
class CatBoost_model(BaseStrategy):
    # ---------------------- init ----------------------
    def __init__(self, sl_rate: float=100.0, tp_rate: float=100.0):
        super().__init__(sl_rate/100, tp_rate/100)

    # ---------------------- Strategy's indicators / Model features calculation functions ----------------------
    def calc_gap_down(self, data:pd.DataFrame, gap_threshold: float = 2.0)-> pd.Series:
        """calculate gaps for long positions towards closing the gap"""
        gap = pd.Series([False] * len(data), index=data.index)
        gap_pct_series = pd.Series([0.0] * len(data), index=data.index)
        for i in range(1, len(data)):
            gap_pct = ((data.loc[i-1,'low'] - data.loc[i,'open']) / data.loc[i-1,'low'])
            gap_pct_series[i] = gap_pct
            if gap_pct < 0:
                gap[i] = False
            else:
                gap[i] = (gap_pct >= (gap_threshold/100))
        return gap, gap_pct_series
        
    def calc_gap_up(self, data:pd.DataFrame, gap_threshold: float = 2.0)-> pd.Series:
        """calculate gaps for short positions towards closing the gap"""
        gap = pd.Series([False] * len(data), index=data.index)
        gap_pct_series = pd.Series([0.0] * len(data), index=data.index)
        for i in range(1, len(data)):
            gap_pct = ((data.loc[i,'open'] - data.loc[i-1,'high']) / data.loc[i-1,'high'])
            gap_pct_series[i] = gap_pct
            if gap_pct < 0:
                gap[i] = False
            else:
                gap[i] = (gap_pct >= (gap_threshold/100))
        return gap, gap_pct_series

    def calc_Gaps(self, data: pd.DataFrame):
        gap_down, pct_down = self.calc_gap_down(data)
        gap_up, pct_up = self.calc_gap_up(data)
        return gap_down, pct_down, gap_up, pct_up      
    
    def calc_ATR(self, data:pd.DataFrame, period:int = 14)-> pd.Series:
        """calculate ATR with shift(1) for a given period"""
        return (np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))).rolling(window=period).mean().shift(1)
    
    def calc_RSI(self,data:pd.DataFrame, period:int = 14)-> pd.Series:
        """calculate RSI with shift(1) for a given period"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.shift(1)

    def calc_vix_level(self, data:pd.DataFrame)-> pd.Series:
        """returns vix level of prev day (by using shift(1)) for each row"""
        vix_df = pd.read_pickle(f'..\\1- data\wrds 1day Data\\vix.pickle')
        #go over each row in data and find the vix level of the previous day
        vix_level = pd.Series([0.0] * len(data), index=data.index)
        for i in range(0, len(data)):
            vix_level[i] = vix_df[vix_df['date'] == data.loc[i,'date']]['close'].values[0]
        return vix_level.shift(1)

    def calc_mov_avg(self,data:pd.DataFrame, period:int = 14)-> pd.Series:
        """calculate moving average with shift(1) for a given period"""
        return data['close'].rolling(window=period).mean().shift(1)

    def calc_ROC(self,data:pd.DataFrame, period:int = 14)-> pd.Series:
        """calculate rate of change with shift(1) for a given period"""
        return data['close'].diff(period) / data['close'].shift(period).shift(1)

    def calc_7_day_momentum(self,data:pd.DataFrame)-> pd.Series:
        """calculate the momentum of the last 7 days with shift(1)"""
        return data['close'].rolling(window=7, min_periods=1).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x)==7 else float('nan')).shift(1)
    
    def calc_7_day_std(self,data:pd.DataFrame)-> pd.Series:
        """calculate the standard deviation of the last 7 days with shift(1)"""
        return data['pct_change'].rolling(window=7).std(ddof=0).shift(1)
    
    def calc_prev_day_data(self, data:pd.DataFrame)-> pd.Series:
        """calculate the direction and momentum of the previous day (by using shift(1))"""
        day_momentum = pd.Series([0.0] * len(data), index=data.index)
        ohlc_ratio = pd.Series([0.0] * len(data), index=data.index)
        for i in range(0, len(data)):
            day_momentum[i] = (data.loc[i, 'close'] - data.loc[i, 'open']) / data.loc[i, 'open']
            ohlc_ratio[i] = (data.loc[i,'close'] - data.loc[i,'open']) / (data.loc[i,'high'] - data.loc[i,'low'])
        return day_momentum.shift(1), ohlc_ratio.shift(1)
    
    def calculateATR(self, data: pd.DataFrame, n: int)-> pd.Series:
        """calculate the average true range for UtBot calculation usage"""
        tr_series = np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))
        # ATR = TR n periods average
        atr_series = tr_series.rolling(n).mean()
        return atr_series
    
    def UtBot(self, data:pd.DataFrame, sensitivity: float = 2.0, atrPeriod: int = 1) -> pd.Series:
        """calculate the UtBot signals using ATR and sensitivity - return a series of signals with shift(1)"""
        ATR = self.calculateATR(data, atrPeriod)
        loss_threshold = sensitivity * ATR
        trailing_stop = np.zeros(len(data), dtype=float) #change
        for i in range(1, len(data)):
            #--loop variables
            currClose = data['close'].iloc[i]
            prevClose = data['close'].iloc[i-1]
            prevTrailingStop = trailing_stop[i-1]
            currLossThreshold = loss_threshold[i]
            #--logic
            if currClose > prevTrailingStop and prevClose > prevTrailingStop:
                trailing_stop[i] = max(prevTrailingStop, currClose - currLossThreshold)
            elif currClose < prevTrailingStop and prevClose < prevTrailingStop:
                trailing_stop[i] = min(prevTrailingStop, currClose + currLossThreshold)
            elif currClose > prevTrailingStop:
                trailing_stop[i] = currClose - currLossThreshold
            else:
                trailing_stop[i] = currClose + currLossThreshold
        #--signals
        signals = pd.Series([StrategySignal.DO_NOTHING] * len(data))
        for i in range(1, len(data)-1):
            #--loop variables
            currClose = data['close'][i]
            prevClose = data['close'][i-1]
            currTrailingStop = trailing_stop[i]
            prevTrailingStop = trailing_stop[i-1]
            #--logic
            if prevClose < prevTrailingStop and currClose > currTrailingStop: 
                signals.iloc[i] = StrategySignal.BUY
            elif prevClose > prevTrailingStop and currClose < currTrailingStop:
                signals.iloc[i] = StrategySignal.SELL

        return signals.shift(1) # the signal is calaulated when close info is available, so the you can act using it in the next candle open price at the earliest - that's why we use shift(1)
    
    def zscore(self, data: pd.DataFrame, window: int) -> pd.Series:
        """calculate the zscore of the close price for a given window"""
        close_mean = data['close'].rolling(window=window).mean()
        close_std = data['close'].rolling(window=window).std()
        zscore = (data['close'] - close_mean) / close_std
        return zscore
    
    def zscore_signals(self, zscore: pd.Series, zscore_threshold: float) -> pd.Series:
        """return a series of True / False values, where True indicates that the zscore crossed the threshold -> with shift(1)"""
        signals = pd.Series([False] * len(zscore), index=zscore.index)
        for i in range(1, len(zscore)):
            if zscore[i-1] <= zscore_threshold and zscore[i] > zscore_threshold: #crossed from below
                signals[i] = True 
            elif zscore[i] < zscore_threshold and zscore[i-1] >= zscore_threshold: #crossed from above
                signals[i] = True
        return signals.shift(1)
    
    def calc_tp_sl(self, data: pd.DataFrame, index : int, gapNgo_tp_sl_ratio: float, close_gap_tp_sl_ratio: float)-> pd.DataFrame:
        """calculate take profit and stop loss levels"""
        open_price = data.loc[index,'open']
        stop_loss = 0.0
        take_profit = 0.0
        if data.loc[index,'gap_direction'] == 'Up':

            if data.loc[index,'strategy_signal'] == StrategySignal.ENTER_LONG: # Gap and Go
                stop_loss = data.loc[index-1,'high']
                take_profit = ((open_price - stop_loss) * gapNgo_tp_sl_ratio) + open_price

            elif data.loc[index,'strategy_signal'] == StrategySignal.ENTER_SHORT: # Close Gap
                take_profit = data.loc[index-1,'high']
                price_range = open_price - take_profit
                stop_loss = (price_range) + open_price
                take_profit = open_price - (close_gap_tp_sl_ratio * price_range)
                
        elif data.loc[index,'gap_direction'] == 'Down':

            if data.loc[index,'strategy_signal'] == StrategySignal.ENTER_LONG: # Close Gap
                take_profit = data.loc[index-1,'low']
                price_range = take_profit - open_price
                stop_loss = open_price - (take_profit - open_price)
                take_profit = open_price + (close_gap_tp_sl_ratio * price_range)

            elif data.loc[index,'strategy_signal'] == StrategySignal.ENTER_SHORT: # Gap and Go
                stop_loss = data.loc[index-1,'low']
                take_profit = open_price - (gapNgo_tp_sl_ratio * (stop_loss - open_price) )

        return stop_loss, take_profit
        
    # ---------------------- Strategy's signals calculation ----------------------
    def calc_signal(self, data: pd.DataFrame, hp: dict)-> pd.DataFrame:
        # ---------------------- Gap class ----------------------
        class Gap:
            def __init__(self, row: pd.Series, index: int, gap_pct: float, gap_direction: str, prev_gap_date):
            # prepare features
                self.date = row['date']
                self.gap_pct = gap_pct
                self.gap_direction = gap_direction
                self.momentum_of_last_7_days = row['momentum of last 7 days']
                self.momentum_of_last_day = row['momentum of last day']
                self.ROC_5 = row['ROC 5']
                self.ROC_7 = row['ROC 7']
                self.ROC_14 = row['ROC 14']
                self.std_of_last_7_days = row['std of last 7 days']
                self.RSI = row['RSI']
                self.VIX = row['VIX']
                self.prev_OHLC_ratio = row['prev OHLC ratio']
                self.day = row['day']
                self.week = row['week']

            def update_scaled_features(self, features: pd.DataFrame):
                self.RSI = features.loc[0,'RSI']
                self.VIX = features.loc[0,'VIX']
                self.day = features.loc[0,'day']
                self.week = features.loc[0,'week']

            def get_features(self, scaler, features_to_scale)-> pd.DataFrame:
                """return all features as an array, after scaling them, by the order of the features in init"""
                features_dict = {'gap_pct': self.gap_pct,'gap_direction': self.gap_direction,
                                    'momentum of last 7 days': self.momentum_of_last_7_days,
                                        'momentum of last day': self.momentum_of_last_day,
                                            'ROC 5': self.ROC_5, 'ROC 7': self.ROC_7, 'ROC 14': self.ROC_14,
                                                'std of last 7 days': self.std_of_last_7_days,
                                                    'RSI': self.RSI, 
                                                        'VIX': self.VIX,
                                                            'prev OHLC ratio': self.prev_OHLC_ratio,
                                                                'day': self.day, 'week': self.week}
                features = pd.DataFrame(features_dict, index=[0])
                features[features_to_scale] = scaler.transform(features[features_to_scale])
                self.update_scaled_features(features)
                return features
            
            def export_gap(self, symbol, position_type, strategy_type, prob_prediction):
                prob_of_0 = prob_prediction[0][0]
                prob_of_1 = prob_prediction[0][1]
                return {'symbol': symbol,
                            'date': self.date,
                                'gap_pct': self.gap_pct,
                                    'gap_direction': self.gap_direction,
                                        'momentum_of_last_7_days': self.momentum_of_last_7_days,
                                                'momentum_of_last_day': self.momentum_of_last_day,
                                                    'ROC_5': self.ROC_5,
                                                        'ROC_7': self.ROC_7,
                                                            'ROC_14': self.ROC_14,
                                                                'std_of_last_7_days': self.std_of_last_7_days,
                                                                    'RSI': self.RSI,
                                                                        'VIX': self.VIX,
                                                                            'prev_OHLC_ratio': self.prev_OHLC_ratio,
                                                                                'day': self.day,
                                                                                    'week': self.week,
                                                                                        'position_type': position_type,
                                                                                            'strategy_type': strategy_type,
                                                                                                'prob_of_short': prob_of_0,
                                                                                                    'prob_of_long': prob_of_1}

        # ---------------------- Strategy's indicators calculation functions ----------------------
        data['zscore'] = self.zscore(data, window = hp['z_score_window'])
        data['CloseGap_exit_signals'] = self.zscore_signals(data['zscore'], zscore_threshold = hp['z_score_treshold'])
        data['Gap&Go_exit_signals'] = self.UtBot(data, sensitivity = 3.0, atrPeriod = hp['atr_window'])

        # ---------------------- Model features calculation ----------------------
        data['gap_down'], data['gap_down_pct'], data['gap_up'], data['gap_up_pct'] = self.calc_Gaps(data)
        data['RSI'] = self.calc_RSI(data)
        data['VIX'] = self.calc_vix_level(data)
        data['ROC 5'] = self.calc_ROC(data, period=5)
        data['ROC 7'] = self.calc_ROC(data, period=7)
        data['ROC 14'] = self.calc_ROC(data, period=14)
        data['momentum of last 7 days'] = self.calc_7_day_momentum(data)
        data['std of last 7 days'] = self.calc_7_day_std(data)
        data['momentum of last day'], data['prev OHLC ratio'] = self.calc_prev_day_data(data)
        data['day'] = data['date'].dt.dayofweek
        data['week'] = data['date'].dt.isocalendar().week
        data['month'] = data['date'].dt.month
        data['Year'] = data['date'].dt.year
        curr_year = data['Year'].min()
        category_columns = ['gap_direction','industry','sub_industry','gap_volume_direction','direction of last day']
        columns_to_scale = ['RSI','ATR','VIX','distance_from_prev_gap','day','month','SMA 5','SMA 10','SMA 20','distance_from_prev_gap','day','week','month']
        columns_to_drop = ['gap_volume_change','gap_volume_direction','direction of last day','ATR', 'SMA 5', 'SMA 10', 'SMA 20', 'distance_from_prev_gap', 'month', 'industry', 'sub_industry']
        category_columns = [col for col in category_columns if col not in columns_to_drop] # if col not in columns_to_drop than it stays in the list
        columns_to_scale = [col for col in columns_to_scale if col not in columns_to_drop] # if col not in columns_to_drop than it stays in the list

        # export_gaps = []
        catBoost_model = None
        prev_gap_date = None
        classifier = CatBoostClassifier(cat_features=category_columns).load_model(f'..\\3- catBoost\\models\\catboost_model_{curr_year-1}.cbm')
        scaler = joblib.load(f'..\\3- catBoost\\models\\scaler{curr_year-1}.pkl')

        symbol = data.loc[0,'symbol']
        for index,row in data.iterrows():
            if row.isnull().any():
                continue

            if row['Year'] != curr_year and row['month'] >= 2: #update model when entering fabruary of a new year
                curr_year = row['Year']
                catBoost_model = CatBoostClassifier(cat_features=category_columns).load_model(f'..\\3- catBoost\\models\\catboost_model_{curr_year-1}.cbm')
                scaler = joblib.load(f'..\\3- catBoost\\models\\scaler{curr_year-1}.pkl')

            if (data.loc[index,'gap_down'] == True) and (not pd.isnull(row['VIX'])):
                data.loc[index,'gap_direction'] = 'Down'
                gap = Gap(row = row, index = index, gap_pct = (-1.0)*row['gap_down_pct'], gap_direction = 'Down', prev_gap_date = prev_gap_date)
                prev_gap_date = data.loc[index,'date']
                features = gap.get_features(scaler=scaler,features_to_scale=columns_to_scale)
                prediction = classifier.predict(features)
                prob_prediction = classifier.predict_proba(features)
                if prediction.size == 0:
                    print(f'\ngap for long\n\tsymbol: {symbol}\n\tdate: {row["date"]}\n\tgap_pct: {row["gap_down_pct"]})')
                    continue
                data.loc[index, 'strategy_signal'] = StrategySignal.ENTER_LONG if prediction[0] == [1] else StrategySignal.ENTER_SHORT
                strategy_signal = 'ENTER_LONG' if prediction[0] == [1] else 'ENTER_SHORT'
                data.loc[index, 'type of gap strategy'] = GapSignal.Close_Gap if prediction[0] == [1] else GapSignal.Gap_N_Go
                type_of_gap_strategy = 'Close_Gap' if prediction[0] == [1] else 'Gap_N_Go'
                data.loc[index, 'prob for short'] = prob_prediction[0][0]
                data.loc[index, 'prob for long'] = prob_prediction[0][1]
                data.loc[index, 'stopLoss'], data.loc[index, 'takeProfit'] = self.calc_tp_sl(data, index, hp['gapNgo_tp_sl_ratio'], hp['close_gap_tp_sl_ratio'])
                gap = None

            elif (data.loc[index,'gap_up'] == True) and (not pd.isnull(row['VIX'])):
                data.loc[index,'gap_direction'] = 'Up'
                gap = Gap(row = row, index = index, gap_pct = row['gap_up_pct'], gap_direction = 'Up', prev_gap_date = prev_gap_date)
                prev_gap_date = data.loc[index,'date']
                features = gap.get_features(scaler=scaler,features_to_scale=columns_to_scale)
                prediction = classifier.predict(features)
                prob_prediction = classifier.predict_proba(features)
                if prediction.size == 0:
                    print(f'\ngap for long\n\tsymbol: {symbol}\n\tdate: {row["date"]}\n\tgap_pct: {row["gap_up_pct"]})')
                    continue
                data.loc[index, 'strategy_signal'] = StrategySignal.ENTER_LONG if prediction[0] == [1] else StrategySignal.ENTER_SHORT
                strategy_signal = 'ENTER_LONG' if prediction[0] == [1] else 'ENTER_SHORT'
                data.loc[index, 'type of gap strategy'] = GapSignal.Gap_N_Go if prediction[0] == [1] else GapSignal.Close_Gap
                type_of_gap_strategy = 'Gap_N_Go' if prediction[0] == [1] else 'Close_Gap'
                data.loc[index, 'prob for short'] = prob_prediction[0][0]
                data.loc[index, 'prob for long'] = prob_prediction[0][1]
                data.loc[index, 'stopLoss'], data.loc[index, 'takeProfit'] = self.calc_tp_sl(data, index, hp['gapNgo_tp_sl_ratio'], hp['close_gap_tp_sl_ratio'])
                gap = None
        return data

                
