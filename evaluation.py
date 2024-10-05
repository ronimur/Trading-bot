import pandas as pd
import numpy as np
from models import StrategySignal

def calc_total_return(portfolio_values):
    if(portfolio_values.iloc[0] == 0):
        return None
    return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0

def calc_annualized_return(portfolio_values):
    yearly_trading_days = 252
    portfolio_trading_days = portfolio_values.shape[0]
    portfolio_trading_years = portfolio_trading_days / yearly_trading_days 
    if portfolio_values.iloc[0] == 0 or portfolio_trading_years == 0 or portfolio_values.iloc[-1] < 0:
        return -1
    return ((portfolio_values.iloc[-1] / portfolio_values.iloc[0])**(1/portfolio_trading_years)) - 1.0

def calc_annualized_sharpe(portfolio_values: pd.Series, rf: float=0.0):
    yearly_trading_days = 252
    annualized_return = calc_annualized_return(portfolio_values)
    annualized_std = portfolio_values.pct_change().std() * np.sqrt(yearly_trading_days)
    if annualized_std is None or annualized_std == 0:
        return 0
    sharpe = (annualized_return - rf) / annualized_std
    return sharpe

def calc_downside_deviation(portfolio_values):
    porfolio_returns = portfolio_values.pct_change().dropna()
    return porfolio_returns[porfolio_returns < 0].std()

def calc_sortino(portfolio_values, rf=0.0):
    yearly_trading_days = 252
    down_deviation = calc_downside_deviation(portfolio_values) * np.sqrt(yearly_trading_days)
    annualized_return = calc_annualized_return(portfolio_values)
    if down_deviation is None or down_deviation == 0:
        return 0
    sortino = (annualized_return - rf) / down_deviation 
    return sortino

def calc_max_drawdown(portfolio_values):
    cumulative_max = portfolio_values.cummax()
    if any(cumulative_max.iloc[i] == 0 for i in range(cumulative_max.shape[0])):
        return None
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    return drawdown.max()

def calc_calmar(portfolio_values):
    max_drawdown = calc_max_drawdown(portfolio_values)
    annualized_return = calc_annualized_return(portfolio_values)
    if max_drawdown is None or max_drawdown == 0:
        return None
    return annualized_return / max_drawdown
   

def evaluate_strategy(b_df, strat_name, hp:dict, id:int, trades_per_year ,logger = None, print_results=False):
    total_return = calc_total_return(b_df['portfolio_value'])
    annualized_return = calc_annualized_return(b_df['portfolio_value'])
    annualized_sharpe = calc_annualized_sharpe(b_df['portfolio_value'])
    sortino_ratio = calc_sortino(b_df['portfolio_value'])
    max_drawdown = calc_max_drawdown(b_df['portfolio_value'])
    calmar_ratio = calc_calmar(b_df['portfolio_value'])
    sum_trades = trades_per_year.sum()

    
    if print_results:
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print(f'num of trades: {sum_trades}')
        
    if logger is not None:
        if total_return is not None and annualized_return is not None and annualized_sharpe is not None and sortino_ratio is not None and max_drawdown is not None and calmar_ratio is not None:
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Annualized Return: {annualized_return:.2%}")
            logger.info(f"Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
            logger.info(f"Sortino Ratio: {sortino_ratio:.2f}")
            logger.info(f"Max Drawdown: {max_drawdown:.2%}")
            logger.info(f"Calmar Ratio: {calmar_ratio:.2f}")
            logger.info(f'num of trades: {sum_trades}')


    return {'id': id,
            'strategy': strat_name,
            'Zscore window': hp['z_score_window'],
            'Zscore treshold': hp['z_score_treshold'],
            'ATR window': hp['atr_window'],
            'Gap&Go SL/TP ratio': hp['gapNgo_tp_sl_ratio'],
            'Close Gap SL/TP ratio': hp['close_gap_tp_sl_ratio'],
            'total return': total_return,
            'annualized return': annualized_return,
            'annualized sharpe': annualized_sharpe,
            'sortino ratio': sortino_ratio,
            'max drawdown': max_drawdown,
            'calmar ratio': calmar_ratio,
            '2012': trades_per_year[2012] if 2012 in trades_per_year.index else 0,
            '2012 return': calc_total_return(b_df[b_df['year'] == 2012]['portfolio_value']) if 2012 in b_df['year'].unique() else 0,
            '2013': trades_per_year[2013] if 2013 in trades_per_year.index else 0,
            '2013 return': calc_total_return(b_df[b_df['year'] == 2013]['portfolio_value']) if 2013 in b_df['year'].unique() else 0,
            '2014': trades_per_year[2014] if 2014 in trades_per_year.index else 0,
            '2014 return': calc_total_return(b_df[b_df['year'] == 2014]['portfolio_value']) if 2014 in b_df['year'].unique() else 0,
            '2015': trades_per_year[2015] if 2015 in trades_per_year.index else 0,
            '2015 return': calc_total_return(b_df[b_df['year'] == 2015]['portfolio_value']) if 2015 in b_df['year'].unique() else 0,
            '2016': trades_per_year[2016] if 2016 in trades_per_year.index else 0,
            '2016 return': calc_total_return(b_df[b_df['year'] == 2016]['portfolio_value']) if 2016 in b_df['year'].unique() else 0,
            '2017': trades_per_year[2017] if 2017 in trades_per_year.index else 0,
            '2017 return': calc_total_return(b_df[b_df['year'] == 2017]['portfolio_value']) if 2017 in b_df['year'].unique() else 0,
            '2018': trades_per_year[2018] if 2018 in trades_per_year.index else 0,
            '2018 return': calc_total_return(b_df[b_df['year'] == 2018]['portfolio_value']) if 2018 in b_df['year'].unique() else 0,
            '2019': trades_per_year[2019] if 2019 in trades_per_year.index else 0,
            '2019 return': calc_total_return(b_df[b_df['year'] == 2019]['portfolio_value']) if 2019 in b_df['year'].unique() else 0,
            '2020': trades_per_year[2020] if 2020 in trades_per_year.index else 0,
            '2020 return': calc_total_return(b_df[b_df['year'] == 2020]['portfolio_value']) if 2020 in b_df['year'].unique() else 0,
            '2021': trades_per_year[2021] if 2021 in trades_per_year.index else 0,
            '2021 return': calc_total_return(b_df[b_df['year'] == 2021]['portfolio_value']) if 2021 in b_df['year'].unique() else 0,
            '2022': trades_per_year[2022] if 2022 in trades_per_year.index else 0,
            '2022 return': calc_total_return(b_df[b_df['year'] == 2022]['portfolio_value']) if 2022 in b_df['year'].unique() else 0,
            '2023': trades_per_year[2023] if 2023 in trades_per_year.index else 0,
            '2023 return': calc_total_return(b_df[b_df['year'] == 2023]['portfolio_value']) if 2023 in b_df['year'].unique() else 0,
            'total_trades': sum_trades}