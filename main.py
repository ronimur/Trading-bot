from backtesting import *
from strategies import *
from evaluation import *
from ploting_graph import *
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
from time import sleep
from twilio.rest import Client
import psutil
from itertools import product

#----- GLOBAL VARIABLES -----# True / False
IN_SAMPLE = True 
OUT_SAMPLE = not IN_SAMPLE

# class to store the results of the optimization back to the main process from worker process
class stockBacktest:
    def __init__(self, id, symbol, portfolio_value, trades_per_year, hp, evaluation):
        self.id = id
        self.symbol = symbol
        self.portfolio_value = portfolio_value
        self.trades_per_year = trades_per_year
        self.hp = hp
        self.evaluation = evaluation

def set_priority(priority='high'):
    pid = os.getpid()
    process = psutil.Process(pid)
    
    if priority == 'high':
        process.nice(psutil.HIGH_PRIORITY_CLASS)
    elif priority == 'above_normal':
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    elif priority == 'normal':
        process.nice(psutil.NORMAL_PRIORITY_CLASS)
    elif priority == 'below_normal':
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    elif priority == 'idle':
        process.nice(psutil.IDLE_PRIORITY_CLASS)
    else:
        raise ValueError("Invalid priority level")
    
def send_message(message):
    random = np.random.randint(0, 5)
    sleep(random)
    account_sid = 'AC2e60fdb4a5764900869b4709218d8dcd'
    auth_token = '4d198c3e1908cd3e25433915754875bc'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body=f'{message}',
    to='whatsapp:+972525269768'
    )   

def define_logger(name):
    # Set up logging for this year
    logger = logging.getLogger(f'logger_{name}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'logs/{name}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, file_handler

def close_logger(logger, file_handler):
    file_handler.close()
    logger.removeHandler(file_handler)
    logging.shutdown()

def hp_to_dict(hp):
    return {'z_score_window': hp[0], 'z_score_treshold': hp[1], 'atr_window': hp[2], 'gapNgo_tp_sl_ratio': hp[3], 'close_gap_tp_sl_ratio': hp[4]}

def optimization(df, symbol, hp, id, logger, export_graphs, export_stock_backtesting):
    balance = 1500
    slippage_factor = 30
    commission = 0.01 # TradeStation commission: 0.01$ per share or 1$ if less than 100 shares
    open_col = 'open'
    close_col = 'close'
    sl_rate = 100
    tp_rate = 100
    logger.info(f'\nOptimization parameters:\n\tBalance: {balance}\n\tSlippage factor: {slippage_factor}\n\tCommission: {commission}\n\tTakeProfit: {tp_rate}%\n\tStopLoss: {sl_rate}%')
    strategy = CatBoost_model(sl_rate=sl_rate, tp_rate=tp_rate)
    b_df = backtest(df,strategy, balance, slippage_factor, commission, close_col, open_col,hp)
    if export_stock_backtesting:
        b_df.to_csv(f'GRAPHS/{symbol}_backtesting_{id}.csv') #TODO - delete or uncomment
    if export_graphs:
        plot_candlestick(b_df, symbol, 1) #if problem -> clear pip cache by running 'pip cache purge' in terminal #TODO - uncomment

    trades_per_year = b_df.groupby(b_df['date'].dt.year).apply(lambda x: x[(x['position open order'] == StrategySignal.ENTER_LONG) | (x['position open order'] == StrategySignal.ENTER_SHORT)].shape[0])
    portfolio_value_stock_df = b_df[['date', 'portfolio_value']].sort_values('date')
    portfolio_value_stock_df['year'] = portfolio_value_stock_df['date'].dt.year
    evaluate_strategy_dict = evaluate_strategy(portfolio_value_stock_df, symbol, hp, id, trades_per_year, logger)
    portfolio_value_stock_df['symbol'] = symbol
    return stockBacktest(id, symbol, portfolio_value_stock_df[['date', 'symbol', 'portfolio_value']], trades_per_year, hp, evaluate_strategy_dict)

def processPoolExe(filename : str, hp : dict, id : int, in_sample_start : str, out_sample_start: str, export_graphs, export_stock_backtesting):
    set_priority(priority='high')
    symbol = f'{filename.split("_")[0]} - {id}'
    logger, file_handler = define_logger(name = symbol)
    try:
        path = f'..\\1- data\\wrds 1day Data\\pickels\\{filename}'
        in_or_out = 'in' if IN_SAMPLE else 'out'
        logger.info(f' ------------------------------- Starting {in_or_out} sample of stock {symbol} -------------------------------')
        symbol = filename.split('_')[0]
        df = pd.read_pickle(path)

        month_eralier = pd.to_datetime(in_sample_start) - pd.DateOffset(months=1)
        out_sample_start_dt = pd.to_datetime(out_sample_start)

        df = df[(df.date >= month_eralier) & (df.date < out_sample_start_dt)]
        df = df.sort_values('date')
        df = df.reset_index(drop=True)

        logger.info(f'\nIn sample dates: {in_sample_start} - {out_sample_start}\n\tFirst day: {df.date.min()}\n\tLast day: {df.date.max()}')

        if df.empty:
            logger.error(f'Empty dataframe for stock {symbol}')
            logger.info(f' ------------------------------- Ending {in_or_out} sample of stock {symbol} -------------------------------')
            print(f'Empty dataframe for stock {symbol}')
            close_logger(logger, file_handler)
            return None
        
        # check if there is enough data for the stock
        in_sample_min_date_threshold = pd.to_datetime('2014-01-01')
        out_sample_min_date_threshold = pd.to_datetime('2020-01-01')
        if (IN_SAMPLE and df.date.min() > in_sample_min_date_threshold) or (OUT_SAMPLE and df.date.min() > out_sample_min_date_threshold):
            logger.error(f'Not enough data for stock {symbol}')
            print(f'Not enough data for stock {symbol}')
            logger.info(f' ------------------------------- Ending {in_or_out} sample of stock {symbol} -------------------------------')
            close_logger(logger, file_handler)
            return None

        stock_res = optimization(df, symbol, hp, id, logger, export_graphs, export_stock_backtesting)
        logger.info(f' ------------------------------- Ending {in_or_out} sample of stock {symbol} -------------------------------')
        close_logger(logger, file_handler)
        return stock_res
    except Exception as e:
        logger.error(f'Error in stock {symbol}: {e}')
        print(f'Error in stock {symbol}: {e}')
        logger.info(f' ------------------------------- Ending {in_or_out} sample of stock {symbol} -------------------------------')
        close_logger(logger, file_handler)
        return None

# ------------------------------------------------------------------------------------------------------------------ MAIN --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('GRAPHS', exist_ok=True)
    os.makedirs('optimization results', exist_ok=True)
    os.makedirs('gaps export', exist_ok=True)
    os.makedirs('gaps export/csv', exist_ok=True)

    # --- Read the optimization stocks and hyperparameters
    backtest_stocks = pd.read_excel('backtest_stocks.xlsx')
    input_df = pd.read_excel('input.xlsx')
    (start, end, step) = (int(input_df['start'].values[0]), int(input_df['end'].values[0]), int(input_df['step'].values[0]))
    z_score_window = [i for i in range(start, end+step, step)]
    (start, end, step) = (float(input_df['start'].values[1]), float(input_df['end'].values[1]), float(input_df['step'].values[1]))
    z_score_treshold = np.arange(start, end + step, step).tolist()
    (start, end, step) = (int(input_df['start'].values[2]), int(input_df['end'].values[2]), int(input_df['step'].values[2]))
    atr_window = [i for i in range(start, end+step, step)]
    (start, end, step) = (float(input_df['start'].values[3]), float(input_df['end'].values[3]), float(input_df['step'].values[3]))
    gap_n_go_sl_tp_ratio = np.arange(start, end + step, step).tolist()
    (start, end, step) = (float(input_df['start'].values[4]), float(input_df['end'].values[4]), float(input_df['step'].values[4]))
    close_gap_sl_tp_ratio = np.arange(start, end + step, step).tolist()
    


    # calc all combinations and check if opt can proceed + export graphs
    combinations = list(product(z_score_window, z_score_treshold, atr_window, gap_n_go_sl_tp_ratio, close_gap_sl_tp_ratio))
    files = [f for f in os.listdir(f'..\\1- data\\wrds 1day Data\\pickels') if f.endswith('fixed.pickle') and f.split("_")[0] in backtest_stocks['backtesting stocks'].values]
    print(f'Zscore window range: {z_score_window}')
    print(f'Zscore treshold range: {z_score_treshold}')
    print(f'ATR window range: {atr_window}')
    print(f'Gap&Go Stoploss/Takeprofit ratio range: {gap_n_go_sl_tp_ratio}')
    print(f'Close-Gap Stoploss/Takeprofit ratio range: {close_gap_sl_tp_ratio}')
    print(f'Number of hyperparameters combinations: {len(combinations)}')
    print(f'Number of stocks: {len(files)}')
    print(f'Number of total combinations: {len(combinations)*len(files)}')
    print('Proceed? (y/n)')
    print('Pay attention that IN_SAMPLE is set' if IN_SAMPLE else 'Pay attention that OUT_SAMPLE is set')
    if input() != 'y':
        print('Exiting...')
        exit()
    print('Export graphs and stock backtesting tables? (y/n)')
    export_graphs = False
    export_stock_backtesting = False
    if input() == 'y':
        export_graphs = True
        export_stock_backtesting = True
        print('Exporting graphs and stock backtesting tables...')

    in_sample_start = '2012-01-01'
    out_sample_start = '2020-01-01'

    out_sample_end = '2024-01-01'


    evaluate_hp_lst = []
    with ProcessPoolExecutor() as executor:
        for id, hp in enumerate(combinations, start=1):
            opt_evals = []
            portfolio_values = []
            trades_per_year = []
            hp_dict = hp_to_dict(hp)
            if IN_SAMPLE:
                futures = {executor.submit(processPoolExe, filename , hp_dict, id, in_sample_start, out_sample_start, export_graphs, export_stock_backtesting): filename for filename in files}
            if OUT_SAMPLE:
                futures = {executor.submit(processPoolExe, filename , hp_dict, id, out_sample_start, out_sample_end, export_graphs, export_stock_backtesting): filename for filename in files}
            num_tasks = len(combinations)
            print(f'optimization {id}/{num_tasks} -> for {len(files)} stocks...')
            counter = 1
            for future in as_completed(futures):
                print(f'stock {counter}/{len(files)} in optimization {id}/{num_tasks}')
                result = future.result()
                if result is not None:
                    portfolio_values.append(result.portfolio_value)
                    trades_per_year.append(result.trades_per_year)
                    opt_evals.append(result.evaluation)
                    counter += 1

            opt_trades_per_year = pd.concat(trades_per_year)
            opt_trades_per_year = opt_trades_per_year.groupby(opt_trades_per_year.index).sum()

            opt_portfolio_value = pd.concat(portfolio_values).groupby('date').apply(lambda x: pd.Series({'sum of portfolio_value': 0})).reset_index()
            for port_val in portfolio_values:
                symbol = port_val['symbol'].values[0]
                opt_portfolio_value = opt_portfolio_value.merge(port_val[['date', 'portfolio_value']], on='date', how='left').rename(columns={'portfolio_value': symbol})
            opt_portfolio_value.fillna(1500, inplace=True)
            opt_portfolio_value['sum of portfolio_value'] = opt_portfolio_value.iloc[:, 2:].sum(axis=1)
            opt_portfolio_value.rename(columns={'sum of portfolio_value': 'portfolio_value'}, inplace=True)
            opt_portfolio_value.to_excel(f'optimization results/portfolio_value_{id}.xlsx', index=False)
            opt_portfolio_value['year'] = opt_portfolio_value['date'].dt.year

            evaluate_hp_lst.append(evaluate_strategy(opt_portfolio_value[['date','year','portfolio_value']], 'CatBoost Gap Strategy', hp_dict, id, opt_trades_per_year))

            opt_evals_df = pd.DataFrame(opt_evals)
            opt_evals_df = opt_evals_df.sort_values('total return', ascending=False)
            opt_evals_df.to_csv(f'optimization results/scores_per_stock_{id}.csv')

    print('All done')
    scores_df = pd.DataFrame(evaluate_hp_lst)
    scores_df = scores_df.sort_values('total return', ascending=False)
    scores_df.to_pickle('optimization results/scores_df.pickle')

    try:
        scores_df.to_excel('optimization results/scores_df.xlsx', index=False)
    except Exception as e:
        print(f'Error in saving excel file: {e}')

    try:
        # open each log file in logs folder and save it to a single file
        with open('optimization results/logs.txt', 'a') as log_file:
            for filename in os.listdir('logs'):
                if filename.endswith('.log'):
                    with open(f'logs/{filename}', 'r') as f:
                            log_file.write(f.read())
                    log_file.write('\n\n')
                    os.remove(f'logs/{filename}')
                    
    except Exception as e:
        print(f'Error in saving log file: {e}')
    
    try:
        send_message('Optimization done')
    except Exception as e:
        print(f'Error in sending message: {e}')
