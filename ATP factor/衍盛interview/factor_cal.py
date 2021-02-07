# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:02:03 2021

@author: eiahb
"""
#%% import 
import os
import sys  

import pandas as pd
import numpy as np
import ray

from tqdm import tqdm
from time import time
from multiprocessing import Pool, cpu_count 
from logger import Logger

# reload(sys)  
# sys.setdefaultencoding('utf8')

os.environ['NUMEXPR_MAX_THREADS'] = '16'
tqdm.pandas()
pd.set_option('display.max_r',10)
logger = Logger("apb")

#%% parameter
ROOT = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(ROOT, "data")
TEST_MODE = False


#%% load data
# load data from the csv in the first time and store them in h5 file
# always use the h5 data in the following work
calendar_raw = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"), names = ['calandar'])
st_stock = pd.read_csv(os.path.join(DATA_DIR, "st_stock.csv"))
summary = pd.read_csv(os.path.join(DATA_DIR, "summary.csv"))
hs300 = pd.read_csv(os.path.join(DATA_DIR, "000300_index.csv"))
zz500 = pd.read_csv(os.path.join(DATA_DIR, "000905_index.csv"))

try:
    stock_data_h5 = pd.HDFStore("h5_data/stock_data.h5")
    all_data = stock_data_h5['stock_data']
except :
    all_data = pd.DataFrame()
    for f in os.listdir(DATA_DIR):
        if f not in ["calendar.csv", "st_stock.csv", "summary.csv", "000300_index.csv", "000905_index.csv"]:
            stock_table = pd.read_csv(os.path.join(DATA_DIR, f))
            stock_table.insert(0, "ticker", f[:9])
            print(stock_table)
            all_data = pd.concat([all_data, stock_table], sort=False)      
        
    stock_data_h5 = pd.HDFStore("h5_data/stock_data.h5")
    stock_data_h5['stock_data'] = all_data
    stock_data_h5.close()
#%% adj vwap

# calculate the adj vwap with adj factor
if TEST_MODE:
    select_tickers = all_data.ticker.unique()[:30]
    data = all_data.loc[all_data.ticker.isin(select_tickers) ,['ticker', 'date','vwap','adj_factor', 'volume',  'status']]
else:
    data = all_data.loc[: ,['ticker', 'date','vwap','adj_factor', 'volume',  'status']]
data = data.set_index(['ticker', 'date'])
adj_vwap = data.loc[:, 'vwap'] * data.loc[:, 'adj_factor']
data['adj_vwap'] = adj_vwap


#%%
def cal_APB(df: pd.DataFrame, threshold: int) -> pd.Series:
    '''
    calculate the apb factor for a certain day
    which takes n days (the length of df) to calculate
    and the threshold filters days that does not have enough non nan data 

    Parameters
    ----------
    df : pd.DataFrame
        the data in the look back window
    threshold : int
        the least day count needed in a look back window

    Returns
    -------
    pd.DataFrame
        the apb with the index of related date

    '''

    idx = df.index.get_level_values(1)[-1]
    trade_day = df['status'].sum()
    # 过滤小于threshold数量的股票
    if trade_day > threshold:
        return pd.DataFrame(
                    {
                        'apb':
                            np.mean(df['adj_vwap']) /
                            np.average(df['adj_vwap'], weights=df['volume'])
                    },
                    index=[idx]
                )
    else:
        return pd.DataFrame({'apb': np.nan}, index=[idx])

def rolling_apply(df, func, win_size) -> pd.Series:
    '''
    use the np.stride trick to perform rolling technic
    the np.lib.stride_tricks.as_strided returns a 3 dim np array 
    and apply func to each rolling block through the first dim

    Parameters
    ----------
    df : pd.DataFrame
        
    func : callable
        
    win_size : int
        

    Returns
    -------
    pd.DataFrame
          return the res of rolling result of the func with the given df

    '''
    if len(df)<win_size:
        return None
    
    iidx = np.arange(len(df))
    shape = (iidx.size - win_size + 1, win_size)
    strides = (iidx.strides[0], iidx.strides[0])
    strided_array = np.lib.stride_tricks.as_strided(
        iidx, shape=shape, strides=strides, writeable=True)
    res = pd.concat((func(df.iloc[r]) for r in strided_array), axis=0)
    return res



    
@ray.remote    
def rolling_5apb(xdf):
    '''
    cal the rolling apb with look back window of 5 days\n
    Parameters
    ----------
    xdf : pd.DataFrame
        input data with multi-index:(ticker, trade_date)

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    '''
    tqdm.pandas()
    res = xdf.groupby(level='ticker').progress_apply(
        lambda x: rolling_apply(x, lambda x_df: cal_APB(x_df, 3), 5))
    
    return res

@ray.remote 
def rolling_30apb(xdf):
    '''
    cal the rolling apb with look back window of 5 days\n
    Parameters
    ----------
    xdf : pd.DataFrame
        input data with multi-index:(ticker, trade_date)

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    '''
    tqdm.pandas()
    return xdf.groupby(level='ticker').progress_apply(
        lambda x: rolling_apply(x, lambda x_df: cal_APB(x_df, 15), 30))

#%% ways to apply multiprocessing 
# ray is faster because it does not need to pickle the data
# however it seems to require python 3.5 or more 
# therefore I provide two ways to perform apply with multiprocessing

def ray_apply(df, func, workers = None):
    if workers == None:
        workers = cpu_count()
    ray.init(num_cpus = workers, ignore_reinit_error=True)
    codes = df.index.get_level_values('ticker').drop_duplicates()
    mapping = pd.Series(np.arange(0, len(codes)), index=codes)
    workerid = df.index.get_level_values('ticker').map(mapping) % (workers)
    groupeds = df.groupby(workerid.values)
    res_list = ray.get([func.remote(g) for _,g in list(groupeds)])
    ray.shutdown()
    return pd.concat(res_list)

# the performance of original multiprocessing is awful
def multiprocessing_apply(df, func, workers = None):
    if workers == None:
        workers = cpu_count()
    codes = df.index.get_level_values('ticker').drop_duplicates()
    mapping = pd.Series(np.arange(0, len(codes)), index=codes)
    res = df.index.get_level_values('ticker').map(mapping) % (workers* 10)
    groupeds = df.groupby(res.values)
    with Pool(workers) as p:
        res_list = p.map(func, [g for _, g in groupeds])
    return res_list

#%% def main 
def main():
    # run rolling with ray apply
    # 708.7584984302521 s
    tic = time()
    res5 = ray_apply(data, rolling_5apb)
    toc = time()
    print(toc - tic)

    factor_data = pd.HDFStore("h5_data/factor_data.h5")
    factor_data['apb_5d'] = res5
    factor_data.close()
    #%%
    # run rolling_30apb with ray apply
    # 699.8334312438965 s
    tic = time()
    res30 = ray_apply(data, rolling_30apb)
    toc = time()
    print(toc - tic)

    factor_data = pd.HDFStore("h5_data/factor_data.h5")
    factor_data['apb_30d'] = res30
    factor_data.close()

#%% main
if __name__ == "__main__":
    main()