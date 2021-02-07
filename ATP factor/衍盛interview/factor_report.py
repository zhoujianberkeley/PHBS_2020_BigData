# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:02:03 2021

@author: eiahb
"""
#%% import 
import os
import alphalens
import warnings

import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime  import date
from time import time
from logger import Logger

os.environ['NUMEXPR_MAX_THREADS'] = '16'
pd.set_option('display.max_r',20)
logger = Logger("apb")
plt.style.use('seaborn')
warnings.filterwarnings("ignore")
#%% parameter
ROOT = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(ROOT, "data")
H5_DATA_DIR = os.path.join(ROOT, "h5_data")

START_DATE = '2012-01-01'
END_DATE = None
WIN_SIZE = 30
#%% load data
# load data from the csv in the first time and store them in h5 file
# always use the h5 data in the following work

logger.info("start loading factor")
calendar_raw = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"), names = ['trade_date'])
st_stock_raw = pd.read_csv(os.path.join(DATA_DIR, "st_stock.csv"))
summary = pd.read_csv(os.path.join(DATA_DIR, "summary.csv"))
hs300 = pd.read_csv(os.path.join(DATA_DIR, "000300_index.csv"))
zz500 = pd.read_csv(os.path.join(DATA_DIR, "000905_index.csv"))

try:
    stock_data_h5 = pd.HDFStore("h5_data/stock_data.h5")
    factor_data_h5 = pd.HDFStore("h5_data/factor_data.h5")
    all_data = stock_data_h5['stock_data']
    apb_5d = factor_data_h5['apb_5d']
    apb_30d = factor_data_h5['apb_30d']
except Exception as e:
    print(e.message)
    print("please run factor_cal first")

#%% st 
st_stock = st_stock_raw.rename(columns={"symbol":"ticker"})
st_stock = st_stock.set_index("ticker")
st_stock = st_stock.fillna(date.today().strftime("%Y%m%d"))
st_stock.exit_date = pd.to_datetime(st_stock.exit_date.astype(int).astype(str), format="%Y%m%d")
st_stock.entry_date = pd.to_datetime(st_stock.entry_date, format="%Y%m%d")

all_st = pd.DataFrame()
for i in st_stock.itertuples():
    st_date = pd.date_range(i.entry_date, i.exit_date)
    df = pd.DataFrame({'date':st_date,"ticker":i.Index})
    all_st = all_st.append(df)
is_st = pd.Series(data=1, index=pd.MultiIndex.from_frame(all_st), name="is_st")


#%% prepare data for stock return 
logger.info("start loading data")
data = all_data.loc[:, ["ticker", "date", "close", "vwap", "adj_factor", "status"]]
data.date = pd.to_datetime(data.date, format = "%Y%m%d")
data = data.set_index(["date", "ticker"])
data['adj_close'] = data['close'] *data['adj_factor']
data['adj_vwap'] = data['vwap'] *data['adj_factor']
data['close_pctChange'] = data['adj_close'].groupby(level = "ticker").pct_change().shift(-1)
data['vwap_pctChange'] = data['adj_vwap'].groupby(level = "ticker").pct_change().shift(-1)
data = pd.merge(data,is_st,'left',left_index=True,right_index=True)
data.is_st = data.is_st.fillna(value=0)
#%% last treatment after getting raw factor data
# including ln and rolling mean

def last_touch(daily_abp, win_size = 30, start_date = START_DATE, end_date = END_DATE):
    log_apb = np.log(daily_abp['apb'])
    apb_ma30 = log_apb.groupby(level='ticker').transform(
        lambda x: x.rolling(win_size).mean())

    factor = apb_ma30.swaplevel().sort_index()
    factor.index.names = ['date', 'ticker']
    factor = factor.reset_index()
    factor.date = pd.to_datetime(factor.date, format = "%Y%m%d")
    factor = factor.set_index(['date'])
    factor = factor.loc[start_date:end_date]
    factor = factor.set_index(keys = "ticker", append = True)
    return(factor)

# # save factor for latter use
# apb_5d_ma30 = last_touch(apb_5d)
# apb_30d_ma30 = last_touch(apb_30d)
# factor_data_h5['apb_5d_ma30'] = apb_5d_ma30
# factor_data_h5['apb_30d_ma30'] = apb_30d_ma30
# backtest_data_h5 = pd.HDFStore("h5_data/backtest_data.h5")
# backtest_data_h5['backtest_data'] = data
# stock_data_h5.close()
# factor_data_h5.close()
# backtest_data_h5.close()

## merge factor with stock return
# apb_5d_ma30_df = pd.merge(left=apb_5d_ma30, right=data.loc[data.status == 1, 'vwap_pctChange'].rename('pctChange'), left_index=True,
#                             right_index=True, how='left').dropna()
# apb_30d_ma30_df = pd.merge(left=apb_30d_ma30, right=data.loc[data.status == 1, 'vwap_pctChange'].rename('pctChange'), left_index=True,
#                             right_index=True, how='left').dropna()
#%%  t value stat result

def get_t_value(factor, returns):
    x = sm.add_constant(factor)
    mod = sm.OLS(returns, x)
    res = mod.fit()
    return res.tvalues[0]

def get_t_report(factor_df: pd.DataFrame):

    rept = {}
    tvalue_ser = factor_df.groupby(level='date').apply(
        lambda x: get_t_value(x["apb"], x["pctChange"]))
    abs_tvalue_ser = abs(tvalue_ser)

    rept['t均值'] = tvalue_ser.mean()
    rept['|t|均值'] = abs_tvalue_ser.mean()
    rept['|t|>2占比'] = np.sum(
        np.where(abs_tvalue_ser > 2, 1, 0)) / len(abs_tvalue_ser)
    return rept

# print(get_t_report(apb_5d_ma30_df))
# print(get_t_report(apb_30d_ma30_df))

#%% ic stat result

def get_ic_report(factor_df: pd.DataFrame):
    rept = {}
    rank_ic = factor_df.groupby(level='date').apply(
        lambda x: st.pearsonr(x["apb"].rank(), x["pctChange"].rank())[0])
    
    # ic = factor_df.groupby(level='date').apply(
    #     lambda x: st.pearsonr(x["apb"], x["pctChange"])[0])

    rept['rank ic均值'] = rank_ic.mean()
    rept['rank ic标准差'] = rank_ic.std()
    rept['rank IC_IR'] = rank_ic.mean()/rank_ic.std()
    rept['rank IC大于0的比率'] = len(rank_ic[rank_ic > 0])/len(rank_ic)
    return rept

# print(get_ic_report(apb_5d_ma30_df))
# print(get_ic_report(apb_30d_ma30_df))

#%% factor return stat result
def set_group(tmp, group_num = 10):
    tmp = tmp.sort_values("apb", ascending=True)
    tmp['group'] = np.arange(len(tmp))//((len(tmp)//group_num)+1) + 1 
    tmp.index = tmp.index.droplevel('date')
    return(tmp)

def get_group_mean_return(tmp):
    return(tmp['pctChange'].groupby(level="date").mean())

def get_factor_return(factor_df: pd.DataFrame):
    factor_df = factor_df.groupby(level = "date").apply(set_group)
    factor_returns = factor_df.groupby("group").apply(get_group_mean_return).T
    factor_returns['long/short'] = factor_returns[10] - factor_returns[1]
    return(factor_returns)

# apb_5d_ma30_factorReturn = get_factor_return(apb_5d_ma30_df)
# apb_30d_ma30_df_factorReturn = get_factor_return(apb_30d_ma30_df)
# print(apb_5d_ma30_factorReturn)
# print(apb_30d_ma30_df_factorReturn)
#%% factor return report 
def get_max_drawdown(df:pd.DataFrame):
    md = {}
    for label,col in df.items():
        # 最大回撤
        max_nv = np.maximum.accumulate(col)
        mdd = -np.min(col/max_nv-1)
        md[label] = mdd
    return pd.Series(md)

def analyze_risk(f_returns:pd.DataFrame):
    f_cum = (1+f_returns).cumprod()
    f_cum.plot(legend = True)

    # 计算累计收益率
    cum_ret_rate = f_cum.iloc[-1] - 1
    # 计算年化收益率
    annual_ret = (f_cum.iloc[-1] ) ** (250. / float(len(f_cum))) - 1
    mdd = get_max_drawdown(f_cum)                      
    volatility = np.std(f_returns)*np.sqrt(250)
    sharpe = (annual_ret-0.04) / volatility
    df = pd.concat([annual_ret,cum_ret_rate,mdd,sharpe,volatility],axis=1)
    df.columns = ['年化收益','累计收益','最大回撤','夏普','波动率']
    return df

# print(analyze_risk(apb_5d_ma30_factorReturn))
# print(analyze_risk(apb_30d_ma30_df_factorReturn))
# %% def main
def main():
    # preprocess
    apb_5d_ma30 = last_touch(apb_5d, win_size = WIN_SIZE)
    apb_30d_ma30 = last_touch(apb_30d, win_size = WIN_SIZE)

    # merge factor with stock return
    apb_5d_ma30_df = pd.merge(left=apb_5d_ma30, right=data.loc[(data.status == 1) & (data.is_st == 0), 'close_pctChange'].rename('pctChange'), left_index=True,
                            right_index=True, how='left').dropna()
    apb_30d_ma30_df = pd.merge(left=apb_30d_ma30, right=data.loc[(data.status == 1) & (data.is_st == 0), 'close_pctChange'].rename('pctChange'), left_index=True,
                            right_index=True, how='left').dropna()
    
    # t report
    logger.debug("start analyze t report")
    print("t report")
    print("apb_5d")
    print(get_t_report(apb_5d_ma30_df))
    print("\napb_30d")
    print(get_t_report(apb_30d_ma30_df))
    print("t report done")
    # ic report 
    logger.debug("start analyze ic report")
    print("\nic report")
    print("\napb_5d")
    print(get_ic_report(apb_5d_ma30_df))
    print("\napb_30d")
    print(get_ic_report(apb_30d_ma30_df))
    print("ic report done")

    # factor return
    logger.debug("start cal factor return")
    print("\nfactor return")
    apb_5d_ma30_factorReturn = get_factor_return(apb_5d_ma30_df)
    apb_30d_ma30_df_factorReturn = get_factor_return(apb_30d_ma30_df)
    print("\napb_5d")
    print(apb_5d_ma30_factorReturn)
    print("\napb_30d")
    print(apb_30d_ma30_df_factorReturn)
    print("factor return done")

    # factor return report 
    logger.debug("start analyze factor return")
    print("\nfactor return report ")
    print("\napb_5d")
    print(analyze_risk(apb_5d_ma30_factorReturn))
    print("\napb_30d")
    print(analyze_risk(apb_30d_ma30_df_factorReturn))
    print("factor report done")


# main
if __name__ == '__main__':
    main()


# %%
