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
FREQ = 'w'
#%% laod data
hs300_raw = pd.read_csv(os.path.join(DATA_DIR, "000300_index.csv"))
zz500_raw = pd.read_csv(os.path.join(DATA_DIR, "000905_index.csv"))

backtest_data_h5 = pd.HDFStore("h5_data/backtest_data.h5")
data = backtest_data_h5['backtest_data']
data = data.reset_index()
data = data.drop_duplicates(["ticker", "date"])
data = data.loc[(data.status ==1) & (data.is_st == 0)]
adj_vwap_table = pd.pivot(data, 'date', 'ticker', 'vwap_pctChange')
factor_data_h5 = pd.HDFStore("h5_data/factor_data.h5")
apb_5d_ma30 = factor_data_h5['apb_5d_ma30']
apb_5d_ma30 = apb_5d_ma30.reset_index()
apb_5d_ma30_table = pd.pivot(apb_5d_ma30, 'date', 'ticker', 'apb')
trade_date = apb_5d_ma30_table.index.get_level_values(0)
apb_5d_ma30_table = apb_5d_ma30_table.reindex_like(adj_vwap_table)
#%% set change hand freq
changehand_date_m = pd.date_range(trade_date[0], trade_date[-2], freq="BM")
changehand_date_w = pd.date_range(trade_date[25], trade_date[-2], freq="W-MON")
CHANGE_HAND_DATE = changehand_date_w if FREQ == "w" else changehand_date_m

#%% back test
stock_pool = {}
portfolio_return = pd.DataFrame()
all_return = pd.Series()

for period_start,  period_end in zip(CHANGE_HAND_DATE[1:], CHANGE_HAND_DATE[2:]):
    real_chd = apb_5d_ma30_table.index.get_indexer([period_start, period_end],'ffill')
    chd_factor = apb_5d_ma30_table.iloc[real_chd[0]].dropna()
    picked_stocks = chd_factor.sort_values(ascending=False)[:len(chd_factor)//10].index.to_list()
    stock_pool.update({
        apb_5d_ma30_table.index[real_chd[0]]:picked_stocks
    })
    # 收盤後計算因子
    # 隔天 vwap 買入，所以可以得到 t+1 到 t+2 vwap pctchage 的收益
    period_portfolio_return = adj_vwap_table.iloc[real_chd[0]+1:real_chd[1]+1, :][picked_stocks]

    #等權買進
    period_return = period_portfolio_return.mean(axis=1)
    all_return = all_return.append(period_return)

#%% get all returns 
hs300 = hs300_raw.set_index('date')
hs300.index = pd.to_datetime(hs300.index, format="%Y%m%d")
hs300['pctChange'] = hs300['close'].pct_change().shift(-1)
hs_300_return = hs300.loc[all_return.index, 'pctChange'].rename("hs300")

zz500 = zz500_raw.set_index('date')
zz500.index = pd.to_datetime(zz500.index, format="%Y%m%d")
zz500['pctChange'] = zz500['close'].pct_change().shift(-1)
zz500_return = zz500.loc[all_return.index, 'pctChange'].rename("zz500")
alpha_hs300_return = all_return-hs_300_return.reindex_like(all_return)
alpha_zz500_return = all_return-zz500_return.reindex_like(all_return)

f_return = pd.DataFrame({
    "long_side_return": all_return,
    'hs300':hs_300_return.reindex_like(all_return),
    'zz500':alpha_zz500_return.reindex_like(all_return),
    "alpha_hs300_return":alpha_hs300_return,
    "alpha_zz500_return":alpha_zz500_return
})
#%% plot return
(all_return+1).rename("abp").cumprod().plot()
(hs_300_return+1).cumprod().plot()
(zz500_return+1).cumprod().plot()
plt.legend()
#%% analyze the return
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


pd.Series(stock_pool).to_csv("stock_pool/stockpool_{}.csv".format(FREQ))
analyze_risk(f_return).T

# %%
