import numpy as np
import pandas as pd
from position_bot.environment import SingleStockMarket

def model_eval(env):
    return(env.get_sharpe(), (env.df.iloc[:env.idx]['vwap_returns'] * env.df.iloc[env.start_idx:env.idx]['position'] +1).prod())

def buy_eval(env):
    env_buy = SingleStockMarket(env.df)
    env_buy.idx = env.idx
    env_buy.df['position'] = 1
    return(env_buy.get_sharpe(), (env_buy.df.iloc[env_buy.start_idx:env_buy.idx]['vwap_returns']+1).prod())

def momentum_eval(env,momentum=10):
    env_momentum = SingleStockMarket(env.df)
    env_momentum.idx = env.idx 
    env_momentum.df['position'] = (env_momentum.df['vwap'].rolling(momentum).mean() < env_momentum.df['vwap']).astype(int)
    env_momentum.df['position'] = env_momentum.df['position'].apply(lambda x: -1 if x==0 else 1)
    return(env_momentum.get_sharpe(), 
           (env_momentum.df.iloc[env_momentum.start_idx:env_momentum.idx]['position'] *
            env_momentum.df.iloc[env_momentum.start_idx:env_momentum.idx]['vwap_returns']+1).prod())