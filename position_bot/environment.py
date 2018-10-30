import numpy as np
import pandas as pd

class SingleStockMarket:
    def __init__(self, df:pd.DataFrame = None, start_idx:int = 0, seq_len = 60, 
                 commission = 1e-3):
        #constants
        self.start_idx = start_idx + seq_len
        self.seq_len = seq_len
        self.action_space = [-1,0,1]
        self.n_actions = len(self.action_space)
        self.df = df.copy()
        self.commission = commission
        
        #reset semi-constants
        self.reset()
        
    def reset(self):
        self.idx =  self.start_idx
        self.open_idx = self.start_idx - self.seq_len
        #set all positions to zero except for first seq_len to be all hold
        self.df['position'] = 0
        self.df.iloc[:self.idx,self.df.columns.get_loc('position')] = 1
        self.current_state = self.get_state()
        return(self.current_state)
        
    def step(self, action):
        action = int(action)
        state = self.get_state()
        old_sharpe = self.get_sharpe()
        #position
        self.df.iloc[self.idx,self.df.columns.get_loc('position')] = self.action_space[action]
        #if new direction; record commission for closing and opening a position
        if (self.action_space[action] != self.df.iloc[self.idx-1, self.df.columns.get_loc('position')]):
            mult = (self.df.iloc[self.open_idx:self.idx , self.df.columns.get_loc('vwap_returns')]+1).prod()
            #for closing a position
            self.df.iloc[self.idx,self.df.columns.get_loc('commission')] = self.commission * mult 
            #for opening a new one; 0 for hold
            self.df.iloc[self.idx,self.df.columns.get_loc('commission')] += self.commission if self.action_space[action]!=0 else 0
            self.open_idx = self.idx
        
        #go to next timestamp
        self.idx+=1
        
        next_state = self.get_state()
        #differential sharpe as reward
        #absolute
#         reward = self.get_sharpe() - old_sharpe
        #percentage
        reward = self.get_sharpe() / old_sharpe - 1 
        reward = np.nan_to_num(reward)
        reward = np.clip(reward,-1,1)
        done = True if self.idx==self.df.shape[0] else False
        info = f'Currently at index {self.idx}'
        
        return(next_state,reward,done,info)
    
    def get_state(self):
        #current state at timestamp is everything BEFORE current prices (up to sequence length)
        from_idx = self.idx - self.seq_len
        #see as much as the timestamp before
        to_idx = self.idx
        #get open,high,low,close,vwap
        state = np.array(self.df.iloc[from_idx:to_idx,1:6])
        #standardize to be N(0,1)
        state_norm = (state - state.mean(axis=0)) / (state.std(axis=0))
        positions = self.df.iloc[from_idx:to_idx,self.df.columns.get_loc('position')][:,None]
        state = np.concatenate([state_norm,positions],axis=1).transpose()[None,:,:]
        return(state)
    
    def get_sharpe(self, rfr = 0.02 / 525600):
        returns = self.get_returns(rfr)
        s = np.nanmean(returns) / (np.nanstd(returns) * np.sqrt(self.idx))
        s = np.nan_to_num(s)
        s = np.clip(s,-1,1)
        return(s)
    
    def get_returns(self,rfr = 0.02 / 525600):
        benchmark_returns = self.df['vwap_returns'][self.start_idx:self.idx] - self.df['commission'][self.start_idx:self.idx] - rfr
        current_position = self.df['position'][self.start_idx:self.idx]
        returns = current_position * benchmark_returns
        return(returns)
        