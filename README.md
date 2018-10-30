# RL Trader
A collection of RL financial applications

## Position Bot

This agent is created to demonstrate how Q-learning can be applied to a real-world problem. Retail trading is chosed as it has a lot of resemblances with many real-world problem such as ads bidding. We created the environment based on [minute-level bitcoin price data from Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data/version/14). We use the original Deep Q-learning implementation as stated in the paper [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). For more advanced implementations as well as other methods, check out our bi-weekly [Bangkok School of AI Reinforcement Learning Workshop](https://github.com/Datatouille/rl-workshop/).

**Disclaimer** You will not get rich with this algorithm. We do not hold any Bitcoin position as of 2018-10-31.

## Environment

The environment is based on Bitcoin price from 2016-08-01 13:21:00 to 2016-10-10 00:00:00. It consists of 6 * 60 values for a state and 3 available actions. The state represents batch-normalized 60-minute previous ohlc, vwap and position. The actions are enter short position, do nothing, enter long position respectively. In our example notebook, we run the agent for 10,000 timesteps. The reward is differential sharpe ratio.

```
Actions look like: 
* 0 - short
* 1 - nothing
* 2 - long
Action size: 3
States look like: (1, 6, 60)
```

## Getting Started

0. Install dependencies.

```
pip install -r requirements.txt
```

2. Follow `position_sandbox.ipynb` to train the agent.

4. Our implementation is divided as follows:
* `replay_memory.py` - Experience Replay Memory
* `agent.py` - Agent
* `qnetwork` - Q-networks for local and target

## Train Agent

These are the steps you can take to train the agent with default settings.

0. Initiate environment

```
env = SingleStockMarket(bitstamp_df)
```

1. Create a experience replay memory.

```
mem = ReplayMemory(10000)
```

2. Create an agent.

```
a = VanillaQAgent(replay_memory = mem)
```

3. Train the agent.

```
state = env.reset()
for i in trange(10000):
    #select action
    action = a.act(state,i)  

    #step
    next_state,reward,done,info = env.step(action)                
    a.step(state,action,reward,next_state,done)

    state = next_state                          
```

## Imitation Bot

Work in progress

## Shortfall Bot

Work in progress