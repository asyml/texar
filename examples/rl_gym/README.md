# Reinforcement Learning for Games #

This example implements three RL algorithms for the Cartpole game based on the OpenAI Gym environment:
* [pg_cartpole.py](./pg_cartpole.py) uses Policy Gradient
* [dqn_cartpole.py](./dqn_cartpole.py) uses Deep-Q
* [ac_cartpole.py](./ac_cartpole.py) uses Actor-critic

The example is for demonstrating the Texar RL APIs (for games), and only implements the most basic versions of respective algorithms.

## Usage ##

Run the following cmd to start training:

```
python pg_cartpole.py --config config 
python dqn_cartpole.py --config config 
python ac_cartpole.py --config config 
```
