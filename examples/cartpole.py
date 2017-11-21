import numpy as np
import gym
from txtgen.models.deep_q_network import DeepQNetwork

env = gym.make('CartPole-v0')

if __name__ == '__main__':
    agent = DeepQNetwork(actions=2, state_dimension=4)

    for i in range(5000):
        reward_sum = 0.0
        observation = env.reset()
        agent.set_initial_state(observation=observation)
        while True:
            action = agent.get_action()
            action_id = np.argmax(action)

            next_observation, reward, is_terminal, info = env.step(action=action_id)
            agent.perceive(next_observation=next_observation, action=action, reward=reward, is_terminal=is_terminal)

            reward_sum += reward
            if is_terminal:
                break
        print 'episode {round_id}: {reward}'.format(round_id=i, reward=reward_sum)