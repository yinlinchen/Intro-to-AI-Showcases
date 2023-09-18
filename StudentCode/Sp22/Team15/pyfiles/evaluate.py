from collections import deque
import numpy as np

def evaluate(env, agent, num_episodes=20000, window=100):
    avg_rewards = deque(maxlen=num_episodes)
    best_avg_reward = float('-inf')
    samp_rewards = deque(maxlen=window)
    
    for i_episode in range(1, num_episodes+1):
        state = env.reset()
        samp_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            samp_reward += reward
            state = next_state
            if done:
                samp_rewards.append(samp_reward)
                break
        
        if (i_episode >= 100):
            avg_reward = np.mean(samp_rewards)
            avg_rewards.append(avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward
