import matplotlib.pyplot as plt
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import Adam
from keras import backend as K
import random
import time
import pickle

np.set_printoptions(suppress=True)

class DQNAgent:
  def __init__(self, state_space, action_space):
    self.state_space = state_space
    self.action_space = action_space
    self.alpha = 0.001
    self.gamma = 0.98
    self.epsilon = 0.7
    self.min_eps= 0.01
    self.epsilon_decay= 0.995
    self.storage = []
    self.ep_rewards = []
    self.loss = []
    self.model = self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Dense(400, input_dim=self.state_space, activation='relu'))
    model.add(Dense(300,  activation='relu'))
    model.add(Dense(self.action_space, activation='linear'))
    model.compile(loss='mse',optimizer=Adam(lr=self.alpha))
    return model

  def sarsa(self, curr_state, action, reward, next_state, done ):
    self.storage.append((curr_state, action, reward, next_state, done))
    self.ep_rewards.append(reward)

  def epsilon_policy(self,curr_state):
    if np.random.rand() <= self.epsilon :
      action = np.random.uniform(-1,1,4)
      return action
    act_values = self.model.predict(curr_state)
    #print(act_values)
    action = act_values[0]
    return action

  def batch_training(self, batch):
    minibatch = random.sample(self.storage, batch)
    self.loss = []
    for curr_state, act, reward, next_state, done in minibatch:
      Q_update = reward
      if not done:
        Q_update = ( (1.0-0.1)*reward + 0.1 * (self.gamma*np.amax(self.model.predict(next_state)[0])))
      Q_change = self.model.predict(curr_state)
      Q_change[0] = Q_update
      history = self.model.fit(curr_state, Q_change,verbose=0,epochs=1)
      self.loss.append(history.history['loss'])
      self.ep_rewards = []

    mean_loss = np.mean(self.loss)
    if self.epsilon >= self.min_eps:
      self.epsilon *= self.epsilon_decay

    return history, mean_loss

  def save(self,name):
    self.model.save(name)

def truncate(n, decimals):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def plot_reward():
  #fig = plt.figure(1)
  plt.figure(1)
  plt.clf()
  plt.xlim([0,NUM_EPISODES])
  plt.plot(reward_history,'ro',label="Rewards")
  plt.plot(rew_mean, label="Mean")
  plt.plot(rew_var, label="Variance")
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.title(f'Reward Per Episode (NUM_STEPS={NUM_STEPS})')
  plt.legend(loc=0)
  #plt.pause(0.01)
  plt.show()

def plot_loss():
  plt.figure(2)
  plt.plot(error)
  plt.xlabel("Episodes")
  plt.ylabel("Average Error")
  plt.title("Average_Loss Vs Episodes")
  #plt.show()

def plot_epsilon():
  plt.figure(3)
  plt.plot(epsilon)
  plt.xlabel("Episodes")
  plt.ylabel("Epsilon value")
  plt.title("Epsilon Vs Episodes")
  #plt.show()

env = gym.make('BipedalWalker-v3')
env = env.unwrapped
env.seed(42)

# Print some info about the environment
print("\n========================================================")
print(f"State space (gym calls it observation space) is {env.observation_space}")
print(f"Action space is {env.action_space}")
print(f'\nobservation space is between {env.observation_space.low[0]} and {env.observation_space.high[0]}')
print(f'action space is between {truncate(env.action_space.low[0],2)} and {truncate(env.action_space.high[0],2)}')
print("========================================================\n")

state_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

# Parameters
NUM_EPISODES = 2000
BATCH_SIZE = 64
NUM_STEPS = int(NUM_EPISODES/90)
LEN_EPISODE = 200

RENDER_REWARD_MIN = 5000
RENDER_ENV = True

agent = DQNAgent(state_space,action_space)

reward_history = []
reward_mean=[]
reward_var=[]
error = []
epsilon = []
ave_reward_list = []
game_win = 0

# Run for NUM_EPISODES
for episode in range(NUM_EPISODES):
  curr_state = env.reset()
  curr_state = curr_state.reshape(1,-1)
  time_now = time.time()

  while True:
    # Comment to stop rendering the environment
    # If you don't render, you can speed things up
    env.render()

    # Randomly sample an action from the action space
    # Should really be your exploration/exploitation policy
        #action = env.action_space.sample()
    action = agent.epsilon_policy(curr_state)

    # Step forward and receive next state and reward
    # done flag is set when the episode ends: either goal is reached or
    #       200 steps are done
    next_state, reward, done, _ = env.step(action)
    next_state = next_state.reshape(1,-1)

    # This is where your NN/GP code should go
    # Create target vector
    # Train the network/GP
    # Update the policy
    agent.sarsa(curr_state, action, reward, next_state, done)

    # Current state for next step
    curr_state = next_state

    time_later = time.time()
    time_delta = time_later - time_now

    if time_delta > 20:
      done = True

    # Record history
    episode_reward = sum(agent.ep_rewards)
    mean = np.mean(agent.ep_rewards)
    var = np.var(agent.ep_rewards)
    if episode_reward < -300:
      done = True

    if done==True:
      reward_history.append(episode_reward)
      max_reward = np.max(reward_history)
      reward_mean.append(mean)
      reward_var.append(var)
      episode_max = np.argmax(reward_history)
      if episode_reward >=300 :
        game_win = game_win + 1
        agent.save("BipedalWalker_model_working.h5")

      print("===============================================")
      print(f"Episode: {episode}")
      print(f"Time: {truncate(time_delta, 4)} seconds")
      print(f"Reward: {episode_reward}")
      print(f"Maximum Reward: {max_reward} on Episode: {(episode_max)}")
      print(f"Times finished successfully: {game_win}")

      if (episode+1) % NUM_STEPS ==0:
        print(f"Mean reward of the past {NUM_STEPS} episodes: {np.mean(reward_history[-100:])}")
        ave_reward_list.append(np.mean(reward_history[-100:]))

      history, mean_loss= agent.batch_training(BATCH_SIZE)
      epsilon.append(agent.epsilon)
      error.append(mean_loss)
      if max_reward > RENDER_REWARD_MIN:
      	RENDER_ENV = True
      break

plot_loss()
plot_epsilon()
plot_reward()
agent.save("BipedalWalker_model2.h5")
