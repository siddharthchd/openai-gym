import gym
import numpy as np
import random
import time
from keras.models import Sequential,Model
from keras.layers import Dropout,Dense,Input,Activation
from keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import random

class Walker:
    def __init__(self,nx,ny,lr,gamma):
        self.nx = nx
        self.ny = ny
        self.lr = lr
        self.los = []
        self.gamma = gamma
        self.memory_deck = deque(maxlen=2000)
        self.epsilon = 0.7
        self.epsilon_ = 0.01
        self.decay = 0.995
        self.model = self.get_model()
        self.episode_observation, self.episode_rewards, self.episode_action, self.new_episode_observation,self.episode_flag = [],[],[],[],[]

    def get_action(self,observation):
        if np.random.rand()<=self.epsilon:
            return np.random.uniform(-1,1,4)
        p = self.model.predict(observation)
        return p[0]

    def memory_recall(self,observation,action,reward,new_observation,flags):
        self.memory_deck.append((observation,action,reward,new_observation,flags))
        self.episode_rewards.append(reward)

    def get_model(self):
        model = Sequential()
        model.add(Dense(400,input_dim=self.nx,activation='relu'))
        model.add(Dense(300,activation='relu'))
        model.add(Dense(self.ny,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.lr))
        return model

    def training(self,batch):
        i = random.sample(self.memory_deck,batch)
        self.los = []
        for obs,act,rew,new_obs,done in i:
            target = rew
            if not done:
                target = ((1.0-0.1)*rew+0.1*(self.gamma*np.amax(self.model.predict(new_obs)[0])))

            old_target = self.model.predict(obs)
            old_target[0] = target
            history = self.model.fit(x=obs,y=old_target,verbose=0,epochs=1)
            self.los.append(history.history['loss'])
            self.episode_observation, self.episode_rewards, self.episode_action, self.new_episode_observation,self.episode_flag = [],[],[],[],[]

        mm = np.mean(self.los)
        if self.epsilon>=self.epsilon_:
            self.epsilon*=self.decay
        return history,mm



episodes = 10000
render = True

env = gym.make('BipedalWalker-v3')
env = env.unwrapped

lr = 0.001
gamma = 0.98
nx = env.observation_space.shape[0]
ny = env.action_space.shape[0]
agent = Walker(nx,ny,lr,gamma)
win=0
rewards_over_time = []
plot_rewards = []

for i in range(episodes):
    observation = env.reset()
    observation = observation.reshape(1,-1)
    start = time.time()
    while True:
        if render==True:
            env.render()

        action = agent.get_action(observation)
        new_observation,reward,flag,inf = env.step(action)
        new_observation = new_observation.reshape(1,-1)
        agent.memory_recall(observation,action,reward,new_observation,flag)
        observation = new_observation

        end = time.time()
        t = end-start
        if t>20:
            flag=True

        total_episode_rewards = sum(agent.episode_rewards)
        if total_episode_rewards<-300:
            flag = True

        if flag == True:
            rewards_over_time.append(total_episode_rewards)
            max_reward = np.max(rewards_over_time)
            if int(total_episode_rewards)>270 and i>2000:
                render=True
            episode_max = np.argmax(rewards_over_time)
            if total_episode_rewards>=300:
                win=win+1
            print('\n')
            print('**********')
            print('Episode : ',i)
            print('Reward : ',int(total_episode_rewards))
            print('Time : ',np.round(t,2),'sec')
            print('Maximum Reward achieved at episode '+ str(episode_max) + ' : ' + str(int(max_reward)))
            print('Wins : '+str(win))
            print('**********')

            hist,mm = agent.training(16)
            plot_rewards.append(total_episode_rewards)

            agent.model.save('winmodel')

            break

eps = np.ndarray(5)
plt.figure(figsize = (10, 7))
plt.plot(plot_rewards)
plt.xlim(0, episodes)
plt.title("Rewards over time")
plt.savefig('rewards.png')
