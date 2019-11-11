import gym
import numpy as np
import sys
from collections import namedtuple
import torch.nn.functional as F
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical,Normal
import matplotlib.pyplot as plt



class Vanilla_Policy_Gradient_Continuous(nn.Module):
	def __init__(self,state_space,action_space,action_bound,hidden_size = 256):
		super(Vanilla_Policy_Gradient_Continuous,self).__init__()

		self.linear1 = torch.nn.Linear(state_space,hidden_size)
		self.linear2 = torch.nn.Linear(hidden_size,action_space)
		self.linear3 = torch.nn.Parameter(torch.tensor([1.0]))

		self.action_bound = action_bound

	def forward(self,state):
		x = F.relu(self.linear1(state))

		mu = self.action_bound[0] * F.tanh(self.linear2(x))
		sigma = F.softplus(self.linear3)

		return mu,sigma


class Pendulum_Agent:

	def __init__(self,env,learning_rate = 1e-3,gamma = 0.98):
		self.train_device = 'cpu'
		self.env = env
		self.state_space = env.observation_space.shape[-1]
		self.action_space = env.action_space.shape[-1]
		self.action_bound = env.action_space.high

		self.policy = Vanilla_Policy_Gradient_Continuous(self.state_space,self.action_space,self.action_bound)
		self.optimizer = torch.optim.RMSprop(self.policy.parameters() , lr = learning_rate)

		self.var_reward = []
		self.loss = []
		self.gamma = gamma


	def get_action(self,state,test = False):

		state = torch.from_numpy(state).float().unsqueeze(0)
		mu,sigma = self.policy(Variable(state))

		dist = Normal(mu,sigma)

		if test == True:
			return mu

		action = dist.sample()
		log_prob = dist.log_prob(action)

		return action,log_prob

	def update_policy(self,rewards,log_probs,baseline = True):
		log_probs = torch.stack(log_probs,dim = 0).to(self.train_device).squeeze(-1)
		rewards = torch.stack(rewards,dim = 0).to(self.train_device).squeeze(-1)

		discounted_r = torch.zeros_like(rewards)
		running_add = 0

		for t in reversed(range(0,rewards.size(-1))):
			running_add = running_add*self.gamma + rewards[t]
			discounted_r[t] = running_add

		G = discounted_r

		if baseline == True:
			G = (G - G.mean())/(G.std() + 1e-9)

		loss = -(log_probs*G.detach()).mean()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()





	def train(self,max_episodes = 3000):
		reward_history = []

		for episode in range(max_episodes):
			rewards = []
			log_probs = []
			episode_rewards = 0

			done = False

			state = self.env.reset()

			while not done:

				action,probs = self.get_action(state)
				action = torch.clamp(action,env.action_space.low[0],env.action_space.high[0])
				new_state, reward, done, _ = self.env.step(action)

				log_probs.append(probs)
				rewards.append(reward)

				episode_rewards += reward
				state = new_state

			self.update_policy(rewards,log_probs)

			if episode%100 == 0:
				print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(episode,episode_rewards,np.mean(np.array(reward_history[episode-100:episode-1]))),end = "")
				sys.stdout.flush()

			reward_history.append(episode_rewards)

			if episode%1000 == 0:
				torch.save(self.policy,"Pendulum_model.pth")



		return reward_history




env = gym.make('Pendulum-v0')
agent = Pendulum_Agent(env)

reward_history = agent.train(10000)







