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
from torch.distributions import Categorical
# import matplotlib.pyplot as plt
# 

class ActorCritic(nn.Module):

	def __init__(self,state_space,action_space,hidden_size = 128):
		super(ActorCritic,self).__init__()

		self.linear1 = nn.Linear(state_space,hidden_size)

		self.actor_linear = nn.Linear(hidden_size,action_space)
		self.critic_linear = nn.Linear(hidden_size,1)

		self.saved_actions = []
		self.rewards = []

	def forward(self,state):

		state = F.relu(self.linear1(state))

		y = F.softmax(self.actor_linear(state),dim = -1)

		x = self.critic_linear(state)

		return x,y


class Vannilla_Policy_Gradient(nn.Module):
	def __init__(self,state_space,action_space,hidden_size = 256):
		super(Vannilla_Policy_Gradient, self).__init__()

		self.linear1 = nn.Linear(state_space,hidden_size)
		self.linear2 = nn.Linear(hidden_size,action_space)


	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.softmax(self.linear2(x),dim = 1)

		return x

class CartPole_agent:

	def __init__(self,env,learning_rate = 3e-4,gamma = 0.99):
		self.env = env
		self.action_space = env.action_space.n
		self.state_space = env.observation_space.shape[0]
		self.gamma = gamma
		self.policy = Vannilla_Policy_Gradient(self.state_space,self.action_space)
		self.optimizer = optim.Adam(self.policy.parameters(),lr = learning_rate)
		self.var_reward = []
		self.loss = []


	def get_action(self,state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		probs = self.policy.forward(Variable(state))
		action = np.random.choice(self.action_space, p = np.squeeze(probs.detach().numpy()))

		return action,probs

	def update_policy(self,rewards,log_probs,baseline):

		discounted_rewards = []

		for t in range(len(rewards)):
			Gt = 0
			pw = 0
			for r in rewards[t:]:
				Gt = Gt + self.gamma**pw * r
				pw = pw + 1
			discounted_rewards.append(Gt)

		# print(len(log_probs),len(discounted_rewards))

		# l = len(discounted_rewards)
		# l = int(l/2)

		# R1 = discounted_rewards[:l]
		# R2 = discounted_rewards[l:]

		# l_p1 = log_probs[:l]
		# l_p2 = log_probs[l:]

		# R1 = torch.FloatTensor(R1)
		# R2 = torch.FloatTensor(R2)

		# R2 = R2 - R1.mean()

		# print(R2,l_p2)

		# baseline = True

		discounted_rewards = torch.FloatTensor(discounted_rewards)

		if baseline == True:
			discounted_rewards = (discounted_rewards - discounted_rewards.mean())

		self.var_reward.append((discounted_rewards.mean()).item())
		

		policy_gradient = []
		for log_prob, Gt in zip(log_probs, discounted_rewards):
			policy_gradient.append(-log_prob * Gt)

		self.optimizer.zero_grad()
		policy_gradient = torch.stack(policy_gradient).sum()
		self.loss.append(policy_gradient)
		policy_gradient.backward()
		self.optimizer.step()

		del log_probs[:]
		del rewards[:]

	def train(self, max_episode=3000,baseline = False):

		reward_history = []

		for episode in range(max_episode):

			state = self.env.reset()
			log_probs = []
			rewards = []
			episode_reward = 0

			done = False

			
			while not done:
				action, probs = self.get_action(state)
				log_prob = torch.log(probs.squeeze(0)[action])
				new_state, reward, done, _ = self.env.step(action)


				

				log_probs.append(log_prob)
				rewards.append(reward)
				episode_reward += reward                
				state = new_state

			self.update_policy(rewards,log_probs,baseline)

			if episode%100 == 0:
				print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(episode,episode_reward,np.mean(np.array(reward_history[episode-100:episode-1]))),end = "")
				sys.stdout.flush()

				if np.mean(np.array(reward_history[episode-100:episode-1])) >= 195:
					print("Solved")
					break

			reward_history.append(episode_reward)

		return reward_history

class ActorCritic_agent():

	def __init__(self,env,learning_rate = 3e-2,gamma = 0.99):
		self.env = env
		self.action_space = env.action_space.n
		self.state_space = env.observation_space.shape[0]
		self.gamma = gamma
		self.policy = ActorCritic(self.state_space,self.action_space)
		self.SavedAction = namedtuple('SavedAction',['log_prob','value'])
		self.optimizer = optim.Adam(self.policy.parameters(),lr = learning_rate)

	def get_action(self,state):
		state = torch.from_numpy(state).float()

		value , probs = self.policy(state)
		action = np.random.choice(self.action_space, p = np.squeeze(probs.detach().numpy()))

		self.policy.saved_actions.append(self.SavedAction(torch.log(probs.squeeze(0)[action]),value))

		return action

	def update_policy(self):
		R = 0
		saved_actions = self.policy.saved_actions
		policy_losses = []
		value_losses = []
		returns = []

		for r in self.policy.rewards[::-1]:
			R = r+self.gamma*R
			returns.insert(0,R)

		returns = torch.tensor(returns)
		returns = (returns - returns.mean())/(returns.std()+1e-9)

		for (log_prob,value), R in zip(saved_actions,returns):
			advantage = R - value.item()

			policy_losses.append(-log_prob*advantage)

			value_losses.append(F.smooth_l1_loss(value,torch.tensor([R])))

		self.optimizer.zero_grad()

		loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

		loss.backward()
		self.optimizer.step()

		del self.policy.rewards[:]
		del self.policy.saved_actions[:]


	def train(self,max_episode = 3000):

		reward_history = []

		for episode in range(max_episode):
			state = self.env.reset()

			episode_reward = 0

			done = False

			while not done:
				action = self.get_action(state)
				state,reward,done,_ = self.env.step(action)

				self.policy.rewards.append(reward)

				episode_reward += reward



			self.update_policy()

			if episode%100 == 0:
				print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(episode,episode_reward,np.mean(np.array(reward_history[episode-100:episode-1]))),end = "")
				sys.stdout.flush()

			reward_history.append(episode_reward)

		return reward_history

# def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
#     # plot the shaded range of the confidence intervals
#     plt.fill_between(range(mean.shape[0]),ub, lb,
#                      color=color_shading, alpha=.5)
#     # plot the mean on top
#     plt.plot(mean, color_mean)


# def plotting(returns,window_size = 100):
#     averaged_returns = np.zeros(len(returns)-window_size+1)
#     max_returns = np.zeros(len(returns)-window_size+1)
#     min_returns = np.zeros(len(returns)-window_size+1)
    
    
#     for i in range(len(averaged_returns)):
#       averaged_returns[i] = np.mean(returns[i:i+window_size])
#       max_returns[i] = np.max(returns[i:i+window_size])
#       min_returns[i] = np.min(returns[i:i+window_size])
    
# #     plt.plot(averaged_returns)
    
# #     plot_mean_and_CI(averaged_returns,min_returns,max_returns,'g--','g')
    
#     return (averaged_returns,max_returns,min_returns)


# env = gym.make('CartPole-v0')
# agent = ActorCritic_agent(env)

# window_size = 100

# reward_history=agent.train(2000)

# avg,max_returns,min_returns = plotting(reward_history,window_size)
# plot_mean_and_CI(avg,min_returns,max_returns,'r','r')
# plt.show()


# def test(agent,env,render = False):
# 	state = env.reset()

# 	r = 0

# 	done = False

# 	for i in range(200):
# 		action = agent.get_action(state)
# 		new_state,reward,done,_ = env.step(action[0])

# 		if render == True:
# 			env.render()

# 		state = new_state

# 		r += reward

# 		if done:
# 			break

# 		# print(done)

# 		# env.render(state)

# 	env.close()

# 	print("\n"+str(r))




# # for i in range(10):
# 	# test(agent,env)



