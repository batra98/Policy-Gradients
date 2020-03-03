import sys
import torch  
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt



def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]),ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)

class Vannilla_Policy_Gradient(nn.Module):
	def __init__(self,state_space,action_space,hidden_size = 256):
		super(Vannilla_Policy_Gradient, self).__init__()

		self.linear1 = nn.Linear(state_space,hidden_size)
		self.linear2 = nn.Linear(hidden_size,action_space)


	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.softmax(self.linear2(x),dim = 1)

		return x

class ActorCritic(nn.Module):

	def __init__(self,state_space,action_space,hidden_size = 256):

		super(ActorCritic,self).__init__()

		# print(state_space,action_space)

		self.critic_linear1 = nn.Linear(state_space,hidden_size)
		self.critic_linear2 = nn.Linear(hidden_size,1)

		self.actor_linear1 = nn.Linear(state_space,hidden_size)
		self.actor_linear2 = nn.Linear(hidden_size,action_space)

	def forward(self,state):
		x = F.relu(self.critic_linear1(state))
		x = self.critic_linear2(x)

		y = F.relu(self.actor_linear1(state))
		y = F.softmax(self.actor_linear2(y),dim = 1)

		return x,y





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

		discounted_rewards = torch.FloatTensor(discounted_rewards)

		if baseline == True:
			discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

		self.var_reward.append((discounted_rewards.sum()).item())

		policy_gradient = []
		for log_prob, Gt in zip(log_probs, discounted_rewards):
			policy_gradient.append(-log_prob * Gt)

		self.optimizer.zero_grad()
		policy_gradient = torch.stack(policy_gradient).sum()
		self.loss.append(policy_gradient)
		policy_gradient.backward()
		self.optimizer.step()

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

			reward_history.append(episode_reward)

		return reward_history




class ActorCritic_agent():
	def __init__(self,env,learning_rate = 3e-4,gamma = 0.99):
		self.env = env
		self.action_space = env.action_space.n
		self.state_space = env.observation_space.shape[0]
		self.gamma = gamma
		self.policy = ActorCritic(self.state_space,self.action_space)
		self.optimizer = optim.Adam(self.policy.parameters(),lr = learning_rate)
		self.beta = 0.001


	def update_policy(self,rewards,values,next_value,log_probs,entropy):
		Qvals = np.zeros(len(values))
		# Qval = next_value

		for i in reversed(range(len(rewards))):
			next_value = rewards[i] + next_value*self.gamma
			Qvals[i] = next_value

		values = torch.FloatTensor(values)
		Qvals = torch.FloatTensor(Qvals)
		log_probs = torch.stack(log_probs)

		advantage = Qvals - values


		Actor_loss = (-log_probs*advantage).mean()
		Critic_loss = advantage.pow(2).mean()


		Actor_Critic_Loss = Actor_loss + 0.5*Critic_loss - self.beta*entropy

		self.optimizer.zero_grad()
		Actor_Critic_Loss.backward()
		self.optimizer.step()


	def get_action(self,state):
		state = torch.from_numpy(state).float().unsqueeze(0)
		value,probs = self.policy.forward(Variable(state))
		action = np.random.choice(self.action_space, p = np.squeeze(probs.detach().numpy()))

		return action,value,probs


	def train(self, max_episode = 3000):

		reward_history = []

		for episode in range(max_episode):

			state = self.env.reset()
			log_probs = []
			rewards = []
			values = []
			episode_reward = 0
			entropy_ = 0

			done = False

			while not done:
				action,value,probs = self.get_action(state)

				log_prob = torch.log(probs.squeeze(0)[action])
				entropy = -torch.sum(probs.mean()*torch.log(probs))
				new_state,reward,done,_ = self.env.step(action)

				rewards.append(reward)
				log_probs.append(log_prob)
				values.append(value.detach().numpy()[0])
				episode_reward += reward
				entropy_ += entropy

				state = new_state


			_, next_value, _ = self.get_action(state)

			self.update_policy(rewards,values,next_value,log_probs,entropy_)

			if episode%100 == 0:
				print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(episode,episode_reward,np.mean(np.array(reward_history[episode-100:episode-1]))),end = "")
				sys.stdout.flush()

			reward_history.append(episode_reward)

		return reward_history















def plotting(returns,window_size = 100):
    averaged_returns = np.zeros(len(returns)-window_size+1)
    max_returns = np.zeros(len(returns)-window_size+1)
    min_returns = np.zeros(len(returns)-window_size+1)
    
    
    for i in range(len(averaged_returns)):
      averaged_returns[i] = np.mean(returns[i:i+window_size])
      max_returns[i] = np.max(returns[i:i+window_size])
      min_returns[i] = np.min(returns[i:i+window_size])
    
#     plt.plot(averaged_returns)
    
#     plot_mean_and_CI(averaged_returns,min_returns,max_returns,'g--','g')
    
    return (averaged_returns,max_returns,min_returns)

	


env = gym.make('CartPole-v0')
learning_rate = 3e-4
gamma = 0.99


episodes = 3000

agent = ActorCritic_agent(env)
reward_history = agent.train(episodes)

window_size = 100

# X = []
# for item in agent.loss:
# 	X.append(item.item())

avg,max_returns,min_returns = plotting(reward_history,window_size)

# agent.var_reward = np.array(agent.var_reward)

# print(np.var(reward_history))
# print(np.var(agent.var_reward))

plot_mean_and_CI(avg,min_returns,max_returns,'r','r')




# plt.plot(X)
# plt.show()


def test(agent,env):
	state = env.reset()

	r = 0

	done = False

	for i in range(200):
		action = agent.get_action(state)
		new_state,reward,done,_ = env.step(action[0])

		env.render()

		state = new_state

		r += reward

		if done:
			break

		# print(done)

		# env.render(state)

	env.close()

	print("\n"+str(r))





# test(agent,env)

# print(policy)


