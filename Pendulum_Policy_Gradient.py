import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import copy
import sys
from collections import namedtuple
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import Categorical,Normal
import matplotlib.pyplot as plt


class ReplayBuffer:
	def __init__(self,state_space,action_space,max_size = int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size,state_space))
		self.action = np.zeros((max_size,action_space))
		self.next_state = np.zeros((max_size,state_space))

		self.reward = np.zeros((max_size,1))
		self.not_done = np.zeros((max_size,1))

		self.device = "cpu"

	def add(self,state,action,next_state,reward,done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward

		self.not_done[self.ptr] = 1. - done


		self.ptr = (self.ptr+1)%self.max_size
		self.size = min(self.size+1,self.max_size)

	def sample(self,batch_size):
		ind = np.random.randint(0,self.size,size = batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to("cpu"),
			torch.FloatTensor(self.action[ind]).to("cpu"),
			torch.FloatTensor(self.next_state[ind]).to("cpu"),
			torch.FloatTensor(self.reward[ind]).to("cpu"),
			torch.FloatTensor(self.not_done[ind]).to("cpu"),



			)




class Actor_DDPG(nn.Module):
	def __init__(self,state_space,action_space,action_bound):
		super(Actor_DDPG,self).__init__()

		self.linear1 = nn.Linear(state_space,400)
		self.linear2 = nn.Linear(400,300)
		self.linear3 = nn.Linear(300,action_space)

		self.action_bound = action_bound

	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))

		return self.action_bound * torch.tanh(self.linear3(x))

class Critic_DDPG(nn.Module):
	def __init__(self,state_space,action_space):
		super(Critic_DDPG,self).__init__()

		self.linear1 = nn.Linear(state_space,400)
		self.linear2 = nn.Linear(400 + action_space,300)
		self.linear3 = nn.Linear(300,1)

	def forward(self,state,action):


		q = F.relu(self.linear1(state))
		q = F.relu(self.linear2(torch.cat([q,action], 1)))

		return self.linear3(q)

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



class DDPG:
	def __init__(self,env,learning_rate = 1e-4,gamma = 0.99):
		self.env = env
		self.state_space = env.observation_space.shape[0]
		self.action_space = env.action_space.shape[0]
		self.action_bound = float(env.action_space.high[0])

		self.actor = Actor_DDPG(self.state_space,self.action_space,self.action_bound).to("cpu")
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = learning_rate)


		self.critic = Critic_DDPG(self.state_space,self.action_space).to("cpu")
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),weight_decay = 1e-2)

		self.gamma = gamma
		self.tau = 0.001
		self.batch_size = 256

	def get_action(self,state):
		state = torch.FloatTensor(state.reshape(1,-1)).to("cpu")
		return self.actor(state).cpu().data.numpy().flatten()


	def update_policy(self,replay_buffer,batch_size):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# print(state)

		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done*self.gamma*target_Q).detach()



		current_Q = self.critic(state,action)

		critic_loss = F.mse_loss(current_Q, target_Q)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		actor_loss = -self.critic(state,self.actor(state)).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		for param, target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
			target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

		for param, target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
			target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)


	def train(self,max_timesteps = 3e4,start_timesteps = 10000):
		replay_buffer = ReplayBuffer(self.state_space,self.action_space)

		state = self.env.reset()
		done = False

		episode_reward = 0

		reward_history = []

		episode_num = 0
		episode_timesteps = 0

		for t in range(int(max_timesteps)):

			episode_timesteps += 1

			if t < start_timesteps:
				action = self.env.action_space.sample()
			else:
				# self.env.render()
				action = (self.get_action(np.array(state)) + np.random.normal(0,self.action_bound*0.1,size = self.action_space)).clip(-self.action_bound,self.action_bound)

			next_state, reward, done, _ = self.env.step(action)

			

			replay_buffer.add(state,action,next_state,reward,float(done))

			# rewards.append(reward)

			

			state = next_state
			episode_reward += reward

			if t >= start_timesteps:
				self.update_policy(replay_buffer,self.batch_size)


			if done:

				# if episode_num%10 == 0:
				print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(episode_num+1,episode_reward,np.mean(np.array(reward_history))),end = "")

				state = self.env.reset()
				done = False

				reward_history.append(episode_reward)


				episode_reward = 0

				episode_num += 1

				episode_timesteps = 0


		return reward_history



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

	def update_policy(self,rewards,log_probs,baseline = False):
		log_probs = torch.stack(log_probs,dim = 0).to(self.train_device).squeeze(-1)
		rewards = torch.stack(rewards,dim = 0).to(self.train_device).squeeze(-1)

		discounted_r = torch.zeros_like(rewards)
		running_add = 0

		for t in reversed(range(0,rewards.size(-1))):
			running_add = running_add*self.gamma + rewards[t]
			discounted_r[t] = running_add

		G = discounted_r

		if baseline == True:
			G = (G - G.mean())

# 
		self.var_reward.append((G.mean()).item())

		loss = -(log_probs*G.detach()).mean()

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()





	def train(self,max_episodes = 3000,baseline = False):
		reward_history = []

		for episode in range(max_episodes):
			rewards = []
			log_probs = []
			episode_rewards = 0

			done = False

			state = self.env.reset()

			while not done:

				action,probs = self.get_action(state)
				action = torch.clamp(action,self.env.action_space.low[0],self.env.action_space.high[0])
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
agent.train()
# reward_history = agent.train()


def test(agent,env,render = False):
	state = env.reset()

	r = 0

	done = False

	while not done:
		action = agent.get_action(state)
		new_state,reward,done,_ = env.step(action)

		if render == True:
			env.render()

		state = new_state

		r += reward


		# print(done)

		# env.render(state)

	env.close()

	print("\n"+str(r))

# for i in range(10):
# 	test(agent,env,True)
