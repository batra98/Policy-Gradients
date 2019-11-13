import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import copy


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


	def train(self,max_timesteps = 1e6,start_timesteps = 10000):
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
				self.env.render()
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







env = gym.make('Pendulum-v0')
agent = DDPG(env)

agent.train()
