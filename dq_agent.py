import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random

from dq_network import DQN
from replay_memory import ReplayMemory_deque as ReplayMemory


	
		
class DQAgent():
	
	def __init__(self, gamma, lr, num_states, n_actions, batch_size=100, #action_space,
			   memory_capacity=10000, tau=0.01, #tau=0.001, #tau=0.005,
			   algorithm=None, env_name=None, checkpoint_dir='./Model_Checkpoints'):
		
		#self.seed = ??
		self.gamma = gamma
		self.lr = lr
		self.batch_size = batch_size
		self.action_space = [i for i in range(n_actions)] #action_space
		self.learn_step_counter = 0
		self.checkpoint_dir = checkpoint_dir
		self.algorithm = algorithm
		self.env_name = env_name
		self.tau = tau # for soft target update 

		self.hidden_nodes = 100 #50 #1000, #128, #1000, #512, #256, #128,
		self.memory = ReplayMemory(memory_capacity)
		
		
		self.Q_policy = DQN(lr=self.lr, input_nodes=num_states, 
							hidden_nodes=self.hidden_nodes,
							output_nodes=n_actions,
							file_name=self.env_name+'_'+self.algorithm+'_Q_policy.pt', 
							checkpoint_dir=self.checkpoint_dir)
						
		
		self.Q_target = DQN(lr=self.lr, 
							input_nodes=num_states, 
							hidden_nodes=self.hidden_nodes,
							output_nodes=n_actions,
							file_name=self.env_name+'_'+self.algorithm+'_Q_target.pt', 
							checkpoint_dir=self.checkpoint_dir)


			
	def choose_action(self, observation, epsilon):
		if np.random.random() > epsilon:
			#state = T.tensor([observation]).to(self.Q_eval.device)
			# to avoid: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (
			observation = np.array(observation)
			state = T.tensor(observation, dtype=T.float32).to(self.Q_policy.device)

			self.Q_target.eval()
			with T.no_grad():
				actions = self.Q_target(state) # Double DQN?

			action = T.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)
		return action
		
		
	def store_transition(self, state, action, reward, next_state, terminal):
		self.memory.append(state, action, reward, next_state, terminal)
		
		
	def sample_memory(self):		
		
		state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample(self.batch_size)
		
		states = T.tensor(state_batch).to(self.Q_policy.device)
		actions = T.tensor(action_batch).to(self.Q_policy.device)
		rewards = T.tensor(reward_batch).to(self.Q_policy.device)
		next_states = T.tensor(next_state_batch).to(self.Q_policy.device)
		terminals = T.tensor(terminal_batch).to(self.Q_policy.device)
		return states, actions, rewards, next_states, terminals


	
	def buffer_memory(self):
		self.memory.buffer_with_unique_experiences() 

	def get_unique_memory(self):
		self.memory.get_unique_experiences() 
		
		
	def save_models(self):
		self.Q_policy.save_checkpoint()
		#self.Q_target.save_checkpoint()
		
	
	def load_models(self, model_dir=None):
		if model_dir is not None:
			print('Here1')
			self.Q_policy.load_checkpoint(model_dir)
		else:
			print('Here2')
			self.Q_policy.load_checkpoint()
		#self.Q_target.load_checkpoint()
	
	
	def update_target_network(self):
		# soft update
		policy_net_state_dict = self.Q_policy.state_dict()
		target_net_state_dict = self.Q_target.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
		self.Q_target.load_state_dict(target_net_state_dict)
		self.Q_target.eval() # NEW, added on may-24

		
	def learn(self):
		#if self.memory.memory_counter < self.batch_size:
		if len(self.memory) < self.batch_size:
			return

		state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.sample_memory()

		batch_indices = np.arange(self.batch_size, dtype=np.int32) 

		q_pred = self.Q_policy(state_batch)[batch_indices, action_batch] # changed on may-24
		
		self.Q_target.eval()
		with T.no_grad():
			q_next = self.Q_target(next_state_batch).max(dim=1)[0] # changed on may-24
			q_next[terminal_batch] = 0.0		
		q_target = reward_batch + self.gamma * q_next
		
		self.Q_policy.optimizer.zero_grad()
		loss = self.Q_policy.loss(q_pred, q_target).to(self.Q_policy.device)

		loss.backward()
		
		self.Q_policy.optimizer.step()

		self.update_target_network()  # soft update
		
		self.learn_step_counter += 1
		return loss.item()


				
	

	def pred_action(self, observation):
		
		with T.no_grad():
			observation = np.array(observation)
			state = T.tensor(observation, dtype=T.float).to(self.Q_policy.device)
			actions = self.Q_policy(state)
			action = T.argmax(actions).item()
			
			return action
		

