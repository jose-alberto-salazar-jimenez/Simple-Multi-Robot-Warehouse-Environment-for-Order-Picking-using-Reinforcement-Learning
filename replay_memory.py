

# ---------------------------------------------		
# numpy version -------------------------------
# ---------------------------------------------

import numpy as np

class ReplayMemory_numpy():


	def __init__(self, memory_capacity, state_input_shape):
		
		self.state_input_shape = state_input_shape
		
		self.memory_capacity = memory_capacity
		self.memory_counter = 0
		self.state_memory = np.zeros((self.memory_capacity, *state_input_shape), dtype=np.float32)
		self.action_memory = np.zeros(self.memory_capacity, dtype=np.int32) #dtype=np.int64) #dtype=np.float32)
		self.reward_memory = np.zeros(self.memory_capacity, dtype=np.float32)
		self.next_state_memory = np.zeros((self.memory_capacity, *state_input_shape), dtype=np.float32)
		self.terminal_memory = np.zeros(self.memory_capacity, dtype=np.int32)		
		self.experience_counter = 0

	def append(self, state, action, reward, next_state, terminal):
		index = self.memory_counter % self.memory_capacity # first unoccupied place to store memory
		self.state_memory[index] = state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.next_state_memory[index] = next_state
		self.terminal_memory[index] = terminal
		self.memory_counter += 1
		self.experience_counter += 1
	
	
	def sample(self, sample_size):
		max_memory = min(self.memory_counter, self.memory_capacity)
		sample_indices = np.random.choice(max_memory, sample_size, replace=False)
		states = self.state_memory[sample_indices]
		actions = self.action_memory[sample_indices] 
		rewards = self.reward_memory[sample_indices]
		next_states = self.next_state_memory[sample_indices] 
		terminals = self.terminal_memory[sample_indices]
		return states, actions, rewards, next_states, terminals
	
	def __len__(self):
		return self.memory_counter

		

# ---------------------------------------------		
# deque version -------------------------------
# ---------------------------------------------

from collections import deque, namedtuple
import random



class ReplayMemory_deque():

	def __init__(self, memory_capacity):
		self.memory = deque([], maxlen=memory_capacity)
		self.experience = namedtuple('experience', ('state', 'action', 'reward', 'next_state', 'terminal'))
	
	def append(self, state, action, reward, next_state, terminal):
		"""Adds a new experience to the memory."""
		experience = self.experience(state, action, reward, next_state, terminal)
		self.memory.append(experience)
	
	def sample(self, sample_size):
		"""Returns a random sample (batch) of experiences from memory."""
		experience_batch = random.sample(self.memory, sample_size)

		states = np.array([e.state for e in experience_batch if e is not None], dtype=np.float32) #.astype(np.float32)
		actions = np.array([e.action for e in experience_batch if e is not None], dtype=np.int32) #.astype(np.int32)
		rewards = np.array([e.reward for e in experience_batch if e is not None], dtype=np.float32) #.astype(np.float32)
		next_states = np.array([e.next_state for e in experience_batch if e is not None], dtype=np.float32) #.astype(np.float32)
		terminals = np.array([e.terminal for e in experience_batch if e is not None], dtype=np.int32) #.astype(np.int32) #.astype(np.uint8) #.astype(np.int32)

		return states, actions, rewards, next_states, terminals

	
	def __len__(self):
		"""Returns the current size of the memory."""
		return len(self.memory)
