
class EpsilonLinearDecay():
	
	def __init__(self, num_episodes, epsilon=1.0, eps_min=0.01):
		assert epsilon <= 1.0 and epsilon >= 0.0
		assert eps_min <= 1.0 and eps_min >= 0.0
		assert epsilon >= eps_min
		
		self.epsilon = epsilon
		self.eps_decay = epsilon/num_episodes
		self.eps_min = eps_min
	
	def decrement(self):
		self.epsilon = max(self.epsilon-self.eps_decay, self.eps_min)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class EpsilonLinearFlatEndDecay():
	
	def __init__(self, num_episodes, epsilon=1.0, eps_min=0.01, last=0.1): #eps_min=0.01, last=0.1): # last 10% of episodes
		assert epsilon <= 1.0 and epsilon >= 0.0
		assert eps_min <= 1.0 and eps_min >= 0.0
		assert epsilon >= eps_min
		
		self.epsilon = epsilon
		self.eps_decay = epsilon/(num_episodes*(1.0-last+eps_min))
		self.eps_min = eps_min
	
	
	def decrement(self):
		self.epsilon = max(self.epsilon-self.eps_decay, self.eps_min)
		
		
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class EpsilonLinearFlatStartDecay():
	
	def __init__(self, num_episodes, epsilon=1.0, eps_min=0.005, first=0.1): # first 10% of episodes
		assert epsilon <= 1.0 and epsilon >= 0.0
		assert eps_min <= 1.0 and eps_min >= 0.0
		assert epsilon >= eps_min
		
		self.epsilon = epsilon
		self.eps_decay = epsilon/(num_episodes*(1.0-first))
		self.eps_min = eps_min
		self.first = first
		self.num_episodes = num_episodes

  
	def decrement(self, episode):
		if episode > self.num_episodes*(self.first):
			self.epsilon = max(self.epsilon-self.eps_decay, self.eps_min)
		
