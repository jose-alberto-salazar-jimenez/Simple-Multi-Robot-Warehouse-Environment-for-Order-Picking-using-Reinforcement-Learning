import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class DQN(nn.Module): # deep q network
	
	def __init__(self, lr, input_nodes, hidden_nodes, output_nodes, checkpoint_dir, file_name):
		super(DQN, self).__init__()	
		self.checkpoint_file = os.path.join(checkpoint_dir, file_name)
		
		self.fc1 = nn.Linear(*input_nodes, hidden_nodes)
		self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
		self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
		self.fc4 = nn.Linear(hidden_nodes, output_nodes)#

		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()

		
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)
			
			
	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		actions = self.fc4(x)
		return actions
		
		
	def save_checkpoint(self):
		print('... saving network checkpoint ...')
		T.save(self.state_dict(), self.checkpoint_file)
		
		
	def load_checkpoint(self, model_dir=None):
		print('... loading network checkpoint ...')
		if model_dir is not None:
			print('Here1')
			T.load(model_dir)
		else:
			print('Here2')
			T.load(self.checkpoint_file)

