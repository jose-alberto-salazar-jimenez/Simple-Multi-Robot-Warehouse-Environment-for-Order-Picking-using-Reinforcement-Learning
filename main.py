from time import time
from environment import WarehouseEnv as Env
from env_utils import action_space, observation_space
from dq_agent import DQAgent
from wh_layouts_v3 import layout_24S8D8C8R_17x18_h1  as WH_LAYOUT
from train_v3 import train_agent


#model_dir = 'Model_Checkpoints/Env-v3_episodes-1000_layout-2S2D2C2R-9x7-h1_num-robots-1_DQNAgent-v2_lr-0.0005_202408191115_Q_policy.pt'
wh_layout_name = 'layout_24S8D8C8R_17x18_h1' #same as WH_LAYOUT
epoch_time_str = str(int(time()))
	

if __name__ == '__main__':
	train_agent(Env, DQAgent, action_space, observation_space, WH_LAYOUT, wh_layout_name, epoch_time_str, 5, 20000, True) 
  #simulate_agent(Env, DQAgent, action_space, observation_space, WH_LAYOUT, wh_layout_name, epoch_time_str, 5, 20000, True, model_dir) 
