import pygame, random, numpy as np
from epsilon_decay import EpsilonLinearFlatEndDecay as EpsilonDecay


def train_agent(Env, DQAgent, action_space, observation_space, WH_LAYOUT, WH_LAYOUT_NAME, EPOCH_TIME_STR,  NUM_ROBOTS,  episodes, render=False):
	MAX_EPISODE_STEPS = 5000
	observation_space_size = len(observation_space)
	action_space_size = len(action_space)
	MAX_DELIVERIES = 100
	EPSILON = 1.0
	LR = 0.0001 

	NUM_ROBOTS = NUM_ROBOTS
	GAMMA = 0.99
	BATCH_SIZE = 50
	MEMORY_CAPACITY = 100000 
	
	env = Env(
			wh_layout=WH_LAYOUT,
			num_robots=NUM_ROBOTS,
			render_mode='human' if render else None
		    )
	
	
	agent = DQAgent(
				 lr=LR,
				 gamma=GAMMA, 
				 batch_size=BATCH_SIZE,
				 n_actions = action_space_size,
				 num_states=[observation_space_size],
				 memory_capacity=MEMORY_CAPACITY, 
				 checkpoint_dir='./Model_Checkpoints',
				 algorithm='DQNAgent-v2_lr-'+str(LR)+'_epoch-'+EPOCH_TIME_STR, 
				 env_name='Env-v3_episodes-'+str(episodes)+'_'+WH_LAYOUT_NAME+'-'+str(NUM_ROBOTS)
				)
				 
	
	total_steps = 0
	best_avg_reward_100 = -np.inf 
	best_episode_reward = -np.inf 
	
	filename = agent.algorithm + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(episodes)+ '_episodes.png'
	
	rewards_hist, deliveries_hist, epsilons_hist, steps_hist, invalids_hist = [],  [], [], [], []

	epsilonstrategy = EpsilonDecay(episodes)
	
	print('Training started...')
	
	print('Episode | Steps | Avg_Loss | Reward | Avg_Reward_10 | Avg_Reward_100 | Best_Avg_Reward_100 | Epsilon | Deliveries | Invalid_Mvts | Invalid_Batt_Lvl' )
	
	for episode in range(episodes):
		
		episode_steps = 0
		episode_reward = 0

		episode_loss = 0
		
		episode_terminal = False
		terminals = [False for i in range(NUM_ROBOTS)]
		
		observations, info = env.reset() 

		while not episode_terminal and episode_steps <= MAX_EPISODE_STEPS:
		
			if env.render_mode == 'human':
				pygame.event.get() 
				epsilon = epsilonstrategy.epsilon	
				
				for robot_idx in range(NUM_ROBOTS):
					observation = observations[robot_idx]
					action = agent.choose_action(observation, epsilon)
					next_observation, reward, terminal, info = env.step(robot_idx, action)

					agent.store_transition(observation, action, reward, next_observation, int(terminal))
					episode_loss += agent.learn() # trains and returns loss

					observations[robot_idx] = next_observation
					episode_reward += reward
					
					terminals[robot_idx] = terminal
					
					episode_steps += 1

				episode_terminal = True if True in terminals else False

				if info['packages_delivered']>=MAX_DELIVERIES: # jus to terminate the episode.... not a real terminal state
					episode_terminal = True 
				
				env.render()
				pygame.time.wait(5)
			
				
			else: 
				epsilon = epsilonstrategy.epsilon	

				for robot_idx in range(NUM_ROBOTS):
					observation = observations[robot_idx]
					action = agent.choose_action(observation, epsilon)
					next_observation, reward, terminal, info = env.step(robot_idx, action)
					agent.store_transition(observation, action, reward, next_observation, int(terminal))

					episode_loss += agent.learn() # trains and returns loss
					observations[robot_idx] = next_observation
					episode_reward += reward
					
					terminals[robot_idx] = terminal
					
					episode_steps += 1

				episode_terminal = True if True in terminals else False

				if info['packages_delivered']>=MAX_DELIVERIES: # jus to terminate the episode.... not a real terminal state
					episode_terminal = True 

			
			
		rewards_hist.append(episode_reward)
		
		total_steps += episode_steps-1
		
		avg_reward_100 = np.mean(rewards_hist[-100:]) # moving 100 average
		avg_reward_10 = np.mean(rewards_hist[-10:]) # moving 100 average


		if total_steps < 100:
			avg_episode_loss = -np.inf
		else:
			avg_episode_loss = round(episode_loss/(episode_steps),5)
					 
		print(episode+1, '|', episode_steps-1, '| %.4f' % avg_episode_loss, '| %.3f' % episode_reward, 
			 '| %.3f' % avg_reward_10, '| %.3f' % avg_reward_100, '| %.3f' % best_avg_reward_100, ' | %.5f' % epsilon,
			 '|', info['packages_delivered'], '|', info['invalid_movements'], '|', info['invalid_battery_level'])
					 
			
		if avg_reward_100 > best_avg_reward_100: 
			best_avg_reward_100 = avg_reward_100
			if episode_reward > avg_reward_100 and episode_reward>0 and episode_reward>avg_reward_100:
				agent.save_models()
				
		epsilonstrategy.decrement()	
	
	
	
	# test to see whether the model works....
	print('TESTING started...')
	
	print('Episode | Steps | Reward | Deliveries | Invalid_Mvts | Last Action' )
	
	agent.Q_policy.eval()    # switch model to evaluation mode
	agent.Q_target.eval() 
	
	
	for episode in range(5):
		
		episode_steps = 0
		episode_reward = 0
		
		episode_terminal = False
		terminals = [False for i in range(NUM_ROBOTS)]
		
		observations, info = env.reset() # 
		
		while not episode_terminal and episode_steps <= MAX_EPISODE_STEPS:
		
			pygame.event.get() # necessary?
			
			for robot_idx in range(NUM_ROBOTS):
				observation = observations[robot_idx]
				action = agent.pred_action(observation)
				next_observation, reward, terminal, info = env.step(robot_idx, action)
				observations[robot_idx] = next_observation
				terminals[robot_idx] = terminal
				episode_reward += reward
				episode_steps += 1
			
			episode_terminal = True if True in terminals else False
			
			env.render()
			pygame.time.wait(5)

	
		print(episode+1,'|', episode_steps-1, '| %.3f' % episode_reward, 
			 '|', info['packages_delivered'], '|', info['invalid_movements'], '|', action)
	
			
	env.close()	
