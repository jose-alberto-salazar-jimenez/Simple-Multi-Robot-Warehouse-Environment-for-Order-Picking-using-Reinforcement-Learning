
import pygame, random, numpy as np

def simulate_agent(episodes, model_filepath):
	MAX_EPISODE_STEPS = 2000 
	NUM_ROBOTS = 3
	env = Env(wh_layout = WH_LAYOUT, render_mode='human')
	
	total_steps = 0
	
	
	EPSILON = 0.0
	
	agent = DQAgent(
				 lr=1e-3,
				 gamma=0.95, 
				 batch_size=100,
				 n_actions = action_space_size,
				 num_states=[observation_space_size],
				 memory_capacity=50000, 
				 checkpoint_dir='./Model_Checkpoints',
				 algorithm='DQNAgent-v02', 
				 env_name='Env-v2')
				 
				 
	
	agent.load_models(model_filepath)
	agent.Q_policy.eval()    # switch model to evaluation mode
	#agent.Q_target.eval() 
	
	for episode in range(episodes):
		
		episode_steps = 0
		episode_reward = 0
		
		episode_terminal = False
		terminals = [False for i in range(NUM_ROBOTS)]
		
		observations, info = env.reset() # 
		
		while not episode_terminal and episode_steps <= MAX_EPISODE_STEPS:	
			pygame.event.get() 

			for robot_idx in range(NUM_ROBOTS):
				observation = observations[robot_idx]
				#action = agent.choose_action(observation, epsilon=0.0)
				action = agent.pred_action(observation)

				next_observation, reward, terminal, info = env.step(robot_idx, action)
				observations[robot_idx] = next_observation

				episode_reward += reward

					
				terminals[robot_idx] = terminal
					
				episode_steps += 1
				episode_terminal = True if True in terminals else False

				if info['packages_delivered']>=MAX_DELIVERIES: # jus to terminate the episode.... not a real terminal state
					episode_terminal = True 
				
			env.render()
			pygame.time.wait(100)
			
		print('episode', episode+1, 
			 ' | steps', episode_steps-1,
			  '| reward %.2f' % episode_reward)
			
			
	env.close()	
