import pygame, sys, random
from pygame.math import Vector2
import numpy as np



# ---------------------------------------------
# Constant Parameters (Colors in RGB format)

COLOR_SCREEN = (255, 255, 255) # white
COLOR_WALL = (175, 175, 175) #(75,50,25) # wall / obstacle

COLOR_MAIN_ROBOT_STORAGE = (125, 150, 225) # blue-ish
COLOR_MAIN_ROBOT_DELIVERY = (25, 75, 225) # blue
COLOR_MAIN_ROBOT_CHARGING = (25, 175, 225) # ??

COLOR_OTHER_ROBOT_STORAGE = (200, 150, 200) # purple-ish
COLOR_OTHER_ROBOT_DELIVERY = (125, 100, 175) # purple
#COLOR_OTHER_ROBOT_CHARGING = (125, 175, 175) # ??
COLOR_OTHER_ROBOT_CHARGING = (225, 150, 150) # ??

#COLOR_ROBOT_COLLISION = (200,0,0) # red

COLOR_CHARGING_LOCATION_DEACTIVATED = (225, 200, 150) # gold-ish (ligth)
COLOR_CHARGING_LOCATION_ACTIVATED = (175, 125, 0) # gold-ish

COLOR_STORAGE_LOCATION_ACTIVATED = (0, 175, 50) # green-ish
COLOR_STORAGE_LOCATION_DEACTIVATED = (175, 225, 175) # green-ish (light)

COLOR_DELIVERY_LOCATION_ACTIVATED = (225, 100, 25) # orange-ish
COLOR_DELIVERY_LOCATION_DEACTIVATED = (225, 200, 175) # orange-ish (light)

CELL_SIZE = 50 #25
# ---------------------------------------------


# note:
# x coord is vertical (rows)
# y coord is horizontal (cols)


class WarehouseEnv():
	def __init__(self, wh_layout, num_robots=1, num_storage_active=None, num_delivery_active=None, num_charging_active=None, render_mode=None):
		super(WarehouseEnv, self).__init__()
		
		self.wh_layout	= np.array(wh_layout)
		self.num_rows, self.num_cols = self.wh_layout.shape
		
		# new in v2, min = 0, max = sqrt(rows2+cols2)
		self.mix_max_scaling_factor = (self.num_rows**2 + self.num_cols**2)**0.5
		
		self.num_robots = num_robots
		
		self.robots_loc = list(zip(*np.where(self.wh_layout == 'R')))
		self.storage_loc = list(zip(*np.where(self.wh_layout == 'S')))
		self.delivery_loc = list(zip(*np.where(self.wh_layout == 'D')))
		self.charging_loc = list(zip(*np.where(self.wh_layout == 'C')))
		
		self.robots_vec = [Vector2(list(i)) for i in self.robots_loc]
		self.storage_vec = [Vector2(list(i)) for i in self.storage_loc]
		self.delivery_vec = [Vector2(list(i)) for i in self.delivery_loc]
		self.charging_vec = [Vector2(list(i)) for i in self.charging_loc]
		
		self.num_robots_loc = len(self.robots_loc)
		self.num_storage_loc = len(self.storage_loc)
		self.num_delivery_loc = len(self.delivery_loc)
		self.num_charging_loc = len(self.charging_loc)
		
		# number of storage and delivery locations active at the same time.
		self.num_storage_active = num_robots if num_storage_active is None else num_storage_active
		self.num_delivery_active = num_robots if num_delivery_active is None else num_delivery_active
		self.num_charging_active = num_robots if num_charging_active is None else num_charging_active

		self.robot_batt_start_lvl_low = 40
		self.robot_batt_start_lvl_high = 60 #80

		self.robot_batt_level_low = 20 # to start looking for a  charging station
		self.robot_batt_level_high = 80 # to stop charging

		assert self.num_robots > 0, f"Num robots has to be greather than 0, got {self.num_robots}"
		assert self.num_robots <= self.num_robots_loc, f"Num robots has to be less than the places available, got {self.num_robots} and {self.num_robot_loc}."
		assert self.num_robots <= self.num_storage_loc, f"Num robots cannot be greater that the numbers of storage locations available, got {self.num_robots} and {self.num_storage_loc}."
		assert self.num_robots <= self.num_delivery_loc, f"Num robots cannot be greater that the numbers of storage locations available, got {self.num_robots} and {self.num_delivery_loc}."
		assert self.num_storage_active <= self.num_storage_loc, f"Number of storage loc active cannot be greater that the numbers of storage locations available, got {self.num_storage_active} and {self.num_storage_loc}."
		assert self.num_storage_active <= self.num_delivery_loc, f"Number of delivery loc active cannot be greater that the numbers of storage locations available, got {self.num_storage_active} and {self.num_delivery_loc}."
		
		self.render_mode = render_mode
		#self.initialize_rendering(render_mode)
		self.initialize_rendering()
		
	
	def reset(self):
		
		self._reset_robot_location()
		self._reset_storage_location()
		self._reset_delivery_location()
		self._reset_charging_location()

		self.robots_terminal = [False for i in range(self.num_robots)]
		
		self.robots_objective = ['s' for i in range(self.num_robots)] # when reset the objetive is to get to an storage location--- added on aug-13
		#possible objective:
		# - 's' --- to get to a storage location... when not loaded, and battery level is not low
		# - 'd'--- to get to a delivery location... when loaded (battery level shouldn't be low?)
		# - 'c'--- to get to a charging location... when not loaded, and battery level is low 
		
		self.count_deliveries = 0 # used to indicate when an espides finishes
		self.invalid_movements = 0 # either collision with wall or crossing the storage or delivery locations when not allowed
		self.invalid_battery_levels = 0 # when a robot's battery falls under 0 level (<=0)

		self.robots_batt = [np.random.randint(self.robot_batt_start_lvl_low, self.robot_batt_start_lvl_high) for i in range(self.num_robots)]

		self.robots_state = [self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) for robot_idx in range(self.num_robots)]
		
		info = {'packages_delivered': self.count_deliveries, 'invalid_movements': self.invalid_movements, 'invalid_battery_level': self.invalid_battery_levels}

		return self.robots_state, info
		
	

	def step(self, robot_idx, robot_action):			
		# Move the agent based on the selected action
		
		robot_objective = self.robots_objective[robot_idx]

		robot_batt = self.robots_batt[robot_idx]

		if robot_batt < self.robot_batt_level_low and robot_objective=='s':
			robot_objective = 'c'
			self.robots_objective[robot_idx] = robot_objective

		elif robot_batt > self.robot_batt_level_high and robot_objective=='c':
			robot_objective = 's'
			self.robots_objective[robot_idx] = robot_objective

		new_robot_pos = self._move(self.robots_pos[robot_idx], robot_action)
		
		# Check if the new position is valid
		valid_position = self._is_valid_position(robot_idx, new_robot_pos)

		valid_battery_level = True 

		if robot_batt<=0:
			valid_battery_level = False # battery level not valid

		
		if valid_position and valid_battery_level:

			robot_at_storage_loc, which_storage_loc = self._is_robot_at_location(new_robot_pos, self.storage_loc)
			robot_at_delivery_loc, which_delivery_loc = self._is_robot_at_location(new_robot_pos, self.delivery_loc)
			robot_at_charging_loc, which_charging_loc = self._is_robot_at_location(new_robot_pos, self.charging_loc)

			if robot_at_storage_loc and robot_objective == 's':
				self.robots_batt[robot_idx] = robot_batt-0.25 #0.5 #-1 #new added on aug-18
				self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
				self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])

				if which_storage_loc == self.storage_pos[robot_idx]: # changed on aug-21
					robot_reward = 5 #2.5 #1 #0.5 #1
					self.robots_objective[robot_idx] = 'd'
					self._select_new_storage_location(robot_idx) # changed on aug-21

				else: # wrong storage location... how to placed this in the state...?
					robot_reward = -5 #-2.5 #-1 #-0.5 #-1 
			
			elif robot_at_storage_loc and robot_objective != 's': #objective is either delivery or charging
				robot_reward = -5 #-2.5 #-1 #-0.5 #-1 #-10 # change to -1 on may-28
				self.robots_batt[robot_idx] = robot_batt-0.25 #0.5 #-1 #new added on aug-18
				self.invalid_movements += 1
				self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
				self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])

				
			elif robot_at_delivery_loc and robot_objective == 'd':
				self.robots_batt[robot_idx] = robot_batt-0.25 #0.5 #1 #new added on aug-18
				self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
				self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])
				
				if which_delivery_loc == self.delivery_pos[robot_idx]: # changed on aug-21
					robot_reward = 5 #2.5 #1 #0.5 #1
					self.count_deliveries += 1
					self.robots_objective[robot_idx] = 's'
					self._select_new_delivery_location(robot_idx) # changed on aug-21

				else: # wrong delivery location...	how to placed this in the state...?
					robot_reward = -5 #-2.5 #-1 #-0.5 #-1 #-10 # change to -1 on may-28

			elif robot_at_delivery_loc and robot_objective != 'd':
				robot_reward = -5 #-2.5 #-1 #-0.5 #-1 #-10 # change to -1 on may-28
				self.robots_batt[robot_idx] = robot_batt-0.25 #0.5 #1 #new added on aug-18
				self.invalid_movements += 1
				self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
				self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])


			elif robot_at_charging_loc and robot_objective=='c':
				self.robots_batt[robot_idx] = robot_batt+2.5 #3 #+2 #new added on aug-18... charges fasters than it discharges
				self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
				self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])

				if which_charging_loc == self.charging_pos[robot_idx]:
					robot_reward = 0.5 #0.25 #0.5 
				else:
					robot_reward = -0.5 #-0.25 # wrong charging location

			elif robot_at_charging_loc and robot_objective!='c':
				robot_reward = -5 #-2.5 #-1 #-0.25 #-0.5 
				self.robots_batt[robot_idx] = robot_batt+2.5 #+3 #+2 #new added on aug-18... charges fasters than it discharges
				self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
				self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])


			else:
				self.robots_batt[robot_idx] = robot_batt-0.25 #0.5 #1 #new added on aug-18

				self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
				self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])
				
				if robot_objective == 'd': # to encourage to get closer to the delivery station if loaded
					objective_dist = self.robots_pos[robot_idx].distance_to(self.delivery_pos[robot_idx])

				elif robot_objective == 's': # to encourage to get closer to the storage station if not loaded
					objective_dist = self.robots_pos[robot_idx].distance_to(self.storage_pos[robot_idx])

				else: # to encourage to get closer to a charging location if battery is low
					objective_dist = self.robots_pos[robot_idx].distance_to(self.charging_pos[robot_idx])

				robot_reward = -round(0.1*objective_dist/self.mix_max_scaling_factor, 5)

		else:
			robot_reward = -10 #-10 #-1 #-10 #-100 # change to -10 on may-28
			self.robots_terminal[robot_idx] = True
			self.robots_pos[robot_idx] = new_robot_pos #new_robot_pos.copy()
			self.robots_state[robot_idx] = self._get_state(robot_idx, self.robots_pos[robot_idx], self.robots_objective[robot_idx]) #, self.robots_batt[robot_idx])
			if not valid_position:
				self.invalid_movements += 1
			if not valid_battery_level:
				self.invalid_battery_levels += 1

		info = {
				'packages_delivered': self.count_deliveries,
				'invalid_movements': self.invalid_movements,
				'invalid_battery_level': self.invalid_battery_levels
			}
		
		return self.robots_state[robot_idx], robot_reward, self.robots_terminal[robot_idx], info



	def render(self):
		# Clear the screen
		self.screen.fill(COLOR_SCREEN)  

		for row in range(self.num_rows): 		# Draw env elements one cell at a time
			for col in range(self.num_cols):
				cell_left = col * self.cell_size
				cell_top = row * self.cell_size

				if self.wh_layout[row, col] == 'W':  # Obstacle
					pygame.draw.rect(self.screen, COLOR_WALL, (cell_left, cell_top, self.cell_size, self.cell_size))
				elif self.wh_layout[row, col] == 'S':  # Storage position
					pygame.draw.rect(self.screen, COLOR_STORAGE_LOCATION_DEACTIVATED , (cell_left, cell_top, self.cell_size, self.cell_size))
				elif self.wh_layout[row, col] == 'D':  # Delivery position
					pygame.draw.rect(self.screen, COLOR_DELIVERY_LOCATION_DEACTIVATED, (cell_left, cell_top, self.cell_size, self.cell_size))
				elif self.wh_layout[row, col] == 'C':  # Charging position
					pygame.draw.rect(self.screen, COLOR_CHARGING_LOCATION_DEACTIVATED, (cell_left, cell_top, self.cell_size, self.cell_size))
				
				#for storage_pos in self.storage_pos:
				for storage_idx in range(len(self.storage_pos)):
					#if np.array_equal(np.array(storage_pos), np.array([row, col])):
					if np.array_equal(np.array(self.storage_pos[storage_idx]), np.array([row, col])):
						pygame.draw.rect(self.screen, COLOR_STORAGE_LOCATION_ACTIVATED , (cell_left, cell_top, self.cell_size, self.cell_size))

						text = self.font.render(str(storage_idx+1), True, (0,0,0))  # added on aug-22
						self.screen.blit(text, text.get_rect(center=(cell_left+self.cell_size/2, cell_top+self.cell_size/2)))  # added on aug-22
				
				#for delivery_pos in self.delivery_pos:
				for delivery_idx in range(len(self.delivery_pos)):
					#if np.array_equal(np.array(delivery_pos), np.array([row, col])):
					if np.array_equal(np.array(self.delivery_pos[delivery_idx]), np.array([row, col])):
						pygame.draw.rect(self.screen, COLOR_DELIVERY_LOCATION_ACTIVATED , (cell_left, cell_top, self.cell_size, self.cell_size))

						text = self.font.render(str(delivery_idx+1), True, (0,0,0))  # added on aug-22
						self.screen.blit(text, text.get_rect(center=(cell_left+self.cell_size/2, cell_top+self.cell_size/2)))  # added on aug-22

				#for charging_pos in self.charging_pos:
				for charging_idx in range(len(self.charging_pos)):
					#if np.array_equal(np.array(charging_pos), np.array([row, col])):
					if np.array_equal(np.array(self.charging_pos[charging_idx]), np.array([row, col])):
						pygame.draw.rect(self.screen, COLOR_CHARGING_LOCATION_ACTIVATED , (cell_left, cell_top, self.cell_size, self.cell_size))

						text = self.font.render(str(charging_idx+1), True, (0,0,0))  # added on aug-22
						self.screen.blit(text, text.get_rect(center=(cell_left+self.cell_size/2, cell_top+self.cell_size/2)))  # added on aug-22
				

				for robot_idx in range(self.num_robots):
					if robot_idx == 0:
						if np.array_equal(np.array(self.robots_pos[robot_idx]), np.array([row, col])):  # Agent position
							if self.robots_objective[robot_idx]=='s':
								pygame.draw.circle(self.screen, COLOR_MAIN_ROBOT_STORAGE, (cell_left+self.cell_size/2, cell_top+self.cell_size/2), self.cell_size/2)
							elif self.robots_objective[robot_idx]=='d':
								pygame.draw.circle(self.screen, COLOR_MAIN_ROBOT_DELIVERY, (cell_left+self.cell_size/2, cell_top+self.cell_size/2), self.cell_size/2)
							else:
								pygame.draw.circle(self.screen, COLOR_MAIN_ROBOT_CHARGING, (cell_left+self.cell_size/2, cell_top+self.cell_size/2), self.cell_size/2)
							
							text = self.font.render(str(robot_idx+1), True, (0,0,0))  # added on aug-22
							self.screen.blit(text, text.get_rect(center=(cell_left+self.cell_size/2, cell_top+self.cell_size/2)))  # added on aug-22
							

					else:
						if np.array_equal(np.array(self.robots_pos[robot_idx]), np.array([row, col])):  # Agent position
							if self.robots_objective[robot_idx]=='s':
								pygame.draw.circle(self.screen, COLOR_OTHER_ROBOT_STORAGE, (cell_left+self.cell_size/2, cell_top+self.cell_size/2), self.cell_size/2)
							elif self.robots_objective[robot_idx]=='d':
								pygame.draw.circle(self.screen, COLOR_OTHER_ROBOT_DELIVERY, (cell_left+self.cell_size/2, cell_top+self.cell_size/2), self.cell_size/2)
							else:
								pygame.draw.circle(self.screen, COLOR_OTHER_ROBOT_CHARGING, (cell_left+self.cell_size/2, cell_top+self.cell_size/2), self.cell_size/2)
							
							text = self.font.render(str(robot_idx+1), True, (0,0,0))  # added on aug-22
							self.screen.blit(text, text.get_rect(center=(cell_left+self.cell_size/2, cell_top+self.cell_size/2)))  # added on aug-22

		pygame.display.update()  # Update the display
		
	
	def initialize_rendering(self):
		if self.render_mode == 'human':
			# Initialize Pygame
			pygame.init()
			# setting display size
			self.cell_size = CELL_SIZE			
			self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))
			self.font = pygame.font.Font(pygame.font.get_default_font(), 20) # added on aug-22
			#self.font = pygame.font.SysFont(None, 10)  # added on aug-22

				
	def close(self):
		if self.render_mode == 'human':
			pygame.quit() # opposite of init
			sys.exit()
		
	
	def _move(self, robot_position, robot_action):
		
		new_robot_position = robot_position.copy()
		
		if robot_action == 0:		 	# 'wait / do nothing'.... to avoid terminal state or when charging
			return new_robot_position
			
		elif robot_action == 1: 			# 'up or north' 
			new_robot_position.x -= 1
			
		elif robot_action == 2: 			# 'up-right or north-east' 
			new_robot_position.x -= 1
			new_robot_position.y += 1
			
		elif robot_action == 3: 			# 'right or east'
			new_robot_position.y += 1
			
		elif robot_action == 4: 			# 'down-right or south-east' 
			new_robot_position.x += 1
			new_robot_position.y += 1
			
		elif robot_action == 5: 			# 'down or south'
			new_robot_position.x += 1
		
		elif robot_action == 6: 			# 'down-left or sout-west'
			new_robot_position.x += 1
			new_robot_position.y -= 1
			
		elif robot_action == 7: 			# 'left or west'
			new_robot_position.y -= 1
		
		elif robot_action == 8: 			# 'up-left or north-west' 
			new_robot_position.x -= 1
			new_robot_position.y -= 1
			
		else:
			print('Invalid Movement... please verify your action function')
		
		return new_robot_position

		
		
	def _is_valid_position(self, robot_idx, robot_position): :
		x, y = int(robot_position.x), int(robot_position.y)
		
		# If agent goes out of the grid
		if x < 0 or y < 0 or x >= self.num_rows or y >= self.num_cols:
			return False

		# If the agent hits an obstacle
		if self.wh_layout[x, y] == 'W':
			return False

		#if robot_batt < 1: #battery is dead
		#	return False
		
		#robots_pos = [self.robots_pos]
		for other_idx, other_pos in enumerate(self.robots_pos): # to skip each robot itself, it's needed due to action 0
			if x == int(other_pos.x) and y == int(other_pos.y) and robot_idx != other_idx:
				return False
		
		return True
		
	
	def _is_robot_at_location(self, robot_loc, locations):
		for loc in locations:
			dis_to_loc = robot_loc.distance_to(loc)
			#if dis_to_loc==0:
			if dis_to_loc < 1e-5:
				return True, loc
		return False, (None, None)
		


	def _select_new_storage_location(self, replace_idx):
		#self.storage_pos = Vector2(random.choice(self.storage_loc))
		#self.storage_pos[idx] = random.choice(self.storage_vec)
		storage_pos = self.storage_pos.copy()
		storage_pos.pop(replace_idx)
		
		while True:
			new_storage = random.choice(self.storage_vec)
			if new_storage not in storage_pos:
				self.storage_pos[replace_idx] = new_storage
				break
				
	def _select_new_delivery_location(self, replace_idx):
		#self.delivery_pos = Vector2(random.choice(self.delivery_loc))
		#self.delivery_pos[idx] = random.choice(self.delivery_vec)
		
		delivery_pos = self.delivery_pos.copy()
		delivery_pos.pop(replace_idx)
		
		while True:
			new_delivery = random.choice(self.delivery_vec)
			if new_delivery not in delivery_pos:
				self.delivery_pos[replace_idx] = new_delivery
				break
	
		
	def _reset_storage_location(self):
		self.storage_active_idx = np.random.choice(self.num_storage_loc, size=self.num_storage_active, replace=False)
		self.storage_pos = [self.storage_vec[i] for i in self.storage_active_idx]
	
		
	def _reset_delivery_location(self):
		self.delivery_active_idx = np.random.choice(self.num_delivery_loc, size=self.num_delivery_active, replace=False)
		self.delivery_pos = [self.delivery_vec[i] for i in self.delivery_active_idx]

	def _reset_charging_location(self):
		self.charging_active_idx = np.random.choice(self.num_charging_loc, size=self.num_charging_active, replace=False)
		self.charging_pos = [self.charging_vec[i] for i in self.charging_active_idx]
	
		
	def _reset_robot_location(self):	
		self.robots_idx = np.random.choice(self.num_robots_loc, size=self.num_robots, replace=False)
		self.robots_pos = [self.robots_vec[i] for i in self.robots_idx]
		#print(self.robots_pos)
		
		#print(self.robots_pos)
		if self.num_robots>1:
			self.robot_main_pos = [self.robots_pos[0]]
			self.robot_other_pos = self.robots_pos[1:]

		
	
	def _get_state(self, robot_idx, robot_pos, robot_objective): :
		valid_1step_north = self._is_valid_position(robot_idx, robot_pos+[-1,0]) 
		valid_1step_north_east = self._is_valid_position(robot_idx, robot_pos+[-1,1]) 
		valid_1step_east = self._is_valid_position(robot_idx, robot_pos+[0,1]) 
		valid_1step_south_east = self._is_valid_position(robot_idx, robot_pos+[1,1]) 
		valid_1step_south = self._is_valid_position(robot_idx, robot_pos+[1,0]) 
		valid_1step_south_west = self._is_valid_position(robot_idx, robot_pos+[1,-1]) 
		valid_1step_west = self._is_valid_position(robot_idx, robot_pos+[0,-1]) 
		valid_1step_north_west = self._is_valid_position(robot_idx, robot_pos+[-1,-1]) 

		valid_2step_north = self._is_valid_position(robot_idx, robot_pos+[-2,0]) 
		valid_2step_north_east = self._is_valid_position(robot_idx, robot_pos+[-2,2]) 
		valid_2step_east = self._is_valid_position(robot_idx, robot_pos+[0,2]) 
		valid_2step_south_east = self._is_valid_position(robot_idx, robot_pos+[2,2]) 
		valid_2step_south = self._is_valid_position(robot_idx, robot_pos+[2,0]) 
		valid_2step_south_west = self._is_valid_position(robot_idx, robot_pos+[2,-2]) 
		valid_2step_west = self._is_valid_position(robot_idx, robot_pos+[0,-2]) 
		valid_2step_north_west = self._is_valid_position(robot_idx, robot_pos+[-2,-2]) 

		valid_2step_north_1step_east = self._is_valid_position(robot_idx, robot_pos+[-2,1]) 
		valid_1step_north_2step_east = self._is_valid_position(robot_idx, robot_pos+[-1,2]) 
		valid_1step_south_2step_east = self._is_valid_position(robot_idx, robot_pos+[1,2]) 
		valid_2step_south_1step_east = self._is_valid_position(robot_idx, robot_pos+[2,1]) 
		valid_2step_south_1step_west = self._is_valid_position(robot_idx, robot_pos+[2,-1]) 
		valid_1step_south_2step_west = self._is_valid_position(robot_idx, robot_pos+[1,-2]) 
		valid_1step_north_2step_west = self._is_valid_position(robot_idx, robot_pos+[-1,-2]) 
		valid_2step_north_1step_west = self._is_valid_position(robot_idx, robot_pos+[-2,-1]) 



		if robot_objective=='s':
			objective_pos = self.storage_pos[robot_idx] #changed on aug-21
			robot_objective = [1, 0, 0] # changed in sept-2

		elif robot_objective=='d':

			objective_pos = self.delivery_pos[robot_idx] #changed on aug-21
			robot_objective = [0, 1, 0] # changed in sept-2
		
		elif robot_objective=='c':
			objective_pos = self.charging_pos[robot_idx] #changed on aug-21
			robot_objective = [0, 0, 1] # changed in sept-2

		else:
			print('Wrong Objective!... check < Objective > definitions')

		objective_dist = robot_pos.distance_to(objective_pos) #changed on aug-21
		dist_to_objective = round(objective_dist/self.mix_max_scaling_factor, 4) #, 3) #changed on aug-21
		
		objective_is_north = robot_pos.x > objective_pos.x # objective is north
		objective_is_east = robot_pos.y < objective_pos.y # objective is east
		objective_is_south = robot_pos.x < objective_pos.x  # objective is south
		objective_is_west = robot_pos.y > objective_pos.y  # objective is west  
		
		state = [
			valid_1step_north, 
			valid_1step_north_east,
			valid_1step_east,
			valid_1step_south_east,
			valid_1step_south,
			valid_1step_south_west,
			valid_1step_west,
			valid_1step_north_west,

			valid_2step_north, 
			valid_2step_north_east,
			valid_2step_east,
			valid_2step_south_east,
			valid_2step_south,
			valid_2step_south_west,
			valid_2step_west,
			valid_2step_north_west,
			valid_2step_north_1step_east,
			valid_1step_north_2step_east,
			valid_1step_south_2step_east,
			valid_2step_south_1step_east,
			valid_2step_south_1step_west,
			valid_1step_south_2step_west,
			valid_1step_north_2step_west,
			valid_2step_north_1step_west,
			
			objective_is_north,
			objective_is_east,
			objective_is_south,
			objective_is_west,
			
			*robot_objective,
			
			dist_to_objective,
		]
		
		return np.array(state, dtype=np.float32)

