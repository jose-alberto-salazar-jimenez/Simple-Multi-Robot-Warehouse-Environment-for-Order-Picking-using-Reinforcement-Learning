
# - changed observation_space to account for diagonal movement

action_space = [ # the numbers represent the index value of the output from the neural net.
	'wait / do nothing', 		# 0
	'up or north', 			# 1
	'up-right or north-east', 	# 2
	'right or east', 			# 3
	'down-right or south-east', 	# 4
	'down or south', 			# 5
	'down-left or sout-west', 	# 6
	'left or west', 			# 7
	'up-left or north-west' 		# 8
	
]


observation_space = [
	# 1 step ahead
	'valid_1step_north', 	# bool 
	'valid_1step_north_east',
	'valid_1step_east',
	'valid_1step_south_east',
	'valid_1step_south',
	'valid_1step_south_west',
	'valid_1step_west',
	'valid_1step_north_west',
	
	# 2 steps ahead
	'valid_2step_north', 
	'valid_2step_north_east',
	'valid_2step_east',
	'valid_2step_south_east',
	'valid_2step_south',
	'valid_2step_south_west',
	'valid_2step_west',
	'valid_2step_north_west',
	'valid_2step_north_1step_east',
	'valid_1step_north_2step_east',
	'valid_1step_south_2step_east',
	'valid_2step_south_1step_east',
	'valid_2step_south_1step_west',
	'valid_1step_south_2step_west',
	'valid_1step_north_2step_west',
	'valid_2step_north_1step_west',
	
	# relative objective location position
	'objective_is_north', # objective is north
	'objective_is_east',  # objective is east
	'objective_is_south', # objective is south
	'objective_is_west',  # objective is west  

  # objective it has to reach to
	'objective: storage location (s)',
	'objective: delivery location (d)',
	'objective: charging location (c)',
	
	# distance to (the correct) objective position location
	'dist_to_objective',
	

]

