
# warehouse configurations



layout_3S3D3R_7x7_h1 = [   # ---- CASE 2
	# 3S: 3 storage locations 
	# 3D: 3 delivery location
	# 3R: 3 robot initial positions
	# 11x9, horizontally aligned
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
]


layout_3S3D3R_7x7_v1 = [	# ---- CASE 2
# 3 storages, 3 delivery, 1 robot, 7x7, horizontally aligned
	[' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', 'R', ' ', 'R', ' ', 'R', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', 'D', ' ', 'D', ' ', 'D', ' '],
]

layout_3S3D3C3R_8x7_h1 = [   # ---- CASE 3
	# 3S: 3 storage locations 
	# 3D: 3 delivery location
	# 3R: 3 robot initial positions
	# 8x7, horizontally aligned
	[' ', ' ', 'C', 'C', 'C', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' '],
]


layout_3S3D3C3R_7x8_v1 = [	# ---- CASE 3
	# 3S: 3 storage locations 
	# 3D: 3 delivery location
	# 3R: 3 robot initial positions
	# 8x7, vertically aligned
	[' ', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', 'R', ' ', 'R', ' ', 'R', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', 'D', ' ', 'D', ' ', 'D', ' '],
]


layout_8S4D4C4R_10x10_h1 = [   # ---- CASE 4
	# 8S: 8 storage locations 
	# 4D: 4 delivery location
	# 4R: 4 robot initial positions
	# 4C: 4 charging locations
	# 10x10, horizontally aligned
	[' ', ' ', ' ', ' ', 'C', 'C', 'C', 'C', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', 'R', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', 'R', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', ' ', 'R', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
]


layout_8S4D4C4R_10x10_v1 = [   # ---- CASE 4
	# 8S: 8 storage locations 
	# 4D: 4 delivery location
	# 4R: 4 robot initial positions
	# 4C: 4 charging locations
	# 10x10, horizontally aligned
	[' ', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', 'R', ' ', ' ', ' ', 'R', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'R', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' '],
]



layout_10S5D5C5R_11x12_v1 = [  # ------ case 5
	# 10S: 5 storage locations 
	# 5D: 5 delivery location
	# 5R: 4 robot initial posible positions
	# 5C: 4 charging locations
	# 13x12, vertically aligned
	[' ', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'R', ' ', 'R', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' '],
]



layout_10S5D5C5R_12x11_h1 = [  # ------ case 5
	# 10S: 5 storage locations 
	# 5D: 5 delivery location
	# 5R: 4 robot initial posible positions
	# 5C: 4 charging locations
	# 12x13, horizontally aligned
	[' ', ' ', ' ', ' ', 'C', 'C', 'C', 'C', 'C', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', 'R', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', 'R', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
]




layout_24S8D8C8R_17x18_h1 = [  # ------ case 6
	# 24S: 24 storage locations 
	# 8D: 8 delivery location
	# 8R: 8 robot initial posible positions
	# 8C: 8 charging locations
	# 17x18, horizonally aligned
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['S', ' ', 'S', ' ', 'S', ' ', ' ', ' ', ' ', ' ', 'R', ' ', ' ', ' ', ' ', ' ', ' ', 'D'],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
]



layout_24S8D8C8R_18x17_v1 = [  # ------ case 6
	# 24S: 24 storage locations 
	# 8D: 8 delivery location
	# 8R: 8 robot initial posible positions
	# 8C: 8 charging locations
	# 18x17, vertically aligned
	[' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' ', 'S', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', 'R', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'R', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', 'R', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'R', ' ', ' ', ' ', 'R', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	['C', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
	[' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' ', 'D', ' '],
]



