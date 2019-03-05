import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import data_grid_mapping


test_file = "raw_input.txt"
result = data_grid_mapping.map(data_path=test_file)
assert result[0, 99, 0] == 2
assert result[1, 99, 0] == 3
assert result[2, 99, 0] == 4

assert result[0, 99, 0] == 2
assert result[1, 99, 0] == 3
assert result[2, 99, 0] == 4

assert result[0, 1, 98] == 2
assert result[1, 1, 98] == 0
assert result[2, 1, 98] == 4

print("test successful")

