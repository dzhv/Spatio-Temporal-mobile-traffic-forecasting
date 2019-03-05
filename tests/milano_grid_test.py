import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from milano_grid import map


assert map(1) == (99, 0)
assert map(15) == (99, 14)
assert map(100) == (99, 99)
assert map(101) == (98, 0)
assert map(200) == (98,99)
assert map(201) == (97, 0)
assert map(9801) == (1, 0)
assert map(9999) == (0, 98)

print("test passed")