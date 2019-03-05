size = (100, 100)

def map(cell_id):
    # returns row_nr, column_nr in a 2D grid
    num_rows, num_columns = size
    cell_indx = cell_id - 1
    row_nr = int(num_rows - 1 - cell_indx // num_rows)
    column_nr = int(cell_indx % num_rows)
    
    return row_nr, column_nr

def test():
	assert map(1) == (99, 0)
	assert map(15) == (99, 14)
	assert map(100) == (99, 99)
	assert map(101) == (98, 0)
	assert map(200) == (98,99)
	assert map(201) == (97, 0)
	assert map(9801) == (1, 0)
	assert map(9999) == (0, 98)

	print("test passed")
