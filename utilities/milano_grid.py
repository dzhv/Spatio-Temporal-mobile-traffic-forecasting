size = (100, 100)

def map(cell_id):
    # returns row_nr, column_nr in a 2D grid
    num_rows, num_columns = size
    cell_indx = cell_id - 1
    row_nr = int(num_rows - 1 - cell_indx // num_rows)
    column_nr = int(cell_indx % num_rows)
    
    return row_nr, column_nr
