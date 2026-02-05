_EMPTY: int = 0
_GRASS: int = 1
_TENT: int = 2
_TREE: int = 3
_MAX_TRIES: int = 10
_FOUR_NEIGHBOURS: list[tuple[int, int]] = [
    (0, 1), (1, 0),
    (0, -1), (-1, 0)
]
_EIGHT_NEIGHBOURS = [
    (-1, 1), (0, 1), (1, 1), (1, 0),
    (1, -1), (0, -1), (-1, -1), (-1, 0)
]
_2x2_CELLS = [
    (0, 0), (0, 1), (1, 0), (1, 1)
]
_NORTH_SOUT_WEST_EAST = [
    (-1, 0), (0, 1),
    (1, 0), (0, -1)
]


def get_neighbours(size_n, size_m, values, x, y, k=4):
    """Returns a list of 4 (k=4) or 8 (k=8) neighbours
        of the cell (x, y).

        k = 4       k = 8
        _ 0 _       0 1 2
        3 x 1       7 x 3
        _ 2 _       6 5 4

    """
    lst_vars = []
    if k == 0:
        neighbours = _2x2_CELLS
    elif k == 4:
        neighbours = _FOUR_NEIGHBOURS
    else:
        neighbours = _EIGHT_NEIGHBOURS

    for row, column in neighbours:
        new_row, new_column = x + row, y + column
        if 0 <= new_row < size_n and 0 <= new_column < size_m:
            lst_vars.append(values[new_row, new_column])

    return lst_vars


def init_north_south_west_east(size_n, size_m, x, y):
    north_south_west_east = {}
    i = 0
    for row, column in _NORTH_SOUT_WEST_EAST:
        new_row, new_column = x + row, y + column
        if 0 <= new_row < size_n and 0 <= new_column < size_m:
            north_south_west_east[i] = (new_row, new_column)
        i += 1

    return north_south_west_east


def print_grid(grid, row_constraints, col_constraints, tent=_TENT):
    size_n = len(grid)
    size_m = len(grid[0])
    print(' ' * 6, end='')
    print(*col_constraints, sep=' ' * 2)
    print('_' * (size_n * 4))

    for row in range(size_n):
        if row_constraints[row] == '':
            print(f'  |', sep='', end=' ')
        else:
            print(f'{row_constraints[row]} |', sep='', end=' ')
        for col in range(size_m):
            if grid[row][col] == tent:
                print("⛺", sep='', end=' ')
            elif grid[row][col] == _TREE:
                print("🌲", sep='', end=' ')
            else:
                print("◻️", sep='', end=' ')
        print()
