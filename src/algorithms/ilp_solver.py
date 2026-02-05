from src.grid.grid_utils import get_neighbours, print_grid
from src.grid.grid_utils import init_north_south_west_east

import time
from ortools.sat.python import cp_model

TREE = 3

def ilp_solver(grid, row_constraints, col_constraints):
    size_n = len(grid)
    size_m = len(grid[0])
    model = cp_model.CpModel()

    # Список расположения деревьев
    lst_trees = []
    for i in range(size_n):
        for j in range(size_m):
            if grid[i][j] == TREE:
                lst_trees.append((i, j))

    # Инициализация переменных
    tents = {}  # Словарь для хранения ссылок на переменные палаток
    for i in range(size_n):
        for j in range(size_m):
            tents[i, j] = model.NewBoolVar(f'x_{i}_{j}')

    # trees = {} # Словарь для хранения ссылок на переменные деревьев
    '''Переменные отвечающие за привязку дерева и палатки:
            north — север  (0)
            east  — восток (1)
            south — юг     (2)
            west  — запад  (3)

            _ 0 _
            3 x 1
            _ 2 _ 
        '''
    north = {}
    east = {}
    south = {}
    west = {}
    trees_links = {} # Словарь для хранения связей к каждому дереву
    for tree in lst_trees:
        n_s_w_e = init_north_south_west_east(size_n, size_m, tree[0], tree[1])
        trees_links[tree] = []
        if 0 in n_s_w_e:
            index = (n_s_w_e[0][0], n_s_w_e[0][1])
            north[index] = model.NewBoolVar(f'north_{index[0]}_{index[1]}')
            trees_links[tree].append(north[index])
        if 1 in n_s_w_e:
            index = (n_s_w_e[1][0], n_s_w_e[1][1])
            east[index] = model.NewBoolVar(f'east_{index[0]}_{index[1]}')
            trees_links[tree].append(east[index])
        if 2 in n_s_w_e:
            index = (n_s_w_e[2][0], n_s_w_e[2][1])
            south[index] = model.NewBoolVar(f'south_{index[0]}_{index[1]}')
            trees_links[tree].append(south[index])
        if 3 in n_s_w_e:
            index = (n_s_w_e[3][0], n_s_w_e[3][1])
            west[index] = model.NewBoolVar(f'west_{index[0]}_{index[1]}')
            trees_links[tree].append(west[index])

    # Определение ограничений

    for i in range(size_n):
        for j in range(size_m):
            # Палатку нельзя разместить на дереве
            if (i, j) in lst_trees:
                model.Add(tents[i, j] == 0)
                continue

            # Если по соседству с ячейкой нет ни одного дерева, то и палатки нет
            found = False
            for tree in lst_trees:
                if abs(tree[0] - i) + abs(tree[1] - j) <= 1:
                    found = True
                    break

            if not found:
                model.Add(tents[i, j] == 0)

    # Никакие две палатки не находятся рядом друг с другом по горизонтали,
    # вертикали или диагонали (в каждом 2 × 2 квадрате может быть
    # поставлено не более одной палатки)
    for i in range(size_n - 1):
        for j in range(size_m - 1):
            tent_four_cells = get_neighbours(size_n, size_m, tents, i, j, k=0)
            model.Add(sum(tent_four_cells) <= 1)

    # Ограничения в ряду
    for k in range(size_n):
        if row_constraints[k] != '':
            lst_vars = [tents[i, j] for i, j in tents if i == k]
            model.Add(sum(lst_vars) == row_constraints[k])

    # Ограничения в столбце
    for k in range(size_m):
        if col_constraints[k] != '':
            lst_vars = [tents[i, j] for i, j in tents if j == k]
            model.Add(sum(lst_vars) == col_constraints[k])

    # К каждому дереву должна быть привязана ровно одна палатка
    for key in lst_trees:
        # model.Add(sum(trees_links[key]) == 1).OnlyEnforceIf(trees[key])
        model.Add(sum(trees_links[key]) == 1)

    # Каждая палатка привязана ровно к одному дереву
    for i in range(size_n):
        for j in range(size_m):
            lst_vars = []
            if (i, j) in north:
                lst_vars.append(north[i, j])
            if (i, j) in east:
                lst_vars.append(east[i, j])
            if (i, j) in south:
                lst_vars.append(south[i, j])
            if (i, j) in west:
                lst_vars.append(west[i, j])

            if lst_vars:
                model.Add(sum(lst_vars) == tents[i, j])

    start_time = time.perf_counter()
    ans = cp_model.CpSolver()
    end_time = time.perf_counter()


    elapsed_time = end_time - start_time
    print('-'*20)
    print(f"The task took {elapsed_time:.7f} seconds to complete.")
    print('-' * 20)
    ans.Solve(model)
    print(ans.ResponseStats())

    grid = []
    lst_vars = []
    for i in range(size_n):
        row = []
        for j in range(size_m):
            row.append(ans.Value(tents[i, j]))
            lst_vars.append((tents[i, j], ans.Value(tents[i, j])))
        grid.append(row)

    for tree in lst_trees:
        grid[tree[0]][tree[1]] = TREE
        # lst_vars.append((trees[tree], ans.Value(trees[tree])))

    for k, v in north.items():
        lst_vars.append((north[k], ans.Value(v)))

    for k, v in south.items():
        lst_vars.append((south[k], ans.Value(v)))

    for k, v in west.items():
        lst_vars.append((west[k], ans.Value(v)))

    for k, v in east.items():
        lst_vars.append((east[k], ans.Value(v)))

    file = open('values.txt', 'w')
    for i in lst_vars:
        file.write(f'{i[0]} = {i[1]}\n')

    file.close()

    print_grid(grid, row_constraints, col_constraints, tent=1)

    return grid
