from src.grid.grid_utils import get_neighbours, print_grid
from src.grid.grid_utils import init_north_south_west_east

import time
from ortools.linear_solver import pywraplp

TREE = 3


def ilp_solver(grid, row_constraints, col_constraints):
    size_n = len(grid)
    size_m = len(grid[0])

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if solver is None:
        raise RuntimeError("SCIP solver is not available in this OR-Tools installation")

    # Список расположения деревьев
    lst_trees = []

    for i in range(size_n):
        for j in range(size_m):
            if grid[i][j] == TREE:
                lst_trees.append((i, j))

    # Переменные палаток
    tents = {}

    for i in range(size_n):
        for j in range(size_m):
            tents[i, j] = solver.BoolVar(f'x_{i}_{j}')

    # Переменные связи дерева и палатки
    north = {}
    east = {}
    south = {}
    west = {}

    trees_links = {}

    for tree in lst_trees:
        n_s_w_e = init_north_south_west_east(
            size_n,
            size_m,
            tree[0],
            tree[1]
        )

        trees_links[tree] = []

        if 0 in n_s_w_e:
            index = (n_s_w_e[0][0], n_s_w_e[0][1])
            north[index] = solver.BoolVar(
                f'north_{index[0]}_{index[1]}'
            )
            trees_links[tree].append(north[index])

        if 1 in n_s_w_e:
            index = (n_s_w_e[1][0], n_s_w_e[1][1])
            east[index] = solver.BoolVar(
                f'east_{index[0]}_{index[1]}'
            )
            trees_links[tree].append(east[index])

        if 2 in n_s_w_e:
            index = (n_s_w_e[2][0], n_s_w_e[2][1])
            south[index] = solver.BoolVar(
                f'south_{index[0]}_{index[1]}'
            )
            trees_links[tree].append(south[index])

        if 3 in n_s_w_e:
            index = (n_s_w_e[3][0], n_s_w_e[3][1])
            west[index] = solver.BoolVar(
                f'west_{index[0]}_{index[1]}'
            )
            trees_links[tree].append(west[index])

    # Ограничения

    for i in range(size_n):
        for j in range(size_m):

            # Палатку нельзя разместить на дереве
            if (i, j) in lst_trees:
                solver.Add(tents[i, j] == 0)
                continue

            # Если рядом нет дерева — палатки быть не может
            found = False

            for tree in lst_trees:
                if abs(tree[0] - i) + abs(tree[1] - j) <= 1:
                    found = True
                    break

            if not found:
                solver.Add(tents[i, j] == 0)

    # Не более одной палатки в каждом квадрате 2x2
    for i in range(size_n - 1):
        for j in range(size_m - 1):
            tent_four_cells = get_neighbours(
                size_n,
                size_m,
                tents,
                i,
                j,
                k=0
            )

            solver.Add(sum(tent_four_cells) <= 1)

    # Ограничения по строкам
    for k in range(size_n):
        if row_constraints[k] != '':
            lst_vars = [
                tents[i, j]
                for i, j in tents
                if i == k
            ]

            solver.Add(
                sum(lst_vars) == row_constraints[k]
            )

    # Ограничения по столбцам
    for k in range(size_m):
        if col_constraints[k] != '':
            lst_vars = [
                tents[i, j]
                for i, j in tents
                if j == k
            ]

            solver.Add(
                sum(lst_vars) == col_constraints[k]
            )

    # К каждому дереву должна быть привязана ровно одна палатка
    for key in lst_trees:
        solver.Add(sum(trees_links[key]) == 1)

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
                solver.Add(
                    sum(lst_vars) == tents[i, j]
                )

    # Решение SCIP
    start_time = time.perf_counter()

    status = solver.Solve()

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    print('-' * 20)
    print(f"The task took {elapsed_time:.7f} seconds to complete.")
    print(f"SCIP status: {status}")
    print('-' * 20)

    if status not in (
        pywraplp.Solver.OPTIMAL,
        pywraplp.Solver.FEASIBLE
    ):
        print("SCIP не нашёл допустимого решения")
        return None

    # Получение результата
    result_grid = []
    lst_vars = []

    for i in range(size_n):
        row = []

        for j in range(size_m):
            value = int(round(tents[i, j].solution_value()))

            row.append(value)
            lst_vars.append((tents[i, j], value))

        result_grid.append(row)

    # Возвращаем деревья на поле
    for tree in lst_trees:
        result_grid[tree[0]][tree[1]] = TREE

    for k, v in north.items():
        lst_vars.append(
            (v, int(round(v.solution_value())))
        )

    for k, v in south.items():
        lst_vars.append(
            (v, int(round(v.solution_value())))
        )

    for k, v in west.items():
        lst_vars.append(
            (v, int(round(v.solution_value())))
        )

    for k, v in east.items():
        lst_vars.append(
            (v, int(round(v.solution_value())))
        )

    with open('values.txt', 'w', encoding='utf-8') as file:
        for var, value in lst_vars:
            file.write(f'{var.name()} = {value}\n')

    print_grid(
        result_grid,
        row_constraints,
        col_constraints,
        tent=1
    )

    return result_grid