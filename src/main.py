import json
import os
import time
import numpy as np

import statistics
from src.algorithms.backtracking_solver import BacktrackSolver
from src.algorithms.ilp_solver import ilp_solver
from src.algorithms.metaheuristics import Metaheuristics
from src.grid.grid import GridOptim


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # время начала
        result = func(*args, **kwargs)
        end_time = time.perf_counter()  # время окончания
        elapsed_time = end_time - start_time
        print(f"The task {func.__name__} took {elapsed_time:.7f} seconds to complete.")
        return result, elapsed_time
        # return result

    return wrapper


def main():
#    test_algorithms(100, 10, 10)
#     for i in range(10, 15):
    test_algorithms(0, 10, 10, from_file=True)


def save_grid(filename, grid, row_constraints, col_constraints):
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    new_entry = {
        "grid": grid.tolist(),
        "row_constraints": row_constraints,
        "col_constraints": col_constraints,
    }

    if not os.path.exists(filename):
        data = []
    else:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []

    data.append(new_entry)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_algorithm_times(filename, name, times, cnt_solve, n, evaluate):
    avg = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

    new_entry = {
        "name": name,
        "min_time": min(times),
        "max_time": max(times),
        "avg_time": sum(times) / n,
        "percent_solve": None if name == "ilp" else cnt_solve / n,
        "time_summary": f"{avg:.4f} ± {std_dev:.4f} секунд",
        "times": times,
        "scores": evaluate
    }

    # Если файла нет — создаём список
    if not os.path.exists(filename):
        data = []
    else:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []

    data.append(new_entry)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def test_algorithms(n, size_n, size_m, from_file = False, test_dir = "src/test"):
    ilp_times = []

    back_times = []
    back_cnt_solve = 0

    ls_times = []
    ls_cnt_solve = 0
    ls_evaluate = []

    sa_times = []
    sa_cnt_solve = 0
    sa_evaluate = []

    tabu_times = []
    tabu_cnt_solve = 0
    tabu_evaluate = []

    filename = f'{size_n}x{size_m}.json'

    tasks_path = os.path.join(test_dir, filename)
    if from_file:
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            tasks = [tasks]
        n = len(tasks)  # игнорируем n из аргументов, работаем по факту файла
    else:
        tasks = None

    for idx in range(n):
        if from_file:
            t = tasks[idx]
            grid = t["grid"]  # list[list[int]]
            row_constraints = t["row_constraints"]
            col_constraints = t["col_constraints"]
        else:
            grid_obj = GridOptim(size_n, size_m, "easy")
            grid, row_constraints, col_constraints = (
                grid_obj.grid, grid_obj.row_constraints, grid_obj.col_constraints
            )
            save_grid(tasks_path, grid, row_constraints, col_constraints)
            grid = grid.tolist()

        grid_copy = [row[:] for row in grid]

        # ilp_time = ilp([row[:] for row in grid_copy], row_constraints[:], col_constraints[:])
        # ilp_times.append(ilp_time[1])
        #
        # boo, back_time = backtrack([row[:] for row in grid_copy], row_constraints[:], col_constraints[:])
        # back_times.append(back_time)
        # if boo:
        #     back_cnt_solve += 1
        #
        # ls_eva, ls_time = local_search([row[:] for row in grid_copy], row_constraints[:], col_constraints[:])
        # ls_times.append(ls_time)
        # ls_evaluate.append(ls_eva)
        # if ls_eva == 0:
        #     ls_cnt_solve += 1

        sa_eva, sa_time = annealing([row[:] for row in grid_copy], row_constraints[:], col_constraints[:])
        sa_times.append(sa_time)
        sa_evaluate.append(sa_eva)
        if sa_eva == 0:
            sa_cnt_solve += 1

        # tabu_eva, tabu_time = tabu([row[:] for row in grid_copy], row_constraints[:], col_constraints[:])
        # tabu_times.append(tabu_time)
        # tabu_evaluate.append(tabu_eva)
        # if tabu_eva == 0:
        #     tabu_cnt_solve += 1

        print(f"Done: {filename} task {idx + 1}/{n}")


    # for _ in range(n):
    #     grid = GridOptim(size_n, size_m, "easy")
    #     grid, row_constraints, col_constraints = grid.grid, grid.row_constraints, grid.col_constraints
    #     save_grid(f'src/test/{filename}', grid, row_constraints, col_constraints)
    #
    #     ilp_time = ilp(grid.copy(), row_constraints[:], col_constraints[:])
    #     ilp_times.append(ilp_time[1])
    #
    #     boo, back_time = backtrack(grid.copy(), row_constraints[:], col_constraints[:])
    #     back_times.append(back_time)
    #     if boo:
    #         back_cnt_solve += 1
    #
    #     ls_eva, ls_time = local_search(grid.copy(), row_constraints[:], col_constraints[:])
    #     ls_times.append(ls_time)
    #     ls_evaluate.append(ls_eva)
    #     if ls_eva == 0:
    #         ls_cnt_solve += 1
    #
    #     sa_eva, sa_time = annealing(grid.copy(), row_constraints[:], col_constraints[:])
    #     sa_times.append(sa_time)
    #     sa_evaluate.append(sa_eva)
    #     if sa_eva == 0:
    #         sa_cnt_solve += 1
    #
    #     tabu_eva, tabu_time = tabu(grid.copy(), row_constraints[:], col_constraints[:])
    #     tabu_times.append(tabu_time)
    #     tabu_evaluate.append(tabu_eva)
    #     if tabu_eva == 0:
    #         tabu_cnt_solve += 1

    # print_algorithm_times('ilp', ilp_times, None, n, [0] * n)
    # save_algorithm_times(f'src/results/result_old_{filename}','ilp', ilp_times, None, n, [0] * n)
    # print_algorithm_times('backtracking', back_times, back_cnt_solve, n, [0] * n)
    # save_algorithm_times(f'src/results/result_old_{filename}', 'backtracking', back_times, back_cnt_solve, n, [0] * n)
    # print_algorithm_times('local', ls_times, ls_cnt_solve, n, ls_evaluate)
    # save_algorithm_times(f'src/results/result_local_fine_6_{filename}', 'local', ls_times, ls_cnt_solve, n, ls_evaluate)
    print_algorithm_times('annealing', sa_times, sa_cnt_solve, n, sa_evaluate)
    save_algorithm_times(f'src/results/result_annealing_fine_6_second_{filename}', 'annealing', sa_times, sa_cnt_solve, n, sa_evaluate)
    # print_algorithm_times('tabu', tabu_times, tabu_cnt_solve, n, tabu_evaluate)
    # save_algorithm_times(f'src/results/result_tabu_fine_6_{filename}', 'tabu', tabu_times, tabu_cnt_solve, n, tabu_evaluate)


def print_algorithm_times(name, times, cnt_solve, n, evaluate):
    print('-' * 50)
    print(f'{name}')
    print(f'min_time = {min(times):.4f}')
    print(f'max_time = {max(times):.4f}')
    print(f'avg_time = {sum(times) / n:.4f}')

    if not name == 'ilp':
        print(f'percent_solve = {cnt_solve / n}')

    avg = statistics.mean(times)
    std_dev = statistics.stdev(times)  # стандартное отклонение

    print(f"time = {avg:.4f} ± {std_dev:.4f} секунд")

    print(f'times_{name} = {times}')
    print(f'scores_{name} = {evaluate}')


@timer
def ilp(grid, row_constraints, col_constraints):
    ilp_time = ilp_solver(grid, row_constraints, col_constraints)
    return ilp_time


@timer
def backtrack(grid, row_constraints, col_constraints):
    grid_np = np.array(grid, dtype=np.int8)
    solver_backtracking = BacktrackSolver(grid_np, row_constraints, col_constraints)
    boo = solver_backtracking.solve()
    return boo


@timer
def local_search(grid, row_constraints, col_constraints, random_init=True):
    local_search_solver = Metaheuristics(grid[:], row_constraints, col_constraints)
    res = (
        local_search_solver.solve(row_constraints, col_constraints, 100, method="local"))
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]

    return best_score


@timer
def annealing(grid, row_constraints, col_constraints):
    annealing_search_solver = Metaheuristics(grid[:], row_constraints, col_constraints)
    res = \
        annealing_search_solver.solve(row_constraints, col_constraints, 150, method="annealing")
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]

    return best_score


@timer
def tabu(grid, row_constraints, col_constraints):
    tabu_search_solver = Metaheuristics(grid[:], row_constraints, col_constraints, max_iters=500)
    res = (
        tabu_search_solver.solve(row_constraints, col_constraints, 100, method="tabu"))
    tents, best_score, max_score, eva, cnt_restarts = res[0], res[1], res[2], res[3], res[4]

    return best_score


def print_init_grid(grid, row_constraints, col_constraints):
    grid2 = []
    print('{')
    print('"grid":', end='')
    for i in range(len(grid)):
        grid2.append([])
        for j in range(len(grid)):
            grid2[i].append(int(grid[i, j]))

    print('[')
    print(*grid2, sep=',\n')
    print('],')
    print(f'"row_constraints": {row_constraints},')
    print(f'"col_constraints": {col_constraints}')
    print('},')


if __name__ == '__main__':
    main()
