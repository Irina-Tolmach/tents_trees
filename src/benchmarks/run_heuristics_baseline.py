import json
import time
from pathlib import Path
import random

from src.algorithms.metaheuristics import Metaheuristics


def run_one(method_name: str, grid, row, col, restarts: int = 10):
    solver = Metaheuristics(grid, row, col)

    t0 = time.perf_counter()
    solution, best_score, max_score, eva, attempts = solver.solve(row, col, restarts=restarts, method=method_name)
    t1 = time.perf_counter()

    return {
        "method": method_name,
        "time_sec": t1 - t0,
        "best_score": best_score,
        "eva": eva,
        "attempts": attempts,
        "solved": (best_score == 0),
        "max_score": max_score,
    }


def main():
    test_dir = Path("src/test")
    out_dir = Path("src/benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    #out_path = out_dir / "tabu_5_set_stable_seed.json"
    #out_path = out_dir / "tabu_6_dict_stable_seed.json"
    #out_path = out_dir / "tabu_7_dict_score_stable_seed.json"
    #out_path = out_dir / "tabu_8_orig_2score_stable_seed.json"
    #out_path = out_dir / "tabu_4_origlist_stable_seed.json"
    #out_path = out_dir / "tabu_10_1_iter.json"
    #out_path = out_dir / "tabu_10_2_iter.json"
    #out_path = out_dir / "anneal_3.json"
    out_path = out_dir / "test_annealing_new_6.json"
    #methods = ["local", "annealing", "tabu"]
    methods = ["annealing"]
    restarts = 10

    results = {
        "meta": {
            "methods": methods,
            "restarts": restarts,
            "source_dir": str(test_dir),
        },
        "runs": []
    }

    files = sorted(test_dir.glob("*x*.json"), key=lambda p: int(p.stem.split("x")[0]))

    for file in files:
        size = int(file.stem.split("x")[0])
        tasks = json.load(open(file, encoding="utf-8"))

        for i, task in enumerate(tasks):
            grid = task["grid"]
            row = task["row_constraints"]
            col = task["col_constraints"]

            for m in methods:
                seed = (size * 1_000_003 + i * 10_007 + restarts) & 0xffffffff
                random.seed(seed)
                r = run_one(m, grid, row, col, restarts=restarts)
                r.update({"size": size, "task_index": i, "file": file.name})
                results["runs"].append(r)

            print(f"Done: {file.name} task {i+1}/{len(tasks)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
