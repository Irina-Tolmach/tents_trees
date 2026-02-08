import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    #path = Path("src/benchmarks/results/tabu_5_set_stable_seed.json")
    #path = Path("src/benchmarks/results/tabu_6_dict_stable_seed.json")
    #path = Path("src/benchmarks/results/tabu_7_dict_score_stable_seed.json")
    #path = Path("src/benchmarks/results/tabu_8_orig_2score_stable_seed.json")
    #path = Path("src/benchmarks/results/tabu_4_origlist_stable_seed.json")
    path = Path("src/benchmarks/results/test_annealing_new_6.json")
    data = json.load(open(path, encoding="utf-8"))

    runs = data["runs"]

    # группировка: (size, method)
    grouped = defaultdict(list)
    for r in runs:
        key = (r["size"], r["method"])
        grouped[key].append(r)

    sizes = sorted({s for s, _ in grouped.keys()})
    methods = sorted({m for _, m in grouped.keys()})

    print("BASELINE SUMMARY")
    print("=" * 60)

    for size in sizes:
        print(f"\nSize {size}x{size}")
        print("-" * 60)

        for method in methods:
            items = grouped[(size, method)]

            solved = [r for r in items if r["solved"]]
            unsolved = [r for r in items if not r["solved"]]

            success_rate = len(solved) / len(items) * 100

            if solved:
                times = [r["time_sec"] for r in solved]
                med_time = statistics.median(times)
            else:
                med_time = None

            if unsolved:
                scores = [r["best_score"] for r in unsolved]
                med_score = statistics.median(scores)
            else:
                med_score = 0

            print(
                f"{method:10s} | "
                f"solved: {success_rate:6.1f}% | "
                f"median time: {med_time if med_time is not None else '-':>8} | "
                f"median score (unsolved): {med_score}"
            )


if __name__ == "__main__":
    main()
