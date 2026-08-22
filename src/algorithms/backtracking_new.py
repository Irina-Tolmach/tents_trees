"""
Ускоренный поиск с возвратом (MRV) для головоломки «Палатки и деревья».

Та же идея, что в исходном BacktrackSolver (DFS + эвристика MRV),
но без лишней работы в каждом узле:

  1. Вместо пересчёта кандидатов для всех деревьев + сортировки — один
     проход с поиском минимума и ранним обрывом:
       - дерево с 0 кандидатов  -> немедленный откат (return False);
       - дерево с 1 кандидатом  -> вынужденный ход, скан прекращается.
  2. Проверка «клетка не соседствует с палаткой» за O(1) через счётчик
     adj[r][c] (число палаток, накрывающих клетку: сама клетка + 8 соседей),
     вместо 8 обращений к set на каждую клетку.
  3. Статичные клетки-кандидаты каждого дерева предвычислены один раз.
  4. Входной grid не мутируется (GRASS-пометки не нужны);
     заодно устранён баг grid[nr, nc] в remove_tent исходной версии.

Интерфейс совместим: BacktrackSolver(grid, row_constraints, col_constraints),
solve() -> 'True'/'False'. Найденное решение доступно в self.solution
(множество клеток с палатками) и self.tents.
"""

import sys

EMPTY: int = 0
GRASS: int = 1
TENT: int = 2
TREE: int = 3
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


class FastBacktrackSolver:
    def __init__(self, grid, row_constraints, col_constraints):
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.row_rem = list(row_constraints)
        self.col_rem = list(col_constraints)

        g = [list(row) for row in grid]
        self.trees = [(r, c) for r in range(self.rows)
                      for c in range(self.cols) if g[r][c] == TREE]

        # статичные клетки-кандидаты каждого дерева: в границах, не дерево
        self.tree_cells = []
        for (r, c) in self.trees:
            cells = []
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols \
                        and g[nr][nc] != TREE:
                    cells.append((nr, nc))
            self.tree_cells.append(cells)

        # adj[r][c] > 0  <=>  клетка занята палаткой или соседствует с ней
        self.adj = [[0] * self.cols for _ in range(self.rows)]
        self.tents = set()
        self.unplaced = set(range(len(self.trees)))
        self.solution = None
        self.nodes = 0  # статистика: число раскрытых узлов

    # ------------------------------------------------------------------

    def _candidates(self, i):
        rr = self.row_rem
        cr = self.col_rem
        adj = self.adj
        out = []
        for cell in self.tree_cells[i]:
            r, c = cell
            if adj[r][c] == 0 and rr[r] > 0 and cr[c] > 0:
                out.append(cell)
        return out

    def _place(self, i, cell):
        r, c = cell
        self.tents.add(cell)
        self.row_rem[r] -= 1
        self.col_rem[c] -= 1
        adj = self.adj
        adj[r][c] += 1
        rows, cols = self.rows, self.cols
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                adj[nr][nc] += 1
        self.unplaced.discard(i)

    def _remove(self, i, cell):
        r, c = cell
        self.tents.discard(cell)
        self.row_rem[r] += 1
        self.col_rem[c] += 1
        adj = self.adj
        adj[r][c] -= 1
        rows, cols = self.rows, self.cols
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                adj[nr][nc] -= 1
        self.unplaced.add(i)

    # ------------------------------------------------------------------

    def _backtrack(self):
        if not self.unplaced:
            self.solution = set(self.tents)
            return True

        self.nodes += 1

        # MRV: один проход, ранний обрыв на 0 (тупик) и на 1 (вынужденный ход)
        best_i = -1
        best_cands = None
        best_len = 5
        for i in self.unplaced:
            cs = self._candidates(i)
            k = len(cs)
            if k == 0:
                return False
            if k < best_len:
                best_len = k
                best_i = i
                best_cands = cs
                if k == 1:
                    break

        for cell in best_cands:
            self._place(best_i, cell)
            if self._backtrack():
                return True
            self._remove(best_i, cell)

        return False

    def solve(self):
        depth = len(self.trees) + 100
        if sys.getrecursionlimit() < depth * 2:
            sys.setrecursionlimit(depth * 2)
        return 'True' if self._backtrack() else 'False'
