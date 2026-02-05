EMPTY: int = 0
GRASS: int = 1
TENT: int = 2
TREE: int = 3
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # вверх, вниз, влево, вправо
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

class BacktrackSolver:
    def __init__(self, grid, row_constraints, col_constraints):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.row_constraints = row_constraints[:]
        self.col_constraints = col_constraints[:]
        self.trees = {(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r][c] == TREE}
        self.tents = set()
        self.links = {
            True: set(),
            False: set(self.trees)
        }

    def is_valid_cell(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_candidates(self, tree):
        r, c = tree
        candidates = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not self.is_valid_cell(nr, nc):
                continue
            if self.grid[nr][nc] != EMPTY:
                continue
            if self.row_constraints[nr] <= 0 or self.col_constraints[nc] <= 0:
                continue
            if any((nr + dr2, nc + dc2) in self.tents for dr2, dc2 in NEIGHBORS):
                continue
            candidates.append((nr, nc))
        return candidates

    def place_tent(self, pos, tree):
        self.links[False].discard(tree)
        self.links[True].add(tree)
        x, y = pos
        self.tents.add(pos)
        self.grid[x][y] = TENT
        self.row_constraints[x] -= 1
        self.col_constraints[y] -= 1

        for r, c in NEIGHBORS:
            nr, nc = r + x, c + y
            if self.is_valid_cell(nr, nc) and self.grid[nr][nc] == EMPTY:
                self.grid[nr][nc] = GRASS

    def remove_tent(self, pos, tree):
        self.links[True].discard(tree)
        self.links[False].add(tree)
        x, y = pos
        self.tents.remove(pos)
        self.grid[x][y] = EMPTY
        self.row_constraints[x] += 1
        self.col_constraints[y] += 1

        for r, c in NEIGHBORS:
            nr, nc = r + x, c + y
            if self.is_valid_cell(nr, nc) and self.grid[nr][nc] == GRASS:
                self.grid[nr, nc] = EMPTY

    def solve(self):
        res = self._backtrack()
        if res:
            return 'True'
        else:
            return 'False'
        # return self._backtrack()

    def _backtrack(self):
        if not self.links[False]:
            return True

        remaining = [t for t in self.links[False]]
        # сортировка по числу кандидатов (эвристика MRV)
        candidates_map = {tree: self.get_candidates(tree) for tree in remaining}
        sorted_trees = sorted(remaining, key=lambda t: len(candidates_map[t]))

        tree = sorted_trees[0]
        candidates = candidates_map[tree]

        if not candidates:
            return False

        for pos in candidates:
            self.place_tent(pos, tree)
            if self._backtrack():
                return True
            self.remove_tent(pos, tree)

        return False