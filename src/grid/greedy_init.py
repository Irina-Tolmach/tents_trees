import random

EMPTY: int = 0
GRASS: int = 1
TENT: int = 2
TREE: int = 3

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # вверх, вниз, влево, вправо
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


class GreedyInitializer:
    def __init__(self, grid, row_constraints, col_constraints):
        self.grid = [row[:] for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.row_constraints = row_constraints[:]
        self.col_constraints = col_constraints[:]
        self.trees = [(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r][c] == TREE]
        self.tents = set()
        self.links = {
            True: {},
            False: set(self.trees)
        }

    def is_valid_cell(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_valid_candidates(self, tree):
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

    def get_candidates(self, tree):
        r, c = tree
        candidates = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not self.is_valid_cell(nr, nc):
                continue
            if self.grid[nr][nc] != EMPTY:
                continue
            candidates.append((nr, nc))
        return candidates

    def place_tent(self, pos, tree):
        self.links[False].discard(tree)
        self.links[True][tree] = pos
        x, y = pos
        self.tents.add(pos)
        self.grid[x][y] = TENT
        self.row_constraints[x] -= 1
        self.col_constraints[y] -= 1

    def initialize(self):
        remaining = [t for t in self.links[False]]
        # сортировка по числу кандидатов (эвристика MRV)
        candidates_map = {tree: self.get_valid_candidates(tree) for tree in remaining}
        trees = sorted(remaining, key=lambda t: len(candidates_map[t]))
        self.links = {
            True: {},
            False: set(self.trees)
        }

        for tree in trees:
            candidates = self.get_valid_candidates(tree)

            if candidates:
                pos = random.choice(candidates)
                self.place_tent(pos, tree)
            else:
                neighbours = self.get_candidates(tree)
                if neighbours:
                    pos = random.choice(neighbours)
                    self.place_tent(pos, tree)
                else:
                    return None, None

        trees = []
        tents = []

        for tree, tent in self.links[True].items():
            trees.append(tree)
            tents.append(tent)

        return trees, tents

