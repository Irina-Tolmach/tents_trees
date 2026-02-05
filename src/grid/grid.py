import numpy as np
import random

_EMPTY = 0
_TENT = 2
_TREE = 3

_FOUR_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_EIGHT_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
                     (0, 1), (1, 1), (1, 0),
                     (1, -1), (0, -1)]


class BadGenerationException(Exception):
    pass


class GridOptim:
    def __init__(self, n: int, m: int, difficulty: str):
        self.size_n = n
        self.size_m = m
        self.grid = np.zeros((n, m), dtype=int)
        self.links = {}  # дерево -> палатка
        self.empty_cells = {(x, y) for x in range(n) for y in range(m)}
        self._cache_all_neighbours()

        if difficulty == 'easy':
            ratios = np.arange(0.19, 0.17, -0.01)
        elif difficulty == 'medium':
            ratios = np.arange(0.22, 0.17, -0.01)
        else:
             ratios = np.arange(0.25, 0.17, -0.01)

        for ratio in ratios:
            self.num_trees = int(ratio * n * m)
            self._reset_grid()
            success = self._place_tents_and_trees()
            if success and self._validate_grid():
                break
        else:
            raise BadGenerationException("Невозможно сгенерировать даже при низкой плотности.")

        self.row_constraints, self.col_constraints = self._get_row_col_constraints()
        print_grid(self.grid, self.row_constraints, self.col_constraints)
        self.grid_init = self.grid.copy()
        self._remove_tents()

    def _reset_grid(self):
        self.grid.fill(_EMPTY)
        self.links.clear()
        self.empty_cells = {(x, y) for x in range(self.size_n) for y in range(self.size_m)}

    def _cache_all_neighbours(self):
        self.cached_4 = {}
        self.cached_8 = {}
        for x in range(self.size_n):
            for y in range(self.size_m):
                self.cached_4[(x, y)] = self._compute_neighbours(x, y, _FOUR_NEIGHBOURS)
                self.cached_8[(x, y)] = self._compute_neighbours(x, y, _EIGHT_NEIGHBOURS)

    def _compute_neighbours(self, x, y, directions):
        neighbours = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size_n and 0 <= ny < self.size_m:
                neighbours.append((nx, ny))
        return neighbours

    def _place_tents_and_trees(self):
        trees_placed = 0
        pairs = self._generate_valid_pairs()
        random.shuffle(pairs)

        used_tents = set()

        for (x, y), (nx, ny) in pairs:
            if trees_placed >= self.num_trees:
                return True
            if (x, y) not in self.empty_cells or (nx, ny) not in self.empty_cells:
                continue
            if (nx, ny) in used_tents:
                continue
            if self._has_adjacent_tent(nx, ny):
                continue

            self.grid[x][y] = _TREE
            self.grid[nx][ny] = _TENT
            self.links[(x, y)] = (nx, ny)
            self.empty_cells.discard((x, y))
            self.empty_cells.discard((nx, ny))
            used_tents.add((nx, ny))
            trees_placed += 1

        return trees_placed >= self.num_trees

    def _generate_valid_pairs(self):
        pairs = []
        for x, y in self.empty_cells:
            for nx, ny in self.cached_4[(x, y)]:
                if (nx, ny) in self.empty_cells and not self._has_adjacent_tent(nx, ny):
                    pairs.append(((x, y), (nx, ny)))
        return pairs

    def _has_adjacent_tent(self, x, y):
        for nx, ny in self.cached_8[(x, y)]:
            if self.grid[nx][ny] == _TENT:
                return True
        return False

    def _get_row_col_constraints(self):
        row_constraints = [int(np.sum(row == _TENT)) for row in self.grid]
        col_constraints = [int(np.sum(col == _TENT)) for col in self.grid.T]
        return row_constraints, col_constraints

    def _remove_tents(self):
        for x, y in self.links.values():
            self.grid[x][y] = _EMPTY

    def _validate_grid(self):
        # Проверим, что каждая палатка привязана к ровно одному дереву и наоборот
        for tree, tent in self.links.items():
            if self.grid[tree[0]][tree[1]] != _TREE:
                return False
            if self.grid[tent[0]][tent[1]] != _TENT:
                return False
            # Проверим, что палатка рядом по 4 сторонам с деревом
            if tent not in self.cached_4[tree]:
                return False
            # Проверим, что палатка не касается других палаток по диагонали
            for nx, ny in self.cached_8[tent]:
                if (nx, ny) != tree and self.grid[nx][ny] == _TENT:
                    return False

        # Проверим, что палатка не используется более одного раза
        seen_tents = set()
        for tent in self.links.values():
            if tent in seen_tents:
                return False
            seen_tents.add(tent)

        return True



def print_grid(grid, row_constraints, col_constraints, tent=_TENT):
    n = len(grid)
    m = len(grid[0])
    print(' ' * 6, end='')
    print(*col_constraints, sep=' ' * 2)
    print('_' * (n * 4))

    for row in range(n):
        print(f'{row_constraints[row]} |', sep='', end=' ')
        for col in range(m):
            if grid[row][col] == tent:
                print("⛺", sep='', end=' ')
            elif grid[row][col] == _TREE:
                print("🌲", sep='', end=' ')
            else:
                print("◻️", sep='', end=' ')
        print()

