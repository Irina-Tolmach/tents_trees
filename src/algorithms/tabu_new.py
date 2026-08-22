"""
Ускоренный Tabu Search для головоломки «Палатки и деревья».

Ключевые отличия от исходной версии:
  1. Дельта-оценка хода за O(1): без копий row/col_counts и без set(tents).
  2. Множество занятых клеток (tent_set) поддерживается инкрементально.
  3. Клетки-кандидаты каждого дерева предвычислены один раз; занятость O(1).
  4. Табу-лист — словарь {(tree_idx, cell): iter_expire}, проверка O(1);
     корректная аспирация: табу-ход разрешён, если бьёт глобальный рекорд.
  5. Candidate list: сначала сканируются только «конфликтные» палатки,
     полный перебор — как fallback (качество не теряется).
  6. Препроцессинг: пропагация вынужденных ходов (деревья с единственной
     допустимой клеткой пиннятся, их окрестность блокируется; клетки в
     строках/столбцах с лимитом 0 или насыщенных пиннингом — выбрасываются).
  7. Мгновенный выход при нахождении хода с нулевым счётом.
  8. Kick (тёплый перезапуск) при стагнации вместо полного рестарта.

Интерфейс совместим с исходным классом Metaheuristics.
"""

import math
import random

EMPTY: int = 0
GRASS: int = 1
TENT: int = 2
TREE: int = 3
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


class FastTabu:
    def __init__(self, grid, row_limits, col_limits, max_iters=1000,
                 w_adj=2, w_line=1):
        self.grid = [list(row) for row in grid]
        self.n = len(self.grid)
        self.m = len(self.grid[0])
        self.row_limits = list(row_limits)
        self.col_limits = list(col_limits)
        self.max_iters = max_iters
        self.w_adj = w_adj
        self.w_line = w_line

        self.trees = [(r, c) for r in range(self.n) for c in range(self.m)
                      if self.grid[r][c] == TREE]

        # статистика (совместимо с исходным классом)
        self.eva = 0
        self.best_score = 0
        self.max_score = 0
        self.last_tents = []

        self._adjcache = {}

        # статичные клетки-кандидаты каждого дерева (без учёта занятости)
        raw_cells = []
        for (r, c) in self.trees:
            cells = []
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.n and 0 <= nc < self.m \
                        and self.grid[nr][nc] != TREE:
                    cells.append((nr, nc))
            raw_cells.append(cells)

        self.infeasible = False
        self.pinned = {}          # tree_idx -> закреплённая клетка
        self.free = []            # индексы деревьев, участвующих в поиске
        self.cands = []           # cands[i] — список клеток для дерева i
        self._preдаprocess(raw_cells)

    # ------------------------------------------------------------------ utils

    def _adj(self, pos):
        """Кэшированная 8-окрестность клетки."""
        a = self._adjcache.get(pos)
        if a is None:
            r, c = pos
            a = [(r + dr, c + dc) for dr, dc in NEIGHBORS]
            self._adjcache[pos] = a
        return a

    # ------------------------------------------------------------ preprocessing

    def _preprocess(self, raw_cells):
        """Пропагация вынужденных ходов до фикспоинта.

        Сужения, сохраняющие все корректные решения:
          - клетки в строках/столбцах с лимитом 0 недопустимы;
          - у дерева ровно одна допустимая клетка -> палатка пиннится,
            сама клетка и её 8-окрестность блокируются для остальных;
          - если пиннинг насытил лимит строки/столбца, остальные клетки
            этой линии блокируются.
        """
        T = len(self.trees)
        cand = []
        for cells in raw_cells:
            cand.append({(r, c) for (r, c) in cells
                         if self.row_limits[r] > 0 and self.col_limits[c] > 0})

        pinned = {}
        blocked = set()
        changed = True
        while changed and not self.infeasible:
            changed = False
            pr = [0] * self.n
            pc = [0] * self.m
            for (x, y) in pinned.values():
                pr[x] += 1
                pc[y] += 1
            for i in range(T):
                if i in pinned:
                    continue
                cs = cand[i]
                new_cs = {p for p in cs
                          if p not in blocked
                          and pr[p[0]] < self.row_limits[p[0]]
                          and pc[p[1]] < self.col_limits[p[1]]}
                if len(new_cs) != len(cs):
                    cand[i] = new_cs
                    changed = True
                    cs = new_cs
                if not cs:
                    self.infeasible = True
                    break
                if len(cs) == 1:
                    pos = next(iter(cs))
                    pinned[i] = pos
                    blocked.add(pos)
                    blocked.update(self._adj(pos))
                    changed = True

        self.pinned = pinned
        self.free = [i for i in range(T) if i not in pinned]
        self.cands = [None] * T
        for i in self.free:
            self.cands[i] = sorted(cand[i])

    # ------------------------------------------------------------------ scoring

    def _full_score(self, tents, tent_set, rc, cc):
        s = 0
        for pos in tents:
            for q in self._adj(pos):
                if q in tent_set:
                    s += 1
        s = (s // 2) * self.w_adj  # каждая пара посчитана дважды
        wl = self.w_line
        for i in range(self.n):
            d = rc[i] - self.row_limits[i]
            s += wl * d * d
        for j in range(self.m):
            d = cc[j] - self.col_limits[j]
            s += wl * d * d
        return s

    def _delta(self, old, new, tent_set, rc, cc):
        """Изменение счёта при переносе палатки old -> new. O(1)."""
        w = self.w_adj
        d = 0
        for p in self._adj(old):
            if p in tent_set:
                d -= w
        for q in self._adj(new):
            if q != old and q in tent_set:
                d += w
        x1, y1 = old
        x2, y2 = new
        wl2 = 2 * self.w_line
        if x1 != x2:
            a = rc[x1] - self.row_limits[x1]
            b = rc[x2] - self.row_limits[x2]
            d += wl2 * (1 - a + b)          # (a-1)^2-a^2 + (b+1)^2-b^2
        if y1 != y2:
            a = cc[y1] - self.col_limits[y1]
            b = cc[y2] - self.col_limits[y2]
            d += wl2 * (1 - a + b)
        return d

    # ------------------------------------------------------------------ init

    def _init_solution(self, rng):
        """Жадная min-conflict инициализация; пиннинги ставятся первыми."""
        T = len(self.trees)
        tents = [None] * T
        tent_set = {}
        for i, pos in self.pinned.items():
            tents[i] = pos
            tent_set[pos] = i

        order = sorted(self.free,
                       key=lambda i: (len(self.cands[i]), rng.random()))
        for i in order:
            opts = [p for p in self.cands[i] if p not in tent_set]
            if not opts:
                return None, None
            pos = min(opts, key=lambda p: (sum(1 for q in self._adj(p)
                                               if q in tent_set),
                                           rng.random()))
            tents[i] = pos
            tent_set[pos] = i
        return tents, tent_set

    # ------------------------------------------------------------------ search

    def _violating(self, tents, tent_set, rc, cc):
        out = []
        rl, cl = self.row_limits, self.col_limits
        for i in self.free:
            pos = tents[i]
            x, y = pos
            if rc[x] != rl[x] or cc[y] != cl[y]:
                out.append(i)
                continue
            for q in self._adj(pos):
                if q in tent_set:
                    out.append(i)
                    break
        return out

    def _select_move(self, tents, tent_set, rc, cc, tabu, it, cur, best, rng):
        viol = self._violating(tents, tent_set, rc, cc)
        rng.shuffle(viol)
        pools = [viol, self.free] if viol else [self.free]

        fallback = None
        for pool in pools:
            bm = None
            bd = None
            for i in pool:
                old = tents[i]
                for pos in self.cands[i]:
                    if pos == old or pos in tent_set:
                        continue
                    self.eva += 1
                    d = self._delta(old, pos, tent_set, rc, cc)
                    ns = cur + d
                    if ns == 0:
                        return (i, pos, d)          # мгновенный выход
                    if tabu.get((i, pos), 0) >= it and ns >= best:
                        continue                     # табу без аспирации
                    if bd is None or d < bd:
                        bd = d
                        bm = (i, pos, d)
            if bm is not None:
                if bd < 0 or pool is self.free:
                    return bm
                fallback = bm  # среди конфликтных нет улучшения — полный скан
        return fallback

    def _apply(self, mv, tents, tent_set, rc, cc):
        i, pos, d = mv
        old = tents[i]
        tents[i] = pos
        del tent_set[old]
        tent_set[pos] = i
        rc[old[0]] -= 1
        cc[old[1]] -= 1
        rc[pos[0]] += 1
        cc[pos[1]] += 1
        return old, d

    def _kick(self, k, tents, tent_set, rc, cc, cur, rng):
        for _ in range(k):
            i = rng.choice(self.free)
            opts = [p for p in self.cands[i]
                    if p not in tent_set and p != tents[i]]
            if not opts:
                continue
            pos = rng.choice(opts)
            d = self._delta(tents[i], pos, tent_set, rc, cc)
            self._apply((i, pos, d), tents, tent_set, rc, cc)
            cur += d
        return cur

    def tabu_search(self, row_limits=None, col_limits=None,
                    tabu_tenure=None, max_stagnation=250, seed=None):
        """Аргументы row/col_limits принимаются для совместимости и игнорируются
        (лимиты заданы в конструкторе)."""
        if self.infeasible:
            return None
        if not self.free:  # всё запиннено препроцессингом
            tents = [None] * len(self.trees)
            for i, pos in self.pinned.items():
                tents[i] = pos
            tent_set = {pos: i for i, pos in self.pinned.items()}
            rc = [0] * self.n
            cc = [0] * self.m
            for (x, y) in tents:
                rc[x] += 1
                cc[y] += 1
            self.best_score = self._full_score(tents, tent_set, rc, cc)
            self.max_score = self.best_score
            self.last_tents = tents[:]
            return tents if self.best_score == 0 else None

        rng = random.Random(seed)
        for _ in range(5):
            tents, tent_set = self._init_solution(rng)
            if tents is not None:
                break
        else:
            return None

        rc = [0] * self.n
        cc = [0] * self.m
        for (x, y) in tents:
            rc[x] += 1
            cc[y] += 1

        cur = self._full_score(tents, tent_set, rc, cc)
        best = cur
        best_tents = tents[:]
        self.max_score = cur

        T = len(self.free)
        tenure = tabu_tenure or max(8, math.isqrt(len(self.trees)) + 2)
        kick_k = max(2, T // 12)
        kick_period = max(20, max_stagnation // 3)
        limit = self.max_iters * max(1, len(self.trees))

        tabu = {}
        stag = 0
        it = 0
        while it < limit and best > 0 and stag < max_stagnation:
            it += 1
            mv = self._select_move(tents, tent_set, rc, cc,
                                   tabu, it, cur, best, rng)
            if mv is None:
                stag += 1
            else:
                old, d = self._apply(mv, tents, tent_set, rc, cc)
                cur += d
                tabu[(mv[0], old)] = it + tenure
                if cur < best:
                    best = cur
                    best_tents = tents[:]
                    stag = 0
                else:
                    stag += 1

            if stag and stag % kick_period == 0:
                cur = self._kick(kick_k, tents, tent_set, rc, cc, cur, rng)

            if len(tabu) > 50 * tenure:
                tabu = {k: v for k, v in tabu.items() if v >= it}

        self.best_score = best
        self.last_tents = best_tents[:]
        return best_tents

    # ------------------------------------------------------------------ solve

    def solve(self, row_limits=None, col_limits=None, restarts=10, seed=None):
        rng = random.Random(seed)
        best_global = float('inf')
        best_sol = None
        for attempt in range(restarts):
            sol = self.tabu_search(seed=rng.randrange(1 << 30))
            if sol is None:
                continue
            if self.best_score == 0:
                return sol, 0, self.max_score, self.eva, attempt
            if self.best_score < best_global:
                best_global = self.best_score
                best_sol = sol[:]
        return best_sol, best_global, self.max_score, self.eva, restarts
