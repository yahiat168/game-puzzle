
import heapq
import time
from collections import deque
# Maps (one string per row)

MAP_A = [
    "S..#....",
    "##.#.##.",
    "....~~..",
    ".####.#.",
    "...~..#G",
]

MAP_B = [
    "S~~~~~~~G",
    ".#######.",
    ".........",
    ".#.#.#.#.",
    ".........",
]

MAP_C = [
    "S..#....",
    ".#.#.##.",
    "...#..#.",
    ".###.##.",
    "....#..G",
]

GRID = MAP_A

CELL_COST = {".": 1, "S": 1, "G": 1, "~": 3}
WALL = "#"

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Problem helpers
def find_start_goal(grid):
    """Scan the grid and return the (row, col) of S and G."""
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Map must contain exactly one S and one G")
    return start, goal


def neighbors(grid, cell):
    """Yield (next_cell, step_cost) for every legal move from cell."""
    r, c = cell
    rows, cols = len(grid), len(grid[0])
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
            yield (nr, nc), CELL_COST[grid[nr][nc]]


def manhattan(a, b):
    """Manhattan distance. Admissible here because every move costs >= 1."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct(parent, goal):
    """Follow parent links back from the goal to build the path S -> G."""
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def path_cost(grid, path):
    """Total cost of a path (cost of every cell entered; S not counted)."""
    return sum(CELL_COST[grid[r][c]] for r, c in path[1:])


def result(name, path, grid, expanded, max_frontier):
    """Package the outcome of one search in a dictionary."""
    return {
        "algorithm": name,
        "path": path,
        "cost": path_cost(grid, path) if path else None,
        "length": len(path) - 1 if path else None,  # number of moves
        "expanded": expanded,
        "max_frontier": max_frontier,
    }

# Uninformed search

def bfs(grid):
    """Breadth-First Search: fewest MOVES, ignores cell costs."""
    start, goal = find_start_goal(grid)
    frontier = deque([start])
    parent = {start: None}          
    expanded, max_frontier = 0, 1

    while frontier:
        cell = frontier.popleft()
        expanded += 1
        if cell == goal:
            return result("BFS", reconstruct(parent, goal), grid, expanded, max_frontier)
        for nxt, _ in neighbors(grid, cell):
            if nxt not in parent:
                parent[nxt] = cell
                frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))
    return result("BFS", None, grid, expanded, max_frontier)


def dfs(grid):
    """Depth-First Search: follows one branch as deep as possible before
    backtracking. Ignores cell costs; the path found is not guaranteed to be
    the shortest or the cheapest."""
    start, goal = find_start_goal(grid)
    frontier = [start]             
    parent = {start: None}
    visited = set()               
    expanded, max_frontier = 0, 1

    while frontier:
        cell = frontier.pop()
        if cell in visited:        
            continue
        visited.add(cell)
        expanded += 1
        if cell == goal:
            return result("DFS", reconstruct(parent, goal), grid, expanded, max_frontier)
        
        for nxt, _ in reversed(list(neighbors(grid, cell))):
            if nxt not in visited:
                parent[nxt] = cell  
                frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))
    return result("DFS", None, grid, expanded, max_frontier)


# Informed search

def greedy(grid):
    """Greedy Best-First: expands the node with smallest h(n) only."""
    start, goal = find_start_goal(grid)
    tie = 0
    frontier = [(manhattan(start, goal), tie, start)]
    parent = {start: None}
    closed = set()
    expanded, max_frontier = 0, 1

    while frontier:
        _, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue
        closed.add(cell)
        expanded += 1
        if cell == goal:
            return result("Greedy Best-First", reconstruct(parent, goal), grid, expanded, max_frontier)
        for nxt, _ in neighbors(grid, cell):
            if nxt not in closed and nxt not in parent:
                parent[nxt] = cell
                tie += 1
                heapq.heappush(frontier, (manhattan(nxt, goal), tie, nxt))
        max_frontier = max(max_frontier, len(frontier))
    return result("Greedy Best-First", None, grid, expanded, max_frontier)


ALGORITHMS = [bfs, dfs, greedy]


# Output
def draw_path(grid, path):
    """Return the map as text with path cells (except S and G) marked '*'."""
    rows = [list(row) for row in grid]
    if path:
        for r, c in path[1:-1]:
            rows[r][c] = "*"
    return "\n".join("".join(row) for row in rows)


def print_result(res, grid):
    print(f"Algorithm      : {res['algorithm']}")
    if res["path"] is None:
        print("No path found")
        print(f"Nodes expanded : {res['expanded']}")
    else:
        print(f"Path           : {res['path']}")
        print(f"Path length    : {res['length']} moves")
        print(f"Total cost     : {res['cost']}")
        print(f"Nodes expanded : {res['expanded']}")
        print(draw_path(grid, res["path"]))
    print("-" * 60)


def timed(algo, grid, repeats=200):
    """Run an algorithm several times and return (result, average ms)."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        res = algo(grid)
    return res, (time.perf_counter() - t0) * 1000 / repeats


def results_table():
    maps = {"MAP_A": MAP_A, "MAP_B": MAP_B, "MAP_C": MAP_C}
    header = f"{'Map':<6} {'Algorithm':<18} {'Length':>6} {'Cost':>5} {'Expanded':>8} {'MaxFront':>8} {'Time(ms)':>9}"
    print(header)
    print("=" * len(header))
    for name, grid in maps.items():
        for algo in ALGORITHMS:
            res, ms = timed(algo, grid)
            length = res["length"] if res["path"] else "-"
            cost = res["cost"] if res["path"] else "-"
            print(f"{name:<6} {res['algorithm']:<18} {length!s:>6} {cost!s:>5} "
                  f"{res['expanded']:>8} {res['max_frontier']:>8} {ms:>9.4f}")
        print("-" * len(header))
    print("'-' = No path found")


if __name__ == "__main__":
    print("=" * 40)
    print("Map in GRID:")
    print("\n".join(GRID))
    print("=" * 40)
    for algo in ALGORITHMS:
        print_result(algo(GRID), GRID)

    print("\nRESULTS TABLE (all maps)\n")
    results_table()
