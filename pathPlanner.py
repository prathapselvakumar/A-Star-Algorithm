# The main path planning function. Additional functions, classes,
# variables, etc. can be added to the file, but this function must
# always be defined with these arguments and must return a list of
# coordinates (col,row).
# DO NOT EDIT THIS FUNCTION DECLARATION
def do_a_star(grid, start, end, display_message):
    """
    A* path planning on a 2D occupancy grid.

    Grid format (as used by the provided GUI):
      - grid[col][row] == 1 : free cell
      - grid[col][row] == 0 : obstacle (wall)

    Motion model:
      - 4-connected moves only: up/down/left/right
      - Each move has cost 1, so g(n) is the number of steps so far

    Heuristic:
      - Euclidean distance to the goal (as required by the coursework)
        h(n) = sqrt((dx)^2 + (dy)^2)  -> implemented with **0.5 (no imports)

    Returns:
      - A list of (col,row) coordinates from start (first) to end (last).
      - Returns [] if no path is found or input is invalid.
    """

    # ---- Basic safety checks (prevents GUI crashes) ----
    if grid is None or len(grid) == 0 or len(grid[0]) == 0:
        display_message("Invalid grid", "ERROR")
        return []

    COL = len(grid)
    ROW = len(grid[0])

    sx, sy = start
    ex, ey = end

    # Check start/end inside grid
    if sx < 0 or sx >= COL or sy < 0 or sy >= ROW:
        display_message("Start is outside the grid", "ERROR")
        return []
    if ex < 0 or ex >= COL or ey < 0 or ey >= ROW:
        display_message("End is outside the grid", "ERROR")
        return []

    # Check start/end are not obstacles
    if grid[sx][sy] == 0:
        display_message("Start is on an obstacle", "ERROR")
        return []
    if grid[ex][ey] == 0:
        display_message("End is on an obstacle", "ERROR")
        return []

    # Trivial case
    if start == end:
        return [start]

    # ---- Heuristic: Euclidean distance ----
    def heuristic(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    # ---- A* data structures ----
    open_list = [start]          # discovered nodes not yet expanded
    closed_set = set()           # expanded nodes (use set for speed)
    parent = {}                  # parent pointers for path reconstruction
    g_cost = {start: 0}          # cost from start to node
    f_cost = {start: heuristic(start, end)}  # estimated total cost

    display_message("Starting A* search", "DEBUG")

    # ---- Main A* loop ----
    while len(open_list) > 0:

        # Select node in open_list with smallest f(n)
        current = open_list[0]
        best_f = f_cost[current]
        for node in open_list[1:]:
            fn = f_cost[node]
            if fn < best_f:
                best_f = fn
                current = node

        # Goal check
        if current == end:
            display_message("Goal reached", "DEBUG")

            # Reconstruct path by following parents from goal back to start
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            path.reverse()  # start -> goal
            return path

        # Move current from open to closed
        open_list.remove(current)
        closed_set.add(current)

        cx, cy = current
        # 4-connected neighbours only
        neighbours = (
            (cx + 1, cy),
            (cx - 1, cy),
            (cx, cy + 1),
            (cx, cy - 1),
        )

        for nx, ny in neighbours:

            # Bounds check
            if nx < 0 or nx >= COL or ny < 0 or ny >= ROW:
                continue

            # Obstacle check
            if grid[nx][ny] == 0:
                continue

            neighbour = (nx, ny)

            # Skip if already expanded
            if neighbour in closed_set:
                continue

            # Tentative g cost via current (each move cost = 1)
            tentative_g = g_cost[current] + 1

            # If neighbour is new, or we found a better route to it:
            old_g = g_cost.get(neighbour)
            if (old_g is None) or (tentative_g < old_g):
                parent[neighbour] = current
                g_cost[neighbour] = tentative_g
                f_cost[neighbour] = tentative_g + heuristic(neighbour, end)

                if neighbour not in open_list:
                    open_list.append(neighbour)

    # If we empty the open_list, no path exists
    display_message("No path found", "WARN")
    return []

# end of file