# =============================================================================
# AERO60492 - Autonomous Mobile Robots  |  Coursework 2 - Path Planning
# File : pathPlanner.py
# Author :Prathap Selvakumar
# Student ID: 14354077
# =============================================================================
#
# IMPORTANT NOTE ON IMPORTS:
#   This file uses NO additional library imports, as required by the
#   coursework specification. Every data structure and algorithm is built
#   from Python's built-in types (list, dict, set, tuple) only.
#   Sub-functions defined with 'def' are permitted and are used below.

# ADDITIONAL NOTE:
# Use of AI//LLM : I have used AI language model (ChatGPT) to assist with improve grammer and Vocabulary in the comments and docstrings.
# The code logic and implementation is entirely my own work, and I have not used AI to generate any code or algorithmic content.
# -----------------------------------------------------------------------------
# BACKGROUND - PATH PLANNING IN ROBOT NAVIGATION  (Lecture 9+10, Slide 6)
# -----------------------------------------------------------------------------
# Robot navigation is split into four sub-problems:
#   (1) Localisation   - where is the robot within the map/frame?
#   (2) Mapping        - a virtual representation of the local environment
#   (3) Path Planning  - creating a planned route of motion  <- THIS FILE
#   (4) Motion Control - executing movement along the planned trajectory
#
# Path planning requires knowledge of:
#   * The start and goal locations       (from localisation)
#   * The obstacles in the workspace     (from the occupancy grid / mapping)
#   * What motions are permitted         (motion model - 4-connected here)
#
# -----------------------------------------------------------------------------
# THE A* ALGORITHM  (Hart, Nilsson & Raphael, IEEE TSSC, 1968)
# -----------------------------------------------------------------------------
# A* was first described in:
#   P. E. Hart, N. J. Nilsson and B. Raphael, "A Formal Basis for the
#   Heuristic Determination of Minimum Cost Paths," IEEE Transactions on
#   Systems Science and Cybernetics, vol. 4, no. 2, pp. 100-107, July 1968.
#   https://doi.org/10.1109/TSSC.1968.300136
#
# A* is an INFORMED SEARCH ALGORITHM - a goal-directed extension of
# Dijkstra's algorithm. Rather than exploring all nodes breadth-first,
# it uses a heuristic estimate h(n) to bias exploration towards the goal,
# dramatically reducing the number of nodes expanded in practice (slide 49).
#
# -----------------------------------------------------------------------------
# THE COST FUNCTION  f(n) = g(n) + h(n)  (Hart et al. 1968, Eq. 1)
# -----------------------------------------------------------------------------
#
#   f(n) = g(n) + h(n)
#
#   g(n) - ACTUAL COST of the cheapest path found so far from start to
#           node n, travelling only through free cells (obeying occupancy
#           and connectivity of the graph). Because all edge weights are
#           uniform (cost = 1 per move), g(n) equals the number of steps
#           taken from the start node to reach node n.
#           This is identical to the running cost used in Dijkstra's algorithm.
#
#   h(n) - ESTIMATED (HEURISTIC) COST from node n to the goal.
#           Does NOT need to obey graph connectivity - it is a straight-line
#           "as the crow flies" lower bound.
#           REQUIRED BY COURSEWORK: Euclidean distance to destination -
#               h(n) = sqrt( (col_n - col_goal)^2 + (row_n - row_goal)^2 )
#           Implemented as (dx*dx + dy*dy)**0.5 - no imports needed.
#
#   ADMISSIBILITY (Hart et al. 1968, Theorem 1):
#           A heuristic is ADMISSIBLE if h(n) <= h*(n) for all n, where
#           h*(n) is the true optimal cost from n to the goal. An admissible
#           heuristic guarantees that A* finds the OPTIMAL (minimum cost) path.
#           Euclidean distance is admissible on a unit-cost 4-connected grid
#           because the straight-line distance can never exceed the true
#           shortest grid path (you cannot travel in a straight line through
#           obstacles or diagonally).
#
#   CONSISTENCY (Hart et al. 1968, Section III-B):
#           A heuristic is CONSISTENT if for any node n and its successor m:
#               h(n) <= cost(n, m) + h(m)   (a triangle inequality)
#           Euclidean distance satisfies this because of the geometric
#           triangle inequality. Consistency implies admissibility, and also
#           means that once a node is CLOSED its g-cost is provably optimal
#           and it will never need to be reopened (Lemma 2, Hart et al. 1968).
#
# -----------------------------------------------------------------------------
# KEY DATA STRUCTURES  (Lecture 9+10, Slide 29)
# -----------------------------------------------------------------------------
#
# OPEN SET ("frontier"):
#   Nodes that have been DISCOVERED but not yet fully EXPANDED.
#   A node is in the open set when at least one of its neighbours has not
#   yet been visited. Only the start node is added initially.
#
#   IMPLEMENTATION - pure Python min-heap (no heapq import):
#   A list of [f_cost, node] pairs maintained as a binary min-heap.
#   The heap property means the smallest f(n) is always at index [0].
#   This gives O(log n) insertion (_heap_push) and O(log n) extraction
#   (_heap_pop), versus O(n) for a plain linear-scan list - important for
#   the EXECUTION SPEED mark on large grids.
#   A companion set (open_set_members) provides O(1) membership testing
#   since the heap itself cannot answer "is X in the heap?" efficiently.
#
# CLOSED SET ("interior"):
#   Nodes that have been fully EXPANDED (all neighbours inspected).
#   Implemented as a Python set for O(1) membership testing.
#   Because our heuristic is consistent (see above), a closed node's
#   g-cost is optimal and it never needs to be re-opened.
#
# PARENT DICT:
#   Maps each discovered node -> its predecessor on the best known path.
#   Used at termination to reconstruct the path by chaining pointers from
#   goal back to start, then reversing (slide 48).
#
# G_COST DICT:
#   Maps each discovered node -> best g(n) found so far.
#   g(start) = 0; unknown nodes are treated as having cost = infinity.
#
# -----------------------------------------------------------------------------
# ALGORITHM STEPS (Hart et al. 1968, Section II-A; Lecture 9+10 slides 29-48)
# -----------------------------------------------------------------------------
#  1. Initialise:
#       open_set <- {start},  g(start) <- 0,  f(start) <- h(start, goal)
#       closed_set <- {}
#
#  2. While open_set is not empty:
#       a. current <- node in open_set with lowest f(n)  [O(log n) with heap]
#       b. If current == goal:
#            -> trace parent pointers goal->start, reverse, RETURN path
#       c. Move current: open_set -> closed_set
#       d. For each 4-connected neighbour m of current (free & in-bounds):
#            - Skip if m in closed_set  (already expanded optimally)
#            - tentative_g <- g(current) + 1
#            - If tentative_g < g(m):
#                  g(m)      <- tentative_g
#                  f(m)      <- tentative_g + h(m, goal)
#                  parent(m) <- current
#                  Push m onto open_set if not already present
#
#  3. open_set empty without reaching goal -> RETURN []  (no path exists)
#
# =============================================================================


# =============================================================================
# PURE-PYTHON MIN-HEAP HELPERS
# =============================================================================
# A binary min-heap stores items in a list where for any index i:
#   * left child  is at 2*i + 1
#   * right child is at 2*i + 2
#   * parent      is at (i - 1) // 2
#
# The HEAP PROPERTY: every parent's f-cost <= its children's f-costs.
# This guarantees heap[0] is always the entry with the smallest f-cost.
#
# Two operations are needed:
#   _heap_push(heap, item) - add a new item and restore the heap property
#                            by "bubbling up" (sift-up).  O(log n).
#   _heap_pop(heap)        - remove and return the minimum item, restore
#                            the heap property by "sifting down". O(log n).
#
# Each heap entry is a list [f_cost, node] so comparisons use f_cost first.
# Lists are used instead of tuples so entries can be invalidated in-place
# (lazy deletion: set f_cost = infinity to mark a stale entry).
# =============================================================================

def _heap_push(heap, item):
    """
    Insert `item` into the min-heap and restore the heap property via sift-up.

    Sift-up: the new item starts at the end of the list (index = len-1).
    It is repeatedly swapped with its parent while its f-cost is smaller
    than its parent's f-cost, bubbling it towards the root.

    Time complexity: O(log n) where n = current heap size.

    Parameters
    ----------
    heap : list of [f_cost, node]  - the min-heap (modified in place)
    item : [f_cost, node]          - the new entry to insert
    """
    heap.append(item)
    child_idx = len(heap) - 1                  # index of the newly added item

    # Sift-up: swap with parent while child has a smaller f-cost than parent
    while child_idx > 0:
        parent_idx = (child_idx - 1) // 2     # integer floor division
        if heap[child_idx][0] < heap[parent_idx][0]:
            heap[child_idx], heap[parent_idx] = heap[parent_idx], heap[child_idx]
            child_idx = parent_idx             # continue checking from new position
        else:
            break                              # heap property already satisfied


def _heap_pop(heap):
    """
    Remove and return the minimum item (heap[0]) and restore heap property.

    Method:
      1. Swap root (minimum) with the last element.
      2. Remove the last element (which is now the old minimum).
      3. Sift-down the new root: repeatedly swap it with the smaller of its
         two children until the heap property is restored.

    Time complexity: O(log n) where n = current heap size.

    Parameters
    ----------
    heap : list of [f_cost, node]  - the min-heap (modified in place)

    Returns
    -------
    [f_cost, node]  - the entry with the smallest f-cost
    """
    if len(heap) == 1:
        return heap.pop()                      # trivial single-element case

    # Swap root with last element, then remove the last element (old minimum)
    min_item = heap[0]
    heap[0] = heap.pop()                       # place last item at root

    # Sift-down: push the new root down until heap property is restored
    idx = 0
    size = len(heap)

    while True:
        left  = 2 * idx + 1                    # index of left  child
        right = 2 * idx + 2                    # index of right child
        smallest = idx                         # assume current is smallest

        # Check if left child exists and has a smaller f-cost than current
        if left < size and heap[left][0] < heap[smallest][0]:
            smallest = left

        # Check if right child exists and has a smaller f-cost than current
        if right < size and heap[right][0] < heap[smallest][0]:
            smallest = right

        if smallest == idx:
            break                              # heap property restored

        heap[idx], heap[smallest] = heap[smallest], heap[idx]
        idx = smallest                         # continue from new position

    return min_item


# =============================================================================
# MAIN PATH PLANNING FUNCTION
# DO NOT EDIT THIS FUNCTION DECLARATION
# =============================================================================

def do_a_star(grid, start, end, display_message):
    """
    A* path planning on a 2-D occupancy grid.

    Implements the algorithm described in:
      Hart, Nilsson & Raphael, "A Formal Basis for the Heuristic
      Determination of Minimum Cost Paths", IEEE TSSC, 1968.

    Grid convention (Lecture 9+10, Slide 26):
        grid[col][row] == 1  ->  free cell   (traversable node)
        grid[col][row] == 0  ->  obstacle    (blocked; missing node/edge)

    Motion model - 4-connected holonomic (Lecture 9+10, Slides 9, 22):
        Allowed moves: east (+col), west (-col), south (+row), north (-row).
        Diagonal moves are NOT permitted (coursework constraint).
        Every move carries a uniform edge cost of 1.

    Heuristic (coursework constraint; Lecture 9+10, Slide 27):
        Euclidean distance to the goal:
            h(n) = sqrt( (col_n - col_goal)^2 + (row_n - row_goal)^2 )
        Admissible on a unit-cost 4-connected grid -> guarantees optimal path.

    Parameters
    ----------
    grid            : list[list[int]]
        2-D occupancy grid indexed grid[col][row].
    start           : tuple(int, int)
        (col, row) coordinate of the start cell. Must be free and in-bounds.
    end             : tuple(int, int)
        (col, row) coordinate of the goal cell. Must be free and in-bounds.
    display_message : callable(str, str)
        GUI message hook. Usage: display_message("text", "LEVEL")
        Levels: "DEBUG" (progress), "WARN" (no path), "ERROR" (bad input).

    Returns
    -------
    list[tuple(int, int)]
        Ordered list of (col, row) nodes from start (index 0) to goal (last),
        inclusive. Returns [] if no valid path exists or input is invalid.
    """

    # =========================================================================
    # SECTION 1 - INPUT VALIDATION
    # -------------------------------------------------------------------------
    # Explicit guards prevent GUI crashes and provide clear feedback when the
    # mapping or localisation subsystems supply invalid inputs.
    # These checks are standard defensive programming practice.
    # =========================================================================

    # Guard: the occupancy grid must be a non-empty 2-D structure
    if not grid or not grid[0]:
        display_message("Invalid grid", "ERROR")
        return []

    # Cache grid dimensions - avoids repeated len() calls inside the hot loop
    NUM_COLS = len(grid)       # total columns  (col / x-axis)
    NUM_ROWS = len(grid[0])    # total rows     (row / y-axis)

    # Unpack start and goal coordinates into named variables for readability
    start_col, start_row = start
    goal_col,  goal_row  = end

    # Guard: start cell must be within the occupancy grid boundary
    if not (0 <= start_col < NUM_COLS and 0 <= start_row < NUM_ROWS):
        display_message("Start is outside the grid", "ERROR")
        return []

    # Guard: goal cell must be within the occupancy grid boundary
    if not (0 <= goal_col < NUM_COLS and 0 <= goal_row < NUM_ROWS):
        display_message("End is outside the grid", "ERROR")
        return []

    # Guard: start cell must be free (value 1), not an obstacle (value 0)
    # An obstacle represents a missing node in the graph (Slide 26)
    if grid[start_col][start_row] == 0:
        display_message("Start is on an obstacle", "ERROR")
        return []

    # Guard: goal cell must be free (value 1), not an obstacle (value 0)
    if grid[goal_col][goal_row] == 0:
        display_message("End is on an obstacle", "ERROR")
        return []

    # Trivial case: start and goal are the same node; path contains only it
    if start == end:
        return [start]

    # =========================================================================
    # SECTION 2 - HEURISTIC FUNCTION  h(n)
    # -------------------------------------------------------------------------
    # Euclidean (straight-line) distance from a grid node to the goal.
    # Required by the coursework specification and described in Slide 27:
    #
    #   h(n) = sqrt( (col_n - col_goal)^2 + (row_n - row_goal)^2 )
    #
    # Implemented using ** 0.5 (no math import needed).
    #
    # WHY THIS IS ADMISSIBLE:
    #   On a unit-cost grid, the minimum number of steps between two cells
    #   cannot be less than the straight-line (Euclidean) distance between
    #   them (you cannot travel diagonally or through obstacles).
    #   Therefore h(n) <= h*(n) for all n, satisfying admissibility
    #   (Theorem 1, Hart et al. 1968) and guaranteeing an optimal path.
    #
    # WHY THIS IS CONSISTENT (Hart et al. 1968, Section III-B):
    #   For any node n and 4-connected neighbour m (cost = 1):
    #       h(n) <= 1 + h(m)
    #   This follows from the Euclidean triangle inequality.
    #   Consistency means once a node is added to the closed set, its
    #   g-cost is finalised and it never needs to be re-opened.
    #
    # CLOSURE: goal_col and goal_row are captured from the enclosing scope,
    # avoiding the overhead of passing the goal on every call in the hot loop.
    # =========================================================================

    def heuristic(node):
        """
        Euclidean distance from `node` to the goal (Slide 27; coursework spec).

        Uses goal_col and goal_row captured from the enclosing function scope.

        Parameters
        ----------
        node : tuple(int, int) - (col, row) of the node being evaluated

        Returns
        -------
        float - h(n): estimated cost from node to the goal
        """
        d_col = node[0] - goal_col
        d_row = node[1] - goal_row
        return (d_col * d_col + d_row * d_row) ** 0.5

    # =========================================================================
    # SECTION 3 - INITIALISE DATA STRUCTURES  (Slide 29)
    # -------------------------------------------------------------------------
    # All structures are built from Python built-ins only (no imports).
    #
    # open_heap (OPEN SET / FRONTIER):
    #   A min-heap of [f_cost, node] pairs, maintained by _heap_push /
    #   _heap_pop defined above. Always keeps the node with the lowest
    #   f(n) = g(n) + h(n) at index [0] for O(log n) access.
    #   This is the key speed improvement: vs O(n) for a plain list scan.
    #
    #   LAZY DELETION: when a shorter path is found to a node already in
    #   the heap, a new entry is pushed with the lower f-cost. The old
    #   (stale) entry remains but will be discarded when popped because
    #   the node will already be in the closed set. This avoids expensive
    #   heap re-indexing (O(n)) and is a standard A* optimisation.
    #
    # open_set_members (set):
    #   Mirrors the heap contents for O(1) "is this node in the open set?"
    #   checks, which the heap alone cannot provide efficiently.
    #
    # closed_set (set):
    #   Fully expanded nodes (interior). O(1) membership testing.
    #   Because the heuristic is consistent, a closed node's g-cost is
    #   optimal and will never be improved, so re-opening is never needed
    #   (Lemma 2, Hart et al. 1968).
    #
    # parent (dict):
    #   Maps node -> predecessor on the best known path.
    #   Chains of parent pointers are followed at termination to reconstruct
    #   the full path from goal back to start (Slide 48).
    #
    # g_cost (dict):
    #   Maps node -> best g(n) found so far.
    #   Nodes absent from the dict are treated as having g = infinity.
    # =========================================================================

    # f(start) = g(start) + h(start) = 0 + h(start)
    start_h = heuristic(start)

    # Initialise min-heap with the start node: entry format is [f_cost, node]
    open_heap = [[start_h, start]]

    # O(1) set mirror of heap contents for fast membership testing
    open_set_members = {start}

    # Closed set: fully expanded nodes whose optimal g-costs are confirmed
    closed_set = set()

    # Parent pointers for path reconstruction (no parent for start node)
    parent = {}

    # Best actual cost from start to each discovered node; g(start) = 0
    g_cost = {start: 0}

    display_message("Starting A* search", "DEBUG")

    # =========================================================================
    # SECTION 4 - MAIN A* SEARCH LOOP  (Slides 30-47; Hart et al. 1968, Sec II)
    # -------------------------------------------------------------------------
    # Continues while the open set (frontier) contains candidate nodes.
    # Each iteration expands the most-promising node (smallest f(n)),
    # guided by the heuristic towards the goal.
    #
    # Comparison with Dijkstra (Slide 49):
    #   Dijkstra uses h(n) = 0 for all n, expanding nodes breadth-first
    #   with no goal-direction, so it visits many irrelevant nodes.
    #   A* uses h(n) > 0 to steer towards the goal, expanding far fewer
    #   nodes while still guaranteeing the optimal path.
    # =========================================================================

    while open_heap:

        # ---------------------------------------------------------------------
        # STEP 4a - SELECT CURRENT NODE  (Slide 30: "pick lowest cost node n")
        # ---------------------------------------------------------------------
        # Pop the [f_cost, node] entry with the smallest f(n) from the heap.
        # This is the most-promising node: best combined estimate of cost so
        # far (g) plus estimated remaining cost to goal (h).
        # _heap_pop is O(log n) - the critical speed advantage over O(n)
        # linear scan.
        #
        # LAZY DELETION: if this node is already in the closed set, it was
        # placed there via a shorter route found after this heap entry was
        # pushed. Discard this stale entry and move to the next iteration.
        # ---------------------------------------------------------------------
        entry = _heap_pop(open_heap)
        current = entry[1]

        # Discard stale heap entries (lazy deletion pattern)
        if current in closed_set:
            continue

        # Keep the O(1) membership mirror consistent
        open_set_members.discard(current)

        # ---------------------------------------------------------------------
        # STEP 4b - GOAL CHECK  (Slide 48; Hart et al. 1968 Step 3)
        # ---------------------------------------------------------------------
        # When the goal is popped from the min-heap, its g-cost is provably
        # optimal (admissible heuristic + min-heap selection guarantee).
        # Reconstruct the path by tracing parent pointers goal -> start,
        # then reverse to produce start -> goal order.
        # Hart et al. (1968): "reconstruct a minimum cost path from s to t
        # by chaining back from t to s through the pointers."
        # ---------------------------------------------------------------------
        if current == end:
            display_message("Goal reached", "DEBUG")

            # Trace parent chain: goal -> ... -> start
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent.get(node)    # .get() returns None at start node

            path.reverse()                 # reorder: start -> goal
            return path

        # ---------------------------------------------------------------------
        # STEP 4c - CLOSE CURRENT NODE
        # (Slide 30: "remove n from open list to close list")
        # ---------------------------------------------------------------------
        # Transfer current to the closed set (interior). All four neighbours
        # are about to be inspected, completing the expansion of this node.
        # Under the consistency assumption, the g-cost is now confirmed optimal
        # and this node will never need to be re-opened (Lemma 2, Hart 1968).
        # ---------------------------------------------------------------------
        closed_set.add(current)

        # ---------------------------------------------------------------------
        # STEP 4d - EXPAND NEIGHBOURS  (Slides 30-47)
        # ---------------------------------------------------------------------
        # Generate the four 4-connected neighbours of `current`.
        # The 4-connected motion model (holonomic, Slides 9 & 22) restricts
        # movement to the four cardinal directions with uniform cost = 1.
        # Diagonal movement is explicitly prohibited by the coursework.
        # ---------------------------------------------------------------------
        curr_col, curr_row = current

        for nb_col, nb_row in (
            (curr_col + 1, curr_row),      # east  (+col)
            (curr_col - 1, curr_row),      # west  (-col)
            (curr_col,     curr_row + 1),  # south (+row)
            (curr_col,     curr_row - 1),  # north (-row)
        ):

            # -- BOUNDARY CHECK -----------------------------------------------
            # The neighbour must lie within the occupancy grid extents.
            # Cells outside the grid boundary have no corresponding node
            # (Slide 26: "cover entire area/volume with nodes").
            if not (0 <= nb_col < NUM_COLS and 0 <= nb_row < NUM_ROWS):
                continue

            # -- OBSTACLE CHECK -----------------------------------------------
            # Cells with grid value 0 are obstacles: missing nodes with no
            # edges (Slide 26: "remove nodes for areas that cannot be visited").
            if grid[nb_col][nb_row] == 0:
                continue

            neighbour = (nb_col, nb_row)

            # -- CLOSED SET CHECK ---------------------------------------------
            # Skip nodes already in the closed set. Their optimal g-costs are
            # confirmed; the consistent heuristic guarantees they cannot be
            # improved (Lemma 2, Hart et al. 1968; Slide 30).
            if neighbour in closed_set:
                continue

            # -- COMPUTE TENTATIVE g-COST -------------------------------------
            # Cost of reaching `neighbour` via `current`.
            # All edges in the occupancy grid have uniform weight = 1
            # (Slide 26: "assume all edge weightings are equal").
            # Therefore: tentative g(neighbour) = g(current) + 1.
            tentative_g = g_cost[current] + 1

            # -- UPDATE IF BETTER PATH FOUND ----------------------------------
            # Compare tentative_g against the best g-cost for this neighbour
            # recorded so far. Use a large sentinel value (float("inf")) for
            # nodes not yet discovered (absent from g_cost dict).
            # If the new route is cheaper (or the node is brand new), update
            # all records. This is Hart et al. (1968) Step 4: "store with each
            # successor node n both the cost of getting to n by the lowest
            # cost path found thus far, and a pointer to the predecessor."
            # This also implements: "if there is a shorter path via parent,
            # update shortest path g(m)" (Slide 31).
            if tentative_g < g_cost.get(neighbour, float("inf")):

                # Record the improved route to this neighbour
                parent[neighbour] = current
                g_cost[neighbour] = tentative_g

                # f(neighbour) = g(neighbour) + h(neighbour)  (Slide 28)
                f_neighbour = tentative_g + heuristic(neighbour)

                # Push new entry onto the min-heap: O(log n).
                # Any existing stale entry for this node will be discarded
                # via lazy deletion when it is eventually popped.
                _heap_push(open_heap, [f_neighbour, neighbour])
                open_set_members.add(neighbour)

    # =========================================================================
    # SECTION 5 - NO PATH EXISTS
    # -------------------------------------------------------------------------
    # The open set (frontier) has been fully exhausted without reaching the
    # goal. This conclusively proves the goal is unreachable from the start
    # (e.g. fully enclosed by obstacles).
    # A* is a COMPLETE algorithm (Hart et al. 1968, Theorem 1 proof, Case 2):
    # if any valid path exists it is guaranteed to be found; reaching this
    # point means no valid path exists in the given occupancy grid.
    # =========================================================================
    display_message("No path found", "WARN")
    return []


# end of file