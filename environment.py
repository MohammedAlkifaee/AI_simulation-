"""
environment.py
--------------
Defines the World Environments (Models) for the Simple Problem-Solving Agent:
1. AIMA Romania Road Map (Classic Russell & Norvig Chapter 3 Example)
2. 2D Grid City / Maze Map (Interactive Grid Navigation)

Each environment provides:
- States (nodes/locations)
- Coordinates for 2D visualization
- Actions & Transition Model: RESULT(state, action) -> next_state
- Step Costs: ACTION-COST(state, action, next_state)
- Heuristic Estimates: h(state, goal) for informed search (A* / Greedy)
"""

import math
from typing import Dict, List, Tuple, Optional


class MapEnvironment:
    """
    Represents a deterministic, known environment (نفرض الأحداث محددة / نبني نموذجاً)
    in which the Simple Problem-Solving Agent operates.
    """

    def __init__(
        self,
        name: str,
        nodes: Dict[str, Tuple[float, float]],
        edges: Dict[str, Dict[str, int]],
        default_start: str,
        default_goal: str,
        grid_size: Optional[Tuple[int, int]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
    ):
        self.name = name
        self.nodes = nodes  # state_name -> (normalized_x, normalized_y) in [0.0, 1.0]
        self.edges = edges  # state_name -> {neighbor_state: step_cost}
        self.default_start = default_start
        self.default_goal = default_goal
        self.grid_size = grid_size
        self.obstacles = set(obstacles) if obstacles else set()

        # Current physical state of the agent in the environment
        self.agent_location = default_start
        self.step_count = 0
        self.total_cost = 0
        self.history: List[str] = [default_start]

    def reset(self, start_state: Optional[str] = None, goal_state: Optional[str] = None):
        """Reset the environment to the initial state."""
        if start_state and start_state in self.nodes:
            self.default_start = start_state
        if goal_state and goal_state in self.nodes:
            self.default_goal = goal_state
        self.agent_location = self.default_start
        self.step_count = 0
        self.total_cost = 0
        self.history = [self.agent_location]

    def get_percept(self) -> str:
        """
        Returns the current percept from the environment.
        In the AIMA Simple-Problem-Solving-Agent, the percept tells the agent
        its current location/state (e.g., 'In(Arad)').
        """
        return self.agent_location

    def get_available_actions(self, state: str) -> List[str]:
        """Return list of actions available from `state`, formatted as 'Go(Target)'."""
        if state not in self.edges:
            return []
        return [f"Go({neighbor})" for neighbor in self.edges[state].keys()]

    def get_result(self, state: str, action: str) -> str:
        """
        Transition Model: RESULT(state, action) -> next_state.
        Parses 'Go(Target)' and returns 'Target'.
        """
        if action.startswith("Go(") and action.endswith(")"):
            target = action[3:-1]
            if target in self.edges.get(state, {}):
                return target
        return state

    def get_step_cost(self, state: str, action: str, next_state: str) -> int:
        """Return the cost of taking `action` from `state` to `next_state`."""
        return self.edges.get(state, {}).get(next_state, 1)

    def execute_action(self, action: Optional[str]) -> Tuple[str, str, int]:
        """
        Physically moves the agent in the environment according to `action`.
        Returns (old_state, new_state, step_cost).
        """
        old_state = self.agent_location
        if not action or action == "NoOp":
            return old_state, old_state, 0

        new_state = self.get_result(old_state, action)
        cost = self.get_step_cost(old_state, action, new_state)
        self.agent_location = new_state
        self.step_count += 1
        self.total_cost += cost
        self.history.append(new_state)
        return old_state, new_state, cost

    def heuristic(self, state: str, goal: str) -> float:
        """
        Straight-Line Distance (SLD) heuristic between `state` and `goal`.
        Uses classic AIMA SLD table if targeting Bucharest on the Romania map,
        otherwise computes scaled Euclidean distance from node coordinates.
        """
        if state == goal:
            return 0.0
        if self.name.startswith("Romania") and goal == "Bucharest":
            if state in ROMANIA_SLD_BUCHAREST:
                return float(ROMANIA_SLD_BUCHAREST[state])

        if state in self.nodes and goal in self.nodes:
            x1, y1 = self.nodes[state]
            x2, y2 = self.nodes[goal]
            scale = 650.0 if self.name.startswith("Romania") else 10.0
            return round(math.hypot(x2 - x1, y2 - y1) * scale, 1)
        return 0.0


# =====================================================================
# 1. Classic AIMA Chapter 3 Environment: Road Map of Romania
# =====================================================================

ROMANIA_SLD_BUCHAREST = {
    "Arad": 366,
    "Bucharest": 0,
    "Craiova": 160,
    "Drobeta": 242,
    "Eforie": 161,
    "Fagaras": 176,
    "Giurgiu": 77,
    "Hirsova": 151,
    "Iasi": 226,
    "Lugoj": 244,
    "Mehadia": 241,
    "Neamt": 234,
    "Oradea": 380,
    "Pitesti": 100,
    "Rimnicu Vilcea": 193,
    "Sibiu": 253,
    "Timisoara": 329,
    "Urziceni": 80,
    "Vaslui": 199,
    "Zerind": 374,
}


def create_romania_environment() -> MapEnvironment:
    """
    Builds the classic Romania Road Map from Russell & Norvig (AIMA Chapter 3).
    Coordinates are normalized in [0.05, 0.95] for clean canvas rendering.
    """
    nodes = {
        "Oradea": (0.19, 0.10),
        "Zerind": (0.13, 0.22),
        "Arad": (0.09, 0.35),
        "Timisoara": (0.10, 0.54),
        "Lugoj": (0.22, 0.61),
        "Mehadia": (0.23, 0.74),
        "Drobeta": (0.22, 0.87),
        "Craiova": (0.40, 0.89),
        "Sibiu": (0.33, 0.39),
        "Rimnicu Vilcea": (0.37, 0.56),
        "Fagaras": (0.50, 0.41),
        "Pitesti": (0.51, 0.67),
        "Bucharest": (0.67, 0.76),
        "Giurgiu": (0.62, 0.92),
        "Urziceni": (0.77, 0.69),
        "Hirsova": (0.91, 0.69),
        "Eforie": (0.94, 0.85),
        "Vaslui": (0.86, 0.43),
        "Iasi": (0.80, 0.25),
        "Neamt": (0.68, 0.17),
    }

    undirected_roads = [
        ("Arad", "Zerind", 75),
        ("Zerind", "Oradea", 71),
        ("Oradea", "Sibiu", 151),
        ("Arad", "Sibiu", 140),
        ("Arad", "Timisoara", 118),
        ("Timisoara", "Lugoj", 111),
        ("Lugoj", "Mehadia", 70),
        ("Mehadia", "Drobeta", 75),
        ("Drobeta", "Craiova", 120),
        ("Sibiu", "Fagaras", 99),
        ("Sibiu", "Rimnicu Vilcea", 80),
        ("Rimnicu Vilcea", "Pitesti", 97),
        ("Rimnicu Vilcea", "Craiova", 146),
        ("Craiova", "Pitesti", 138),
        ("Fagaras", "Bucharest", 211),
        ("Pitesti", "Bucharest", 101),
        ("Bucharest", "Giurgiu", 90),
        ("Bucharest", "Urziceni", 85),
        ("Urziceni", "Hirsova", 98),
        ("Hirsova", "Eforie", 86),
        ("Urziceni", "Vaslui", 142),
        ("Vaslui", "Iasi", 92),
        ("Iasi", "Neamt", 87),
    ]

    edges: Dict[str, Dict[str, int]] = {city: {} for city in nodes}
    for u, v, cost in undirected_roads:
        edges[u][v] = cost
        edges[v][u] = cost

    return MapEnvironment(
        name="Romania Road Map (AIMA Ch. 3)",
        nodes=nodes,
        edges=edges,
        default_start="Arad",
        default_goal="Bucharest",
    )


# =====================================================================
# 2. 2D Grid City / Maze Environment
# =====================================================================

def create_grid_environment(rows: int = 6, cols: int = 8) -> MapEnvironment:
    """
    Builds a 2D Grid World Map with walls/obstacles where each cell is a state
    named '(r,c)' and actions move between adjacent non-wall cells.
    """
    default_obstacles = [
        (1, 2), (2, 2), (3, 2),
        (1, 5), (2, 5), (4, 4), (4, 5),
        (3, 0),
    ]
    obstacles = set(default_obstacles)

    nodes: Dict[str, Tuple[float, float]] = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                continue
            name = f"({r},{c})"
            # Normalized coordinates for canvas
            nx = 0.10 + (c / max(1, cols - 1)) * 0.80
            ny = 0.12 + (r / max(1, rows - 1)) * 0.76
            nodes[name] = (nx, ny)

    edges: Dict[str, Dict[str, int]] = {name: {} for name in nodes}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                continue
            u = f"({r},{c})"
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles:
                    v = f"({nr},{nc})"
                    edges[u][v] = 1

    return MapEnvironment(
        name="2D Grid Maze World",
        nodes=nodes,
        edges=edges,
        default_start="(0,0)",
        default_goal=f"({rows - 1},{cols - 1})",
        grid_size=(rows, cols),
        obstacles=default_obstacles,
    )
