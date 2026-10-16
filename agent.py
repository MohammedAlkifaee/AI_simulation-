"""
agent.py
--------
Exact implementation of the Simple Problem-Solving Agent from Slide 19
(Russell & Norvig, Artificial Intelligence: A Modern Approach, Chapter 3):

    function SIMPLE-PROBLEM-SOLVING-AGENT(percept) returns an action
      static: seq, an action sequence, initially empty
              state, some description of the current world state
              goal, a goal, initially null
              problem, a problem formulation

      state <- UPDATE-STATE(state, percept)
      if seq is empty then
        goal <- FORMULATE-GOAL(state)
        problem <- FORMULATE-PROBLEM(state, goal)
        seq <- SEARCH(problem)
      action <- FIRST(seq)
      seq <- REST(seq)
      return action

    Note: this is offline problem solving; solution executed "eyes closed."

Arabic Lecture Notes from Slide:
- عندما ينشأ سؤال لا يغير الـ state (When a query/percept arises in static offline world)
- نكتب نموذج منه (We write a model/instance of it)
- نفرض الأحداث محددة (We assume deterministic actions/events)
- نبني نموذجاً (We build a model)
"""

from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import List, Optional, Dict, Tuple, Any
from environment import MapEnvironment


# Exact pseudocode lines from Slide 19 (1-indexed for GUI highlighting)
SLIDE_PSEUDOCODE_LINES = [
    "function SIMPLE-PROBLEM-SOLVING-AGENT(percept) returns an action",
    "  static: seq, an action sequence, initially empty",
    "          state, some description of the current world state",
    "          goal, a goal, initially null",
    "          problem, a problem formulation",
    "  state <- UPDATE-STATE(state, percept)",
    "  if seq is empty then",
    "    goal <- FORMULATE-GOAL(state)",
    "    problem <- FORMULATE-PROBLEM(state, goal)",
    "    seq <- SEARCH(problem)",
    "  action <- FIRST(seq)",
    "  seq <- REST(seq)",
    "  return action",
]


@dataclass
class Problem:
    """
    Formal Problem Formulation created by FORMULATE-PROBLEM(state, goal).
    Consists of:
    1. initial_state: The state that the agent starts in
    2. goal_state: The target goal state
    3. environment: The transition model (ACTIONS, RESULT, ACTION-COST)
    """
    initial_state: str
    goal_state: str
    environment: MapEnvironment

    def actions(self, state: str) -> List[str]:
        return self.environment.get_available_actions(state)

    def result(self, state: str, action: str) -> str:
        return self.environment.get_result(state, action)

    def goal_test(self, state: str) -> bool:
        return state == self.goal_state

    def step_cost(self, state: str, action: str, next_state: str) -> int:
        return self.environment.get_step_cost(state, action, next_state)

    def heuristic(self, state: str) -> float:
        return self.environment.heuristic(state, self.goal_state)

    def __repr__(self) -> str:
        return f"Problem(initial='{self.initial_state}', goal='{self.goal_state}')"


@dataclass
class SearchExpandStep:
    """Records one expansion step of SEARCH(problem) for visual animation."""
    current_node: str
    frontier_nodes: List[str]
    explored_nodes: List[str]
    path_so_far: List[str]
    actions_so_far: List[str]
    cost_so_far: float
    heuristic_val: float
    description: str


@dataclass
class AgentMicroStep:
    """
    Represents a single visualizable micro-step of the slide's pseudocode
    so the student/teacher can watch each line execute live.
    """
    line_index: int  # Index into SLIDE_PSEUDOCODE_LINES
    phase_name: str
    explanation: str
    percept: str
    state: Optional[str]
    goal: Optional[str]
    problem_str: str
    seq: List[str]
    action: Optional[str]
    agent_location: str
    move_from: Optional[str] = None
    move_to: Optional[str] = None
    step_cost: int = 0
    search_step: Optional[SearchExpandStep] = None
    planned_state_path: List[str] = field(default_factory=list)


class SimpleProblemSolvingAgent:
    """
    Implementation of SIMPLE-PROBLEM-SOLVING-AGENT from Slide 19.
    Maintains persistent ('static') variables across calls:
      - seq: list of planned actions, initially empty []
      - state: current world state description, initially None
      - goal: formulated goal state, initially None
      - problem: Problem formulation instance, initially None
    """

    def __init__(
        self,
        environment: MapEnvironment,
        target_goal: Optional[str] = None,
        search_algorithm: str = "A*",
    ):
        self.environment = environment
        self.target_goal = target_goal or environment.default_goal
        self.search_algorithm = search_algorithm

        # === STATIC VARIABLES FROM SLIDE 19 ===
        self.seq: List[str] = []           # static: seq, an action sequence, initially empty
        self.state: Optional[str] = None   # static: state, description of current world state
        self.goal: Optional[str] = None    # static: goal, a goal, initially null
        self.problem: Optional[Problem] = None  # static: problem, a problem formulation

        # Trace & visualization records
        self.last_search_trace: List[SearchExpandStep] = []
        self.planned_state_path: List[str] = []
        self.call_count: int = 0

    def reset(
        self,
        target_goal: Optional[str] = None,
        search_algorithm: Optional[str] = None,
    ):
        """Resets the agent's static memory back to initial empty/null state."""
        if target_goal is not None:
            self.target_goal = target_goal
        if search_algorithm is not None:
            self.search_algorithm = search_algorithm

        self.seq = []
        self.state = None
        self.goal = None
        self.problem = None
        self.last_search_trace = []
        self.planned_state_path = []
        self.call_count = 0

    # =========================================================================
    # HELPER FUNCTIONS DEFINED ON SLIDE 19
    # =========================================================================

    def UPDATE_STATE(self, state: Optional[str], percept: str) -> str:
        """
        state <- UPDATE-STATE(state, percept)
        Updates the agent's internal world state based on the latest percept.
        Note from slide: in offline problem solving ("eyes closed"), the world
        is deterministic and static, so the percept confirms/updates location.
        """
        return percept

    def FORMULATE_GOAL(self, state: str) -> str:
        """
        goal <- FORMULATE-GOAL(state)
        Determines the goal destination from the current state (e.g. 'Bucharest').
        """
        return self.target_goal

    def FORMULATE_PROBLEM(self, state: str, goal: str) -> Problem:
        """
        problem <- FORMULATE-PROBLEM(state, goal)
        Constructs a formal Problem object containing initial state, goal state,
        actions, transition model, and step costs.
        """
        return Problem(
            initial_state=state,
            goal_state=goal,
            environment=self.environment,
        )

    def SEARCH(self, problem: Problem) -> List[str]:
        """
        seq <- SEARCH(problem)
        Executes the chosen search algorithm (A*, UCS, BFS, or DFS) on `problem`
        and returns a sequence of actions `[Go(...), ...]` reaching `goal`.
        Also populates `self.last_search_trace` for step-by-step visualization.
        """
        algo = self.search_algorithm.upper()
        if algo in ("BFS", "BREADTH-FIRST SEARCH"):
            actions, state_path, trace = self._bfs_search(problem)
        elif algo in ("DFS", "DEPTH-FIRST SEARCH"):
            actions, state_path, trace = self._dfs_search(problem)
        elif algo in ("UCS", "UNIFORM COST SEARCH"):
            actions, state_path, trace = self._priority_search(problem, use_heuristic=False)
        else:  # Default: A* Search
            actions, state_path, trace = self._priority_search(problem, use_heuristic=True)

        self.last_search_trace = trace
        self.planned_state_path = state_path
        return actions

    @staticmethod
    def FIRST(seq: List[str]) -> Optional[str]:
        """
        action <- FIRST(seq)
        Returns the first action in the sequence, or None if empty.
        """
        if not seq:
            return None
        return seq[0]

    @staticmethod
    def REST(seq: List[str]) -> List[str]:
        """
        seq <- REST(seq)
        Returns the remaining actions after removing the first action.
        """
        if not seq:
            return []
        return list(seq[1:])

    # =========================================================================
    # MAIN AGENT FUNCTION (EXACT MATCH TO SLIDE 19)
    # =========================================================================

    def __call__(self, percept: str) -> Optional[str]:
        """
        function SIMPLE-PROBLEM-SOLVING-AGENT(percept) returns an action
        """
        self.call_count += 1

        # 1. state <- UPDATE-STATE(state, percept)
        self.state = self.UPDATE_STATE(self.state, percept)

        # If already at goal and seq is empty, no further action needed
        if self.goal is not None and self.state == self.goal and not self.seq:
            return None

        # 2. if seq is empty then
        if len(self.seq) == 0:
            # goal <- FORMULATE-GOAL(state)
            self.goal = self.FORMULATE_GOAL(self.state)

            # problem <- FORMULATE-PROBLEM(state, goal)
            self.problem = self.FORMULATE_PROBLEM(self.state, self.goal)

            # seq <- SEARCH(problem)
            self.seq = self.SEARCH(self.problem)
            if not self.seq:
                return None

        # 3. action <- FIRST(seq)
        action = self.FIRST(self.seq)

        # 4. seq <- REST(seq)
        self.seq = self.REST(self.seq)

        # 5. return action
        return action

    # =========================================================================
    # SEARCH ALGORITHMS (BFS, DFS, UCS, A*) WITH VISUAL TRACE RECORDING
    # =========================================================================

    def _bfs_search(
        self, problem: Problem
    ) -> Tuple[List[str], List[str], List[SearchExpandStep]]:
        initial = problem.initial_state
        if problem.goal_test(initial):
            return [], [initial], []

        frontier = deque([(initial, [initial], [], 0)])
        frontier_set = {initial}
        explored = []
        explored_set = set()
        trace: List[SearchExpandStep] = []

        while frontier:
            node, state_path, action_path, cost = frontier.popleft()
            frontier_set.discard(node)
            if node in explored_set:
                continue
            explored.append(node)
            explored_set.add(node)

            trace.append(
                SearchExpandStep(
                    current_node=node,
                    frontier_nodes=list(frontier_set),
                    explored_nodes=list(explored),
                    path_so_far=list(state_path),
                    actions_so_far=list(action_path),
                    cost_so_far=cost,
                    heuristic_val=0.0,
                    description=f"[BFS] Expanding '{node}' (path cost = {cost})",
                )
            )

            if problem.goal_test(node):
                return action_path, state_path, trace

            for action in problem.actions(node):
                child = problem.result(node, action)
                step_c = problem.step_cost(node, action, child)
                if child not in explored_set and child not in frontier_set:
                    new_states = state_path + [child]
                    new_actions = action_path + [action]
                    new_cost = cost + step_c
                    if problem.goal_test(child):
                        explored.append(child)
                        trace.append(
                            SearchExpandStep(
                                current_node=child,
                                frontier_nodes=list(frontier_set),
                                explored_nodes=list(explored),
                                path_so_far=new_states,
                                actions_so_far=new_actions,
                                cost_so_far=new_cost,
                                heuristic_val=0.0,
                                description=f"[BFS] Goal '{child}' reached! Total cost = {new_cost}",
                            )
                        )
                        return new_actions, new_states, trace
                    frontier.append((child, new_states, new_actions, new_cost))
                    frontier_set.add(child)

        return [], [], trace

    def _dfs_search(
        self, problem: Problem
    ) -> Tuple[List[str], List[str], List[SearchExpandStep]]:
        initial = problem.initial_state
        if problem.goal_test(initial):
            return [], [initial], []

        stack = [(initial, [initial], [], 0)]
        explored = []
        explored_set = set()
        trace: List[SearchExpandStep] = []

        while stack:
            node, state_path, action_path, cost = stack.pop()
            if node in explored_set:
                continue
            explored.append(node)
            explored_set.add(node)
            frontier_nodes = [item[0] for item in stack if item[0] not in explored_set]

            trace.append(
                SearchExpandStep(
                    current_node=node,
                    frontier_nodes=frontier_nodes,
                    explored_nodes=list(explored),
                    path_so_far=list(state_path),
                    actions_so_far=list(action_path),
                    cost_so_far=cost,
                    heuristic_val=0.0,
                    description=f"[DFS] Expanding '{node}' (depth = {len(action_path)}, cost = {cost})",
                )
            )

            if problem.goal_test(node):
                return action_path, state_path, trace

            for action in reversed(problem.actions(node)):
                child = problem.result(node, action)
                if child not in explored_set:
                    step_c = problem.step_cost(node, action, child)
                    stack.append(
                        (child, state_path + [child], action_path + [action], cost + step_c)
                    )

        return [], [], trace

    def _priority_search(
        self, problem: Problem, use_heuristic: bool = True
    ) -> Tuple[List[str], List[str], List[SearchExpandStep]]:
        initial = problem.initial_state
        if problem.goal_test(initial):
            return [], [initial], []

        algo_label = "A*" if use_heuristic else "UCS"
        h0 = problem.heuristic(initial) if use_heuristic else 0.0
        # Heap items: (f_score, tie_breaker, g_cost, current_node, state_path, action_path)
        counter = 0
        pq = [(h0, counter, 0, initial, [initial], [])]
        best_g: Dict[str, int] = {initial: 0}
        explored = []
        explored_set = set()
        trace: List[SearchExpandStep] = []

        while pq:
            f_val, _, g_cost, node, state_path, action_path = heapq.heappop(pq)
            if node in explored_set:
                continue
            explored.append(node)
            explored_set.add(node)

            frontier_nodes = list(
                dict.fromkeys(item[3] for item in pq if item[3] not in explored_set)
            )
            h_val = problem.heuristic(node) if use_heuristic else 0.0
            desc = (
                f"[{algo_label}] Expanding '{node}' | g(n)={g_cost}, h(n)={h_val:.0f}, f(n)={f_val:.0f}"
                if use_heuristic
                else f"[{algo_label}] Expanding '{node}' | path cost g(n)={g_cost}"
            )
            trace.append(
                SearchExpandStep(
                    current_node=node,
                    frontier_nodes=frontier_nodes,
                    explored_nodes=list(explored),
                    path_so_far=list(state_path),
                    actions_so_far=list(action_path),
                    cost_so_far=g_cost,
                    heuristic_val=h_val,
                    description=desc,
                )
            )

            if problem.goal_test(node):
                return action_path, state_path, trace

            for action in problem.actions(node):
                child = problem.result(node, action)
                step_c = problem.step_cost(node, action, child)
                new_g = g_cost + step_c
                if child not in explored_set and (
                    child not in best_g or new_g < best_g[child]
                ):
                    best_g[child] = new_g
                    counter += 1
                    h_child = problem.heuristic(child) if use_heuristic else 0.0
                    f_child = new_g + h_child
                    heapq.heappush(
                        pq,
                        (
                            f_child,
                            counter,
                            new_g,
                            child,
                            state_path + [child],
                            action_path + [action],
                        ),
                    )

        return [], [], trace


def generate_full_episode_microsteps(
    env: MapEnvironment,
    start_state: str,
    goal_state: str,
    search_algorithm: str = "A*",
    include_search_expansion_steps: bool = True,
) -> List[AgentMicroStep]:
    """
    Runs a complete episode of SIMPLE-PROBLEM-SOLVING-AGENT from `start_state` to `goal_state`
    and records every single line of pseudocode execution as a list of `AgentMicroStep`s.
    This powers both the interactive step-by-step GUI and the CLI trace table.
    """
    env.reset(start_state=start_state, goal_state=goal_state)
    agent = SimpleProblemSolvingAgent(
        environment=env,
        target_goal=goal_state,
        search_algorithm=search_algorithm,
    )

    microsteps: List[AgentMicroStep] = []

    def snap(
        line_idx: int,
        phase: str,
        explanation: str,
        percept_val: str,
        action_val: Optional[str] = None,
        move_from: Optional[str] = None,
        move_to: Optional[str] = None,
        step_cost: int = 0,
        search_step: Optional[SearchExpandStep] = None,
    ):
        microsteps.append(
            AgentMicroStep(
                line_index=line_idx,
                phase_name=phase,
                explanation=explanation,
                percept=percept_val,
                state=agent.state,
                goal=agent.goal,
                problem_str=repr(agent.problem) if agent.problem else "null",
                seq=list(agent.seq),
                action=action_val,
                agent_location=env.agent_location,
                move_from=move_from,
                move_to=move_to,
                step_cost=step_cost,
                search_step=search_step,
                planned_state_path=list(agent.planned_state_path),
            )
        )

    # Initial static state snapshot
    snap(
        line_idx=1,
        phase="Initialization (Static Memory)",
        explanation=(
            "Agent initialized with static variables: seq = [], state = null, "
            "goal = null, problem = null."
        ),
        percept_val="None",
    )

    max_cycles = 50
    cycle = 0

    while cycle < max_cycles:
        cycle += 1
        percept = env.get_percept()

        # Function entry
        snap(
            line_idx=0,
            phase=f"Cycle #{cycle}: Percept Received",
            explanation=f"Environment provides percept = '{percept}' to SIMPLE-PROBLEM-SOLVING-AGENT.",
            percept_val=percept,
        )

        # state <- UPDATE-STATE(state, percept)
        agent.state = agent.UPDATE_STATE(agent.state, percept)
        snap(
            line_idx=5,
            phase=f"Cycle #{cycle}: UPDATE-STATE",
            explanation=f"state <- UPDATE-STATE(state, '{percept}')  ==>  state = '{agent.state}'",
            percept_val=percept,
        )

        # Check if goal already reached
        if agent.goal is not None and agent.state == agent.goal and len(agent.seq) == 0:
            snap(
                line_idx=12,
                phase="Goal Reached!",
                explanation=f"Agent is at goal state '{agent.goal}' and action sequence `seq` is empty. Mission complete!",
                percept_val=percept,
                action_val="NoOp (Goal Reached)",
            )
            break

        # if seq is empty then
        is_empty = len(agent.seq) == 0
        snap(
            line_idx=6,
            phase=f"Cycle #{cycle}: Check if seq is empty",
            explanation=(
                "seq is EMPTY ([]) -> Entering offline formulation & search phase!"
                if is_empty
                else f"seq is NOT empty ({len(agent.seq)} actions remaining) -> Skip search! Execute 'eyes closed' (Offline execution)."
            ),
            percept_val=percept,
        )

        if is_empty:
            # goal <- FORMULATE-GOAL(state)
            agent.goal = agent.FORMULATE_GOAL(agent.state)
            snap(
                line_idx=7,
                phase=f"Cycle #{cycle}: FORMULATE-GOAL",
                explanation=f"goal <- FORMULATE-GOAL('{agent.state}')  ==>  goal = '{agent.goal}'",
                percept_val=percept,
            )

            # problem <- FORMULATE-PROBLEM(state, goal)
            agent.problem = agent.FORMULATE_PROBLEM(agent.state, agent.goal)
            snap(
                line_idx=8,
                phase=f"Cycle #{cycle}: FORMULATE-PROBLEM",
                explanation=(
                    f"problem <- FORMULATE-PROBLEM('{agent.state}', '{agent.goal}')  ==>  "
                    f"{agent.problem} (deterministic model built)"
                ),
                percept_val=percept,
            )

            # seq <- SEARCH(problem)
            computed_seq = agent.SEARCH(agent.problem)
            if include_search_expansion_steps:
                for idx, s_step in enumerate(agent.last_search_trace, start=1):
                    snap(
                        line_idx=9,
                        phase=f"Cycle #{cycle}: SEARCH ({search_algorithm} Step {idx}/{len(agent.last_search_trace)})",
                        explanation=s_step.description,
                        percept_val=percept,
                        search_step=s_step,
                    )

            agent.seq = computed_seq
            snap(
                line_idx=9,
                phase=f"Cycle #{cycle}: SEARCH Complete",
                explanation=(
                    f"seq <- SEARCH(problem) found optimal/valid plan with {len(agent.seq)} actions: "
                    f"{agent.seq}"
                    if agent.seq
                    else "seq <- SEARCH(problem) returned failure / empty sequence!"
                ),
                percept_val=percept,
                search_step=agent.last_search_trace[-1] if agent.last_search_trace else None,
            )

            if not agent.seq:
                snap(
                    line_idx=12,
                    phase="Termination",
                    explanation="No actions needed or no path found. Returning None.",
                    percept_val=percept,
                    action_val="None",
                )
                break

        # action <- FIRST(seq)
        action = agent.FIRST(agent.seq)
        snap(
            line_idx=10,
            phase=f"Cycle #{cycle}: FIRST(seq)",
            explanation=f"action <- FIRST(seq) extracts next action: '{action}'",
            percept_val=percept,
            action_val=action,
        )

        # seq <- REST(seq)
        agent.seq = agent.REST(agent.seq)
        snap(
            line_idx=11,
            phase=f"Cycle #{cycle}: REST(seq)",
            explanation=f"seq <- REST(seq) removes '{action}'. Remaining seq = {agent.seq}",
            percept_val=percept,
            action_val=action,
        )

        # return action (and execute in environment so the map moves!)
        old_loc, new_loc, step_cost = env.execute_action(action)
        snap(
            line_idx=12,
            phase=f"Cycle #{cycle}: return action & Move on Map",
            explanation=(
                f"return '{action}' -> Agent moves on map from '{old_loc}' to '{new_loc}' "
                f"(step cost = {step_cost}, total cost = {env.total_cost})."
            ),
            percept_val=percept,
            action_val=action,
            move_from=old_loc,
            move_to=new_loc,
            step_cost=step_cost,
        )

        if env.agent_location == agent.goal and len(agent.seq) == 0:
            # One final percept check to show goal arrival
            final_percept = env.get_percept()
            agent.state = agent.UPDATE_STATE(agent.state, final_percept)
            snap(
                line_idx=5,
                phase="Goal Destination Reached!",
                explanation=(
                    f"Agent arrived at goal '{agent.goal}'! Total actions executed: {env.step_count}, "
                    f"Total path cost: {env.total_cost}. Path: {' -> '.join(env.history)}"
                ),
                percept_val=final_percept,
                action_val="Done (Goal Reached)",
            )
            break

    return microsteps
