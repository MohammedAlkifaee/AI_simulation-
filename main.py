"""
main.py
-------
Main Entry Point for the AI Lecture Project:
    "Simple Problem-Solving Agent (Slide 19 - Russell & Norvig AIMA Chapter 3)"

Usage:
    python3 main.py          # Prints console trace AND launches interactive Map & Pseudocode GUI
    python3 main.py --cli    # Runs step-by-step console trace only (no GUI)
"""

import argparse
from environment import create_romania_environment, create_grid_environment
from agent import SimpleProblemSolvingAgent


def run_cli_demonstration(
    env_type: str = "romania",
    start: str = "Arad",
    goal: str = "Bucharest",
    algorithm: str = "A*",
):
    """
    Demonstrates the exact execution of `SIMPLE-PROBLEM-SOLVING-AGENT(percept)`
    in the terminal with step-by-step inspection of static variables and map movement.
    """
    env = (
        create_romania_environment()
        if env_type.lower() == "romania"
        else create_grid_environment()
    )
    if start not in env.nodes:
        start = env.default_start
    if goal not in env.nodes:
        goal = env.default_goal

    env.reset(start_state=start, goal_state=goal)
    agent = SimpleProblemSolvingAgent(
        environment=env,
        target_goal=goal,
        search_algorithm=algorithm,
    )

    print("=" * 88)
    print(" AI LECTURE PROJECT: SIMPLE-PROBLEM-SOLVING-AGENT (AIMA Chapter 3 - Slide 19)")
    print("=" * 88)
    print(f" Environment      : {env.name}")
    print(f" Search Algorithm : {algorithm}")
    print(f" Start State      : {start}")
    print(f" Goal State       : {goal}")
    print(" Mode             : Offline Problem Solving ('eyes closed' execution)")
    print("-" * 88)
    print(" Initial Static Variables:")
    print(f"   seq     = {agent.seq}")
    print(f"   state   = {agent.state}")
    print(f"   goal    = {agent.goal}")
    print(f"   problem = {agent.problem}")
    print("=" * 88)

    step = 0
    while True:
        step += 1
        percept = env.get_percept()
        print(f"\n[Time Step {step}] Percept received from Environment: percept = '{percept}'")

        seq_was_empty = len(agent.seq) == 0
        print(f"  1) state <- UPDATE-STATE(state, '{percept}')  ==>  state = '{percept}'")

        if percept == goal and seq_was_empty and agent.goal == goal:
            print(f"  ✓ Goal '{goal}' reached and `seq` is empty. Agent stops.")
            break

        if seq_was_empty:
            print("  2) if seq is empty then (TRUE - First step: Formulate & Search):")
            action = agent(percept)
            print(f"       goal    <- FORMULATE-GOAL('{ agent.state }')          ==>  goal = '{agent.goal}'")
            print(f"       problem <- FORMULATE-PROBLEM(state, goal)      ==>  {agent.problem}")
            print(f"       seq     <- SEARCH(problem) [{algorithm}]")
            for idx, s_step in enumerate(agent.last_search_trace, 1):
                print(f"                  [Search {idx:02d}] {s_step.description}")
            full_planned_seq = ([action] if action else []) + agent.seq
            print(f"                  Planned Action Sequence = {full_planned_seq}")
        else:
            print(
                f"  2) if seq is empty then (FALSE - {len(agent.seq)} actions in `seq`, executing 'eyes closed'):"
            )
            action = agent(percept)

        if action is None:
            print("  ✓ No further actions. Halting.")
            break

        print(f"  3) action <- FIRST(seq)  ==>  action = '{action}'")
        print(f"  4) seq    <- REST(seq)   ==>  remaining seq = {agent.seq}")
        print(f"  5) return '{action}'")

        old_loc, new_loc, cost = env.execute_action(action)
        print(
            f"  🚗 [MAP MOVEMENT] Agent moves: {old_loc} ──({action}, cost={cost})──▶ {new_loc} "
            f"| Total Path Cost = {env.total_cost}"
        )

        if env.agent_location == goal and len(agent.seq) == 0:
            print(f"\n[Final Check] Agent is now at '{env.agent_location}' == Goal '{goal}'!")
            print("=" * 88)
            print(f" SUMMARY: Reached '{goal}' in {env.step_count} actions | Total Cost = {env.total_cost}")
            print(f" Route  : {' -> '.join(env.history)}")
            print("=" * 88)
            break


def main():
    parser = argparse.ArgumentParser(
        description="Simple Problem-Solving Agent (Slide 19) with Map & Step Visualization"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line trace mode only without opening the Tkinter GUI window.",
    )
    parser.add_argument(
        "--env",
        default="romania",
        choices=["romania", "grid"],
        help="Environment map to use in CLI mode (romania or grid).",
    )
    parser.add_argument("--start", default="Arad", help="Initial state name.")
    parser.add_argument("--goal", default="Bucharest", help="Target goal state name.")
    parser.add_argument(
        "--algo",
        default="A*",
        choices=["A*", "UCS", "BFS", "DFS"],
        help="Search algorithm for SEARCH(problem).",
    )
    args = parser.parse_args()

    # Always print the clean CLI trace to terminal first so the user can see the output
    run_cli_demonstration(
        env_type=args.env,
        start=args.start,
        goal=args.goal,
        algorithm=args.algo,
    )

    if not args.cli:
        print("\nLaunching Interactive Map & Pseudocode Visualizer GUI...")
        from visualizer_gui import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
