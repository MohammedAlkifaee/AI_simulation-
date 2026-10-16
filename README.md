# Simple Problem-Solving Agent — Full Python Project & Map Visualizer

This project implements the **Simple Problem-Solving Agent** from **Russell & Norvig (*Artificial Intelligence: A Modern Approach*, Chapter 3, Slide 19)** with an interactive **Map & Pseudocode Step-by-Step Visualizer**.

---

## 1. Lecture Slide Pseudocode & Mapping to Python Code

```text
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
```

### Handwritten Lecture Notes Included in the Model:
- **عندما ينشأ سؤال لا يغير الـ state / نفرض الأحداث محددة**: The environment is **static** and **deterministic** (`MapEnvironment` in `environment.py`).
- **نبني نموذجاً / نكتب نموذج منه بالبايثون**: Formalized in `Problem` and `SimpleProblemSolvingAgent` in `agent.py`.
- **Offline Problem Solving ("Eyes Closed")**:
  - In **Cycle 1** (`seq` is empty `[]`), the agent executes `FORMULATE-GOAL`, `FORMULATE-PROBLEM`, and `SEARCH(problem)` to compute the full action sequence `seq`.
  - In **Cycles 2..N** (`seq` is NOT empty), the agent skips the `if seq is empty` block and directly executes `action <- FIRST(seq)` and `seq <- REST(seq)` while moving step-by-step across the map until reaching the goal.

---

## 2. Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | Main entry point — prints the full step-by-step terminal trace and launches the interactive GUI. |
| `agent.py` | Exact implementation of `SimpleProblemSolvingAgent`, `Problem`, and search algorithms (`A*`, `UCS`, `BFS`, `DFS`). |
| `environment.py` | World environments: **Romania Road Map (AIMA Ch. 3)** and **2D Grid Maze World**. |
| `visualizer_gui.py` | Interactive `tkinter` GUI that highlights each line of the slide's pseudocode, inspects static variables (`seq`, `state`, `goal`, `problem`, `action`), animates search expansion, and moves the agent along the map. |

---

## 3. How to Run

### Option A: Launch Interactive GUI + Terminal Trace (Default)
```bash
python3 main.py
```

### Option B: Run Terminal Trace Only (CLI Mode)
```bash
python3 main.py --cli
python3 main.py --cli --algo BFS --start Arad --goal Bucharest
python3 main.py --cli --env grid --start "(0,0)" --goal "(5,7)"
```

---

## 4. Interactive GUI Controls

- **Map Environment**: Switch between **Romania Road Map (AIMA Ch. 3)** and **2D Grid Maze World**.
- **SEARCH Algorithm**: Switch between **A\***, **UCS (Uniform Cost Search)**, **BFS (Breadth-First Search)**, and **DFS (Depth-First Search)**.
- **Start / Goal Selection**: Choose from the dropdowns or **Left-Click** any node on the map to set `Start` and **Right-Click** any node to set `Goal`.
- **▶ Next Step (Line)**: Steps line-by-line through the slide's pseudocode (`UPDATE-STATE` $\rightarrow$ `if seq is empty` $\rightarrow$ `FORMULATE-GOAL` $\rightarrow$ `FORMULATE-PROBLEM` $\rightarrow$ `SEARCH` $\rightarrow$ `FIRST(seq)` $\rightarrow$ `REST(seq)` $\rightarrow$ `return action`).
- **⏭ Next Action (Move)**: Jumps directly to the next action execution and smoothly animates the agent moving across the map.
- **⏵ Auto Play / ⏸ Pause**: Automatically steps through the entire problem-solving lifecycle with adjustable speed.
