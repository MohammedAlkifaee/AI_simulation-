"""
visualizer_gui.py
-----------------
Interactive Visual Simulation of the Simple Problem-Solving Agent (Slide 19)
built with Python's standard `tkinter` library.

Features:
- Real-time highlighting of Slide 19 Pseudocode (`SIMPLE-PROBLEM-SOLVING-AGENT`)
- Live inspection of static variables: `seq`, `state`, `goal`, `problem`, `percept`, `action`
- Animated Map Canvas supporting both:
    1) Classic AIMA Chapter 3 Romania Road Map
    2) 2D Grid Maze World
- Large, lecture-hall-friendly typography across all panels and map labels.
"""

import math
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

from environment import (
    MapEnvironment,
    create_romania_environment,
    create_grid_environment,
)
from agent import (
    SLIDE_PSEUDOCODE_LINES,
    AgentMicroStep,
    generate_full_episode_microsteps,
)


class SimpleProblemSolvingAgentGUI:
    """Main Tkinter Application Window for visualizing Slide 19."""

    # Color Palette (Dark Modern Academic Theme)
    BG_MAIN = "#0f172a"       # Slate 900
    BG_PANEL = "#1e293b"      # Slate 800
    BG_CANVAS = "#0b1120"     # Deep Navy
    BG_CODE = "#090d16"       # Code background
    FG_TEXT = "#f8fafc"       # Slate 50
    FG_MUTED = "#cbd5e1"      # Slate 300
    ACCENT_BLUE = "#38bdf8"   # Sky 400
    ACCENT_GOLD = "#facc15"   # Yellow 400
    ACCENT_GREEN = "#22c55e"  # Green 500
    ACCENT_PURPLE = "#a855f7" # Purple 500
    ACCENT_ORANGE = "#f97316" # Orange 500
    ACCENT_RED = "#ef4444"    # Red 500

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(
            "AI Lecture Project: Simple Problem-Solving Agent Visualizer (AIMA Slide 19)"
        )
        self.root.geometry("1560x940")
        self.root.minsize(1280, 780)
        self.root.configure(bg=self.BG_MAIN)

        # Increase default font for dropdown lists
        self.root.option_add("*TCombobox*Listbox.font", ("Helvetica", 13, "bold"))

        # Environments
        self.environments: Dict[str, MapEnvironment] = {
            "Romania Road Map (AIMA Ch. 3)": create_romania_environment(),
            "2D Grid Maze World": create_grid_environment(),
        }
        self.current_env_name = "Romania Road Map (AIMA Ch. 3)"
        self.env: MapEnvironment = self.environments[self.current_env_name]

        # Simulation state
        self.microsteps: List[AgentMicroStep] = []
        self.current_step_idx: int = 0
        self.is_playing: bool = False
        self.play_after_id: Optional[str] = None
        self.anim_after_id: Optional[str] = None

        # Agent smooth animation position (normalized [0,1] coords)
        self.agent_draw_pos: Tuple[float, float] = self.env.nodes[self.env.default_start]
        self.traveled_edges: List[Tuple[str, str]] = []
        self.visited_states: List[str] = [self.env.default_start]

        self._build_ui()
        self._rebuild_episode()

    # =========================================================================
    # UI CONSTRUCTION
    # =========================================================================

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground="#1e293b",
            background="#334155",
            foreground="#f8fafc",
            font=("Helvetica", 13, "bold"),
        )

        # 1. Top Control Bar
        top_bar = tk.Frame(self.root, bg=self.BG_PANEL, pady=10, padx=14)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 6))

        # Environment selector
        tk.Label(
            top_bar, text="Map:", bg=self.BG_PANEL, fg=self.FG_TEXT,
            font=("Helvetica", 14, "bold")
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.env_var = tk.StringVar(value=self.current_env_name)
        self.env_combo = ttk.Combobox(
            top_bar, textvariable=self.env_var,
            values=list(self.environments.keys()),
            state="readonly", width=24, font=("Helvetica", 13, "bold")
        )
        self.env_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.env_combo.bind("<<ComboboxSelected>>", self._on_env_changed)

        # Algorithm selector
        tk.Label(
            top_bar, text="SEARCH:", bg=self.BG_PANEL, fg=self.FG_TEXT,
            font=("Helvetica", 14, "bold")
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.algo_var = tk.StringVar(value="A*")
        self.algo_combo = ttk.Combobox(
            top_bar, textvariable=self.algo_var,
            values=["A*", "UCS", "BFS", "DFS"],
            state="readonly", width=6, font=("Helvetica", 13, "bold")
        )
        self.algo_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.algo_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_episode())

        # Start State selector
        tk.Label(
            top_bar, text="Start:", bg=self.BG_PANEL, fg=self.ACCENT_GREEN,
            font=("Helvetica", 14, "bold")
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.start_var = tk.StringVar(value=self.env.default_start)
        self.start_combo = ttk.Combobox(
            top_bar, textvariable=self.start_var,
            values=list(self.env.nodes.keys()),
            state="readonly", width=13, font=("Helvetica", 13, "bold")
        )
        self.start_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.start_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_episode())

        # Goal State selector
        tk.Label(
            top_bar, text="Goal:", bg=self.BG_PANEL, fg=self.ACCENT_RED,
            font=("Helvetica", 14, "bold")
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.goal_var = tk.StringVar(value=self.env.default_goal)
        self.goal_combo = ttk.Combobox(
            top_bar, textvariable=self.goal_var,
            values=list(self.env.nodes.keys()),
            state="readonly", width=13, font=("Helvetica", 13, "bold")
        )
        self.goal_combo.pack(side=tk.LEFT, padx=(0, 16))
        self.goal_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_episode())

        # Execution Control Buttons
        btn_cfg = {
            "font": ("Helvetica", 13, "bold"),
            "padx": 10,
            "pady": 5,
            "bd": 0,
            "cursor": "hand2",
        }

        self.btn_reset = tk.Button(
            top_bar, text="⏮ Reset", bg="#475569", fg="#ffffff",
            command=self._on_reset, **btn_cfg
        )
        self.btn_reset.pack(side=tk.LEFT, padx=3)

        self.btn_prev = tk.Button(
            top_bar, text="◀ Prev", bg="#334155", fg="#ffffff",
            command=self._on_prev_step, **btn_cfg
        )
        self.btn_prev.pack(side=tk.LEFT, padx=3)

        self.btn_next = tk.Button(
            top_bar, text="▶ Next Step", bg="#0284c7", fg="#ffffff",
            command=self._on_next_step, **btn_cfg
        )
        self.btn_next.pack(side=tk.LEFT, padx=3)

        self.btn_next_action = tk.Button(
            top_bar, text="⏭ Next Move", bg="#0d9488", fg="#ffffff",
            command=self._on_next_action_move, **btn_cfg
        )
        self.btn_next_action.pack(side=tk.LEFT, padx=3)

        self.btn_play = tk.Button(
            top_bar, text="⏵ Auto Play", bg="#16a34a", fg="#ffffff",
            command=self._toggle_play, **btn_cfg
        )
        self.btn_play.pack(side=tk.LEFT, padx=6)

        # Speed Slider
        tk.Label(
            top_bar, text="Speed:", bg=self.BG_PANEL, fg=self.FG_MUTED,
            font=("Helvetica", 13, "bold")
        ).pack(side=tk.LEFT, padx=(6, 2))
        self.speed_scale = tk.Scale(
            top_bar, from_=150, to=1500, orient=tk.HORIZONTAL,
            bg=self.BG_PANEL, fg=self.FG_TEXT, highlightthickness=0,
            troughcolor="#0f172a", length=85, showvalue=False
        )
        self.speed_scale.set(650)
        self.speed_scale.pack(side=tk.LEFT, padx=4)

        # 2. Main Content Area (Left: Map Canvas, Right: Pseudocode + State Inspector)
        main_pane = tk.Frame(self.root, bg=self.BG_MAIN)
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Column: Map Visualization + Step Banner
        left_frame = tk.Frame(main_pane, bg=self.BG_PANEL, bd=1, relief=tk.FLAT)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Live Phase & Explanation Header above Map
        banner_frame = tk.Frame(left_frame, bg="#172554", padx=14, pady=10)
        banner_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_phase = tk.Label(
            banner_frame,
            text="Phase: Initialization",
            bg="#172554",
            fg=self.ACCENT_GOLD,
            font=("Helvetica", 16, "bold"),
            anchor="w",
        )
        self.lbl_phase.pack(side=tk.TOP, fill=tk.X)

        self.lbl_explanation = tk.Label(
            banner_frame,
            text="Ready to run SIMPLE-PROBLEM-SOLVING-AGENT(percept).",
            bg="#172554",
            fg="#e0f2fe",
            font=("Helvetica", 14, "bold"),
            anchor="w",
            justify=tk.LEFT,
            wraplength=820,
        )
        self.lbl_explanation.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # Map Canvas
        self.canvas = tk.Canvas(
            left_frame, bg=self.BG_CANVAS, highlightthickness=0
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_map())
        self.canvas.bind("<Button-1>", self._on_canvas_left_click)
        self.canvas.bind("<Button-2>", self._on_canvas_right_click)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)

        # Legend Bar below Canvas
        legend_frame = tk.Frame(left_frame, bg=self.BG_PANEL, pady=8, padx=10)
        legend_frame.pack(side=tk.BOTTOM, fill=tk.X)
        legends = [
            ("● Start", self.ACCENT_GREEN),
            ("● Goal", self.ACCENT_RED),
            ("● Explored", self.ACCENT_PURPLE),
            ("● Frontier", self.ACCENT_ORANGE),
            ("--- Planned (seq)", self.ACCENT_BLUE),
            ("━━ Traveled", "#10b981"),
            ("★ Moving Agent", self.ACCENT_GOLD),
        ]
        for text, color in legends:
            tk.Label(
                legend_frame, text=text, bg=self.BG_PANEL, fg=color,
                font=("Helvetica", 12, "bold")
            ).pack(side=tk.LEFT, padx=8)

        # Right Column: Slide 19 Pseudocode + Agent Memory + Log
        right_frame = tk.Frame(main_pane, bg=self.BG_MAIN, width=640)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right_frame.pack_propagate(False)

        # 2A. Slide 19 Pseudocode Panel
        code_panel = tk.Frame(right_frame, bg=self.BG_PANEL, padx=12, pady=10)
        code_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        title_row = tk.Frame(code_panel, bg=self.BG_PANEL)
        title_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        tk.Label(
            title_row,
            text="Slide 19: SIMPLE-PROBLEM-SOLVING-AGENT",
            bg=self.BG_PANEL,
            fg=self.ACCENT_BLUE,
            font=("Helvetica", 14, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            title_row,
            text="Offline ('Eyes Closed')",
            bg=self.BG_PANEL,
            fg=self.ACCENT_GOLD,
            font=("Helvetica", 13, "bold"),
        ).pack(side=tk.RIGHT)

        # Arabic lecture notes banner matching the handwritten notes on the slide
        tk.Label(
            code_panel,
            text="ملاحظات المحاضرة: نفرض الأحداث محددة • نبني نموذجاً • نكتب نموذج منه بالبايثون",
            bg="#0f172a",
            fg="#bae6fd",
            font=("Helvetica", 13, "bold"),
            pady=4,
        ).pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.code_labels: List[tk.Label] = []
        code_box = tk.Frame(code_panel, bg=self.BG_CODE, padx=8, pady=6)
        code_box.pack(side=tk.TOP, fill=tk.X)

        for idx, line in enumerate(SLIDE_PSEUDOCODE_LINES):
            lbl = tk.Label(
                code_box,
                text=f"   {line}",
                bg=self.BG_CODE,
                fg="#e2e8f0",
                font=("Courier", 12, "bold"),
                anchor="w",
                padx=4,
                pady=2,
            )
            lbl.pack(side=tk.TOP, fill=tk.X)
            self.code_labels.append(lbl)

        # 2B. Live Static Variables & Memory Panel
        mem_panel = tk.Frame(right_frame, bg=self.BG_PANEL, padx=12, pady=10)
        mem_panel.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        tk.Label(
            mem_panel,
            text="Agent Persistent (static) Memory & Variables",
            bg=self.BG_PANEL,
            fg=self.ACCENT_GOLD,
            font=("Helvetica", 14, "bold"),
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.var_labels: Dict[str, tk.Label] = {}
        fields = [
            ("percept", "Current Percept:", "#38bdf8"),
            ("state", "state (UPDATE-STATE):", "#4ade80"),
            ("goal", "goal (FORMULATE-GOAL):", "#f87171"),
            ("problem", "problem (FORMULATE):", "#c084fc"),
            ("action", "action (FIRST(seq)):", "#facc15"),
        ]
        for key, label_title, color in fields:
            row = tk.Frame(mem_panel, bg=self.BG_PANEL, pady=2)
            row.pack(side=tk.TOP, fill=tk.X)
            tk.Label(
                row, text=label_title, bg=self.BG_PANEL, fg=self.FG_MUTED,
                font=("Courier", 13, "bold"), width=23, anchor="w"
            ).pack(side=tk.LEFT)
            val_lbl = tk.Label(
                row, text="null", bg="#0f172a", fg=color,
                font=("Courier", 13, "bold"), anchor="w", padx=8, pady=3
            )
            val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.var_labels[key] = val_lbl

        # Sequence `seq` visual queue box
        tk.Label(
            mem_panel,
            text="seq (Planned Action Sequence Queue — FIRST / REST):",
            bg=self.BG_PANEL,
            fg=self.FG_MUTED,
            font=("Courier", 13, "bold"),
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X, pady=(6, 2))

        self.seq_text = tk.Text(
            mem_panel,
            height=3,
            bg="#0f172a",
            fg="#38bdf8",
            font=("Courier", 13, "bold"),
            bd=0,
            padx=8,
            pady=4,
            wrap=tk.WORD,
        )
        self.seq_text.pack(side=tk.TOP, fill=tk.X)
        self.seq_text.config(state=tk.DISABLED)

        # 2C. Step-by-Step Execution Log Panel
        log_panel = tk.Frame(right_frame, bg=self.BG_PANEL, padx=12, pady=8)
        log_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_panel, bg=self.BG_PANEL)
        log_header.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        tk.Label(
            log_header,
            text="Step-by-Step Execution Trace Log",
            bg=self.BG_PANEL,
            fg=self.ACCENT_BLUE,
            font=("Helvetica", 14, "bold"),
        ).pack(side=tk.LEFT)

        self.lbl_step_counter = tk.Label(
            log_header,
            text="Step 1 / 1",
            bg=self.BG_PANEL,
            fg=self.ACCENT_GOLD,
            font=("Helvetica", 13, "bold"),
        )
        self.lbl_step_counter.pack(side=tk.RIGHT)

        self.log_text = tk.Text(
            log_panel,
            bg="#090d16",
            fg="#f1f5f9",
            font=("Courier", 13, "bold"),
            bd=0,
            padx=8,
            pady=6,
            wrap=tk.WORD,
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_panel, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)

    # =========================================================================
    # EPISODE GENERATION & STEP CONTROL
    # =========================================================================

    def _on_env_changed(self, _event=None):
        new_name = self.env_var.get()
        if new_name in self.environments:
            self.current_env_name = new_name
            self.env = self.environments[new_name]
            nodes_list = list(self.env.nodes.keys())
            self.start_combo.config(values=nodes_list)
            self.goal_combo.config(values=nodes_list)
            self.start_var.set(self.env.default_start)
            self.goal_var.set(self.env.default_goal)
            self._rebuild_episode()

    def _rebuild_episode(self):
        self._stop_play()
        start_state = self.start_var.get()
        goal_state = self.goal_var.get()
        algo = self.algo_var.get()

        if start_state not in self.env.nodes:
            start_state = self.env.default_start
            self.start_var.set(start_state)
        if goal_state not in self.env.nodes:
            goal_state = self.env.default_goal
            self.goal_var.set(goal_state)

        self.microsteps = generate_full_episode_microsteps(
            env=self.env,
            start_state=start_state,
            goal_state=goal_state,
            search_algorithm=algo,
            include_search_expansion_steps=True,
        )
        self.current_step_idx = 0
        self._sync_state_to_step(0, animate_move=False)

    def _on_reset(self):
        self._stop_play()
        self.current_step_idx = 0
        self._sync_state_to_step(0, animate_move=False)

    def _on_prev_step(self):
        self._stop_play()
        if self.current_step_idx > 0:
            self.current_step_idx -= 1
            self._sync_state_to_step(self.current_step_idx, animate_move=False)

    def _on_next_step(self):
        if self.current_step_idx < len(self.microsteps) - 1:
            self.current_step_idx += 1
            self._sync_state_to_step(self.current_step_idx, animate_move=True)
        else:
            self._stop_play()

    def _on_next_action_move(self):
        """Jumps to the next step where the agent executes an action on the map."""
        self._stop_play()
        idx = self.current_step_idx + 1
        while idx < len(self.microsteps):
            if self.microsteps[idx].move_to is not None or idx == len(self.microsteps) - 1:
                self.current_step_idx = idx
                self._sync_state_to_step(self.current_step_idx, animate_move=True)
                return
            idx += 1

    def _toggle_play(self):
        if self.is_playing:
            self._stop_play()
        else:
            if self.current_step_idx >= len(self.microsteps) - 1:
                self.current_step_idx = 0
            self.is_playing = True
            self.btn_play.config(text="⏸ Pause", bg="#d97706")
            self._play_loop()

    def _stop_play(self):
        self.is_playing = False
        self.btn_play.config(text="⏵ Auto Play", bg="#16a34a")
        if self.play_after_id:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def _play_loop(self):
        if not self.is_playing:
            return
        if self.current_step_idx < len(self.microsteps) - 1:
            self.current_step_idx += 1
            self._sync_state_to_step(self.current_step_idx, animate_move=True)
            delay = int(self.speed_scale.get())
            self.play_after_id = self.root.after(delay, self._play_loop)
        else:
            self._stop_play()

    # =========================================================================
    # SYNCHRONIZE UI & ANIMATE AGENT MOVEMENT
    # =========================================================================

    def _sync_state_to_step(self, step_idx: int, animate_move: bool = True):
        if not self.microsteps:
            return

        step = self.microsteps[step_idx]

        # Recompute traveled edges up to `step_idx`
        self.traveled_edges = []
        self.visited_states = [self.start_var.get()]
        for i in range(step_idx + 1):
            s = self.microsteps[i]
            if s.move_from and s.move_to:
                self.traveled_edges.append((s.move_from, s.move_to))
                self.visited_states.append(s.move_to)

        # Update top banner
        self.lbl_phase.config(text=f"Step {step_idx + 1}/{len(self.microsteps)} — {step.phase_name}")
        self.lbl_explanation.config(text=step.explanation)
        self.lbl_step_counter.config(text=f"Step {step_idx + 1} / {len(self.microsteps)}")

        # Highlight active pseudocode line
        for idx, lbl in enumerate(self.code_labels):
            if idx == step.line_index:
                lbl.config(
                    bg="#1e3a8a",
                    fg=self.ACCENT_GOLD,
                    font=("Courier", 12, "bold"),
                    text=f" ▶ {SLIDE_PSEUDOCODE_LINES[idx]}",
                )
            else:
                lbl.config(
                    bg=self.BG_CODE,
                    fg="#e2e8f0",
                    font=("Courier", 12, "bold"),
                    text=f"   {SLIDE_PSEUDOCODE_LINES[idx]}",
                )

        # Update Static Memory Inspector labels
        self.var_labels["percept"].config(text=str(step.percept))
        self.var_labels["state"].config(text=str(step.state) if step.state else "null")
        self.var_labels["goal"].config(text=str(step.goal) if step.goal else "null")
        self.var_labels["problem"].config(text=step.problem_str)
        self.var_labels["action"].config(text=str(step.action) if step.action else "null")

        # Update `seq` queue box
        self.seq_text.config(state=tk.NORMAL)
        self.seq_text.delete("1.0", tk.END)
        if not step.seq:
            self.seq_text.insert(tk.END, "[]  (empty action sequence)")
        else:
            first_act = step.seq[0]
            rest_acts = step.seq[1:]
            seq_formatted = f"FIRST: {first_act}\nREST : {rest_acts}"
            self.seq_text.insert(tk.END, seq_formatted)
        self.seq_text.config(state=tk.DISABLED)

        # Update execution log up to current step
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for i in range(step_idx + 1):
            m = self.microsteps[i]
            prefix = "👉 " if i == step_idx else "   "
            self.log_text.insert(
                tk.END, f"{prefix}[{i + 1:02d}] {m.phase_name}\n      {m.explanation}\n\n"
            )
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

        # Handle smooth agent animation on map when moving between states
        if self.anim_after_id:
            self.root.after_cancel(self.anim_after_id)
            self.anim_after_id = None

        if animate_move and step.move_from and step.move_to:
            start_xy = self.env.nodes[step.move_from]
            end_xy = self.env.nodes[step.move_to]
            self._animate_agent_transition(start_xy, end_xy, frame=0, total_frames=12)
        else:
            loc = step.agent_location
            if loc in self.env.nodes:
                self.agent_draw_pos = self.env.nodes[loc]
            self._draw_map()

    def _animate_agent_transition(
        self,
        start_xy: Tuple[float, float],
        end_xy: Tuple[float, float],
        frame: int,
        total_frames: int,
    ):
        t = (frame + 1) / total_frames
        # Smooth ease-in-out interpolation
        smooth_t = t * t * (3.0 - 2.0 * t)
        nx = start_xy[0] + (end_xy[0] - start_xy[0]) * smooth_t
        ny = start_xy[1] + (end_xy[1] - start_xy[1]) * smooth_t
        self.agent_draw_pos = (nx, ny)
        self._draw_map()

        if frame + 1 < total_frames:
            self.anim_after_id = self.root.after(
                25,
                lambda: self._animate_agent_transition(
                    start_xy, end_xy, frame + 1, total_frames
                ),
            )

    # =========================================================================
    # INTERACTIVE MAP CLICK HANDLERS
    # =========================================================================

    def _find_clicked_node(self, event_x: int, event_y: int) -> Optional[str]:
        w = max(self.canvas.winfo_width(), 100)
        h = max(self.canvas.winfo_height(), 100)
        best_node = None
        best_dist = 32.0
        for name, (nx, ny) in self.env.nodes.items():
            px, py = nx * w, ny * h
            d = math.hypot(event_x - px, event_y - py)
            if d < best_dist:
                best_dist = d
                best_node = name
        return best_node

    def _on_canvas_left_click(self, event):
        clicked = self._find_clicked_node(event.x, event.y)
        if clicked:
            self.start_var.set(clicked)
            self._rebuild_episode()

    def _on_canvas_right_click(self, event):
        clicked = self._find_clicked_node(event.x, event.y)
        if clicked:
            self.goal_var.set(clicked)
            self._rebuild_episode()

    # =========================================================================
    # MAP CANVAS DRAWING
    # =========================================================================

    def _draw_map(self):
        self.canvas.delete("all")
        w = max(self.canvas.winfo_width(), 400)
        h = max(self.canvas.winfo_height(), 300)

        if not self.microsteps:
            return

        step = self.microsteps[self.current_step_idx]
        start_node = self.start_var.get()
        goal_node = self.goal_var.get()

        # Draw subtle background grid for aesthetics
        for gx in range(0, w, 40):
            self.canvas.create_line(gx, 0, gx, h, fill="#111827", width=1)
        for gy in range(0, h, 40):
            self.canvas.create_line(0, gy, w, gy, fill="#111827", width=1)

        # If 2D Grid Maze World, draw obstacle cells first
        if self.env.grid_size:
            rows, cols = self.env.grid_size
            cell_w = (0.80 * w) / max(1, cols - 1)
            cell_h = (0.76 * h) / max(1, rows - 1)
            for r, c in self.env.obstacles:
                nx = 0.10 + (c / max(1, cols - 1)) * 0.80
                ny = 0.12 + (r / max(1, rows - 1)) * 0.76
                cx, cy = nx * w, ny * h
                rx, ry = cell_w * 0.36, cell_h * 0.36
                self.canvas.create_rectangle(
                    cx - rx, cy - ry, cx + rx, cy + ry,
                    fill="#334155", outline="#475569", width=2
                )
                self.canvas.create_text(
                    cx, cy, text="WALL", fill="#cbd5e1", font=("Helvetica", 12, "bold")
                )

        # 1. Draw all undirected edges (roads)
        drawn_pairs = set()
        for u, neighbors in self.env.edges.items():
            for v, cost in neighbors.items():
                pair = tuple(sorted((u, v)))
                if pair in drawn_pairs:
                    continue
                drawn_pairs.add(pair)
                x1, y1 = self.env.nodes[u][0] * w, self.env.nodes[u][1] * h
                x2, y2 = self.env.nodes[v][0] * w, self.env.nodes[v][1] * h
                self.canvas.create_line(x1, y1, x2, y2, fill="#334155", width=3)

                # Show road distance label on Romania map
                if not self.env.grid_size:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_rectangle(
                        mx - 16, my - 11, mx + 16, my + 11,
                        fill="#0f172a", outline="#334155"
                    )
                    self.canvas.create_text(
                        mx, my, text=str(cost),
                        fill="#e2e8f0", font=("Helvetica", 12, "bold")
                    )

        # 2. Draw Search Tree Path So Far (during SEARCH phase)
        if step.search_step and len(step.search_step.path_so_far) > 1:
            spath = step.search_step.path_so_far
            for i in range(len(spath) - 1):
                u, v = spath[i], spath[i + 1]
                x1, y1 = self.env.nodes[u][0] * w, self.env.nodes[u][1] * h
                x2, y2 = self.env.nodes[v][0] * w, self.env.nodes[v][1] * h
                self.canvas.create_line(
                    x1, y1, x2, y2, fill=self.ACCENT_PURPLE, width=5, dash=(6, 4)
                )

        # 3. Draw Planned Path `seq` (once SEARCH has completed)
        if len(step.planned_state_path) > 1:
            ppath = step.planned_state_path
            for i in range(len(ppath) - 1):
                u, v = ppath[i], ppath[i + 1]
                x1, y1 = self.env.nodes[u][0] * w, self.env.nodes[u][1] * h
                x2, y2 = self.env.nodes[v][0] * w, self.env.nodes[v][1] * h
                self.canvas.create_line(
                    x1, y1, x2, y2, fill=self.ACCENT_BLUE, width=5, dash=(8, 4)
                )

        # 4. Draw Traveled Edges (Executed Actions)
        for u, v in self.traveled_edges:
            x1, y1 = self.env.nodes[u][0] * w, self.env.nodes[u][1] * h
            x2, y2 = self.env.nodes[v][0] * w, self.env.nodes[v][1] * h
            self.canvas.create_line(
                x1, y1, x2, y2, fill="#10b981", width=7, arrow=tk.LAST, arrowshape=(14, 18, 6)
            )

        # 5. Draw Nodes (Cities / Grid States)
        explored_set = set(step.search_step.explored_nodes) if step.search_step else set()
        frontier_set = set(step.search_step.frontier_nodes) if step.search_step else set()
        current_search_node = step.search_step.current_node if step.search_step else None

        node_radius = 18 if not self.env.grid_size else 18
        for name, (nx, ny) in self.env.nodes.items():
            cx, cy = nx * w, ny * h

            # Determine node fill & border color
            fill_color = "#1e293b"
            outline_color = "#64748b"
            outline_w = 3

            if name in frontier_set:
                fill_color = "#7c2d12"
                outline_color = self.ACCENT_ORANGE
                outline_w = 4
            if name in explored_set:
                fill_color = "#4c1d95"
                outline_color = self.ACCENT_PURPLE
                outline_w = 4
            if name in self.visited_states and len(self.traveled_edges) > 0:
                fill_color = "#064e3b"
                outline_color = "#10b981"
                outline_w = 4
            if name == start_node:
                fill_color = "#14532d"
                outline_color = self.ACCENT_GREEN
                outline_w = 4
            if name == goal_node:
                fill_color = "#7f1d1d"
                outline_color = self.ACCENT_RED
                outline_w = 4

            # Highlight node currently being expanded by SEARCH(problem)
            if name == current_search_node:
                self.canvas.create_oval(
                    cx - node_radius - 7,
                    cy - node_radius - 7,
                    cx + node_radius + 7,
                    cy + node_radius + 7,
                    outline=self.ACCENT_GOLD,
                    width=3,
                    dash=(4, 2),
                )

            self.canvas.create_oval(
                cx - node_radius,
                cy - node_radius,
                cx + node_radius,
                cy + node_radius,
                fill=fill_color,
                outline=outline_color,
                width=outline_w,
            )

            # Label city/cell name with larger font
            if not self.env.grid_size:
                h_val = self.env.heuristic(name, goal_node)
                self.canvas.create_text(
                    cx,
                    cy - node_radius - 14,
                    text=name,
                    fill=self.FG_TEXT,
                    font=("Helvetica", 14, "bold"),
                )
                if self.algo_var.get() == "A*":
                    self.canvas.create_text(
                        cx,
                        cy + node_radius + 13,
                        text=f"h={h_val:.0f}",
                        fill="#38bdf8",
                        font=("Helvetica", 11, "bold"),
                    )
            else:
                self.canvas.create_text(
                    cx, cy, text=name, fill=self.FG_TEXT, font=("Helvetica", 11, "bold")
                )

        # 6. Draw Moving Agent Marker
        ax, ay = self.agent_draw_pos[0] * w, self.agent_draw_pos[1] * h
        # Outer glow ring
        self.canvas.create_oval(
            ax - 25, ay - 25, ax + 25, ay + 25,
            outline=self.ACCENT_GOLD, width=3
        )
        # Inner solid agent token
        self.canvas.create_oval(
            ax - 15, ay - 15, ax + 15, ay + 15,
            fill=self.ACCENT_GOLD, outline="#0f172a", width=2
        )
        self.canvas.create_text(
            ax, ay, text="★", fill="#0f172a", font=("Helvetica", 15, "bold")
        )
        # Floating AGENT badge above the moving agent
        self.canvas.create_rectangle(
            ax - 38, ay - 46, ax + 38, ay - 26,
            fill=self.ACCENT_GOLD, outline="#0f172a", width=2
        )
        self.canvas.create_text(
            ax, ay - 36, text="AGENT", fill="#0f172a", font=("Helvetica", 12, "bold")
        )


def launch_gui():
    root = tk.Tk()
    app = SimpleProblemSolvingAgentGUI(root)
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
