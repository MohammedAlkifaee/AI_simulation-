"""
generate_demo_video.py
----------------------
Generates a Full HD (1920x1080, 30 FPS, H.264 MP4) demonstration video of the
Simple Problem-Solving Agent (Slide 19) running step-by-step on:
1. The AIMA Chapter 3 Romania Road Map (Arad -> Bucharest using A* Search)
2. The 2D Grid Maze World ((0,0) -> (5,7) using BFS)

Outputs:
- /Users/macbookshop/Desktop/AI project/agent/AI_Agent_Simulation_Demo.mp4
- /Users/macbookshop/Desktop/AI_Agent_Simulation_Demo.mp4
"""

import math
import shutil
import subprocess
import textwrap
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

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

WIDTH = 1920
HEIGHT = 1080
FPS = 30

# Color Palette (Matching visualizer_gui.py)
BG_MAIN = (15, 23, 42)        # #0f172a
BG_PANEL = (30, 41, 59)       # #1e293b
BG_CANVAS = (11, 17, 32)      # #0b1120
BG_CODE = (9, 13, 22)         # #090d16
BG_BANNER = (23, 37, 84)      # #172554
FG_TEXT = (248, 250, 252)     # #f8fafc
FG_MUTED = (203, 213, 225)    # #cbd5e1
ACCENT_BLUE = (56, 189, 248)  # #38bdf8
ACCENT_GOLD = (250, 204, 21)  # #facc15
ACCENT_GREEN = (34, 197, 94)  # #22c55e
ACCENT_EMERALD = (16, 185, 129)
ACCENT_PURPLE = (168, 85, 247)
ACCENT_ORANGE = (249, 115, 22)
ACCENT_RED = (239, 68, 68)


def load_fonts() -> Dict[str, ImageFont.FreeTypeFont]:
    sans_path = "/System/Library/Fonts/HelveticaNeue.ttc"
    mono_path = "/System/Library/Fonts/Menlo.ttc"
    return {
        "title": ImageFont.truetype(sans_path, 22, index=1),
        "topbar": ImageFont.truetype(sans_path, 20, index=1),
        "phase": ImageFont.truetype(sans_path, 24, index=1),
        "explain": ImageFont.truetype(sans_path, 19, index=0),
        "section": ImageFont.truetype(sans_path, 20, index=1),
        "code": ImageFont.truetype(mono_path, 16, index=1),
        "var_lbl": ImageFont.truetype(mono_path, 17, index=1),
        "var_val": ImageFont.truetype(mono_path, 17, index=1),
        "log": ImageFont.truetype(mono_path, 16, index=0),
        "node": ImageFont.truetype(sans_path, 19, index=1),
        "heur": ImageFont.truetype(sans_path, 15, index=1),
        "edge": ImageFont.truetype(sans_path, 16, index=1),
        "legend": ImageFont.truetype(sans_path, 17, index=1),
        "badge": ImageFont.truetype(sans_path, 16, index=1),
    }


def draw_arrow_line(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Tuple[int, int, int],
    width: int = 6,
    node_radius: float = 22.0,
):
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    # Place arrowhead slightly before target node boundary
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist > node_radius + 10:
        tip_x = x2 - math.cos(angle) * (node_radius + 2)
        tip_y = y2 - math.sin(angle) * (node_radius + 2)
        arrow_len = 16
        arrow_angle = 0.45
        p1 = (
            tip_x - arrow_len * math.cos(angle - arrow_angle),
            tip_y - arrow_len * math.sin(angle - arrow_angle),
        )
        p2 = (
            tip_x - arrow_len * math.cos(angle + arrow_angle),
            tip_y - arrow_len * math.sin(angle + arrow_angle),
        )
        draw.polygon([(tip_x, tip_y), p1, p2], fill=color)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: Tuple[int, int, int],
    width: int = 5,
    dash_len: int = 12,
    gap_len: int = 8,
):
    total_len = math.hypot(x2 - x1, y2 - y1)
    if total_len == 0:
        return
    dx = (x2 - x1) / total_len
    dy = (y2 - y1) / total_len
    pos = 0.0
    while pos < total_len:
        end_pos = min(pos + dash_len, total_len)
        sx, sy = x1 + dx * pos, y1 + dy * pos
        ex, ey = x1 + dx * end_pos, y1 + dy * end_pos
        draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
        pos += dash_len + gap_len


def render_frame(
    env: MapEnvironment,
    algo_name: str,
    microsteps: List[AgentMicroStep],
    step_idx: int,
    agent_pos: Tuple[float, float],
    traveled_edges: List[Tuple[str, str]],
    visited_states: List[str],
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_MAIN)
    draw = ImageDraw.Draw(img)

    step = microsteps[step_idx]
    start_node = env.default_start
    goal_node = env.default_goal

    # 1. macOS Window Title Bar
    draw.rectangle([0, 0, WIDTH, 42], fill=(30, 41, 59))
    draw.ellipse([18, 13, 34, 29], fill=(239, 68, 68))
    draw.ellipse([44, 13, 60, 29], fill=(250, 204, 21))
    draw.ellipse([70, 13, 86, 29], fill=(34, 197, 94))
    title_str = "AI Lecture Project: Simple Problem-Solving Agent Visualizer (AIMA Slide 19)"
    draw.text((WIDTH // 2 - 370, 9), title_str, fill=FG_TEXT, font=fonts["title"])

    # 2. Top Control Bar
    draw.rectangle([16, 52, WIDTH - 16, 114], fill=BG_PANEL, outline=(51, 65, 85), width=2)
    top_y = 70
    draw.text((34, top_y), f"Map: {env.name}", fill=ACCENT_BLUE, font=fonts["topbar"])
    draw.text((520, top_y), f"SEARCH: {algo_name}", fill=ACCENT_GOLD, font=fonts["topbar"])
    draw.text((730, top_y), f"Start: {start_node}", fill=ACCENT_GREEN, font=fonts["topbar"])
    draw.text((950, top_y), f"Goal: {goal_node}", fill=ACCENT_RED, font=fonts["topbar"])

    # Simulated Active Control Buttons
    buttons = [
        ("⏮ Reset", (71, 85, 105), 1200),
        ("◀ Prev", (51, 65, 85), 1325),
        ("▶ Next Step", (2, 132, 199), 1435),
        ("⏭ Next Move", (13, 148, 136), 1595),
        ("⏵ Auto Play [ACTIVE]", (22, 163, 74), 1755),
    ]
    for b_text, b_col, bx in buttons:
        bw = 135 if "ACTIVE" in b_text else 115
        draw.rounded_rectangle([bx, 62, bx + bw, 104], radius=6, fill=b_col)
        draw.text((bx + 10, 72), b_text.replace(" [ACTIVE]", ""), fill=FG_TEXT, font=fonts["legend"])

    # 3. Left Column: Banner + Map Canvas + Legend
    left_x1, left_y1, left_x2, left_y2 = 16, 126, 1145, HEIGHT - 16
    draw.rectangle([left_x1, left_y1, left_x2, left_y2], fill=BG_PANEL, outline=(51, 65, 85), width=2)

    # Banner
    draw.rectangle([left_x1 + 2, left_y1 + 2, left_x2 - 2, left_y1 + 92], fill=BG_BANNER)
    phase_txt = f"Step {step_idx + 1}/{len(microsteps)} — {step.phase_name}"
    draw.text((left_x1 + 18, left_y1 + 12), phase_txt, fill=ACCENT_GOLD, font=fonts["phase"])
    wrapped_exp = textwrap.shorten(step.explanation, width=105, placeholder="...")
    draw.text((left_x1 + 18, left_y1 + 50), wrapped_exp, fill=(224, 242, 254), font=fonts["explain"])

    # Map Canvas Region
    cx1, cy1, cx2, cy2 = left_x1 + 4, left_y1 + 94, left_x2 - 4, left_y2 - 48
    cw = cx2 - cx1
    ch = cy2 - cy1
    draw.rectangle([cx1, cy1, cx2, cy2], fill=BG_CANVAS)

    # Subtle background grid
    for gx in range(cx1, cx2, 48):
        draw.line([(gx, cy1), (gx, cy2)], fill=(17, 24, 39), width=1)
    for gy in range(cy1, cy2, 48):
        draw.line([(cx1, gy), (cx2, gy)], fill=(17, 24, 39), width=1)

    # Draw 2D Grid Walls if Grid environment
    if env.grid_size:
        rows, cols = env.grid_size
        cell_w = (0.80 * cw) / max(1, cols - 1)
        cell_h = (0.76 * ch) / max(1, rows - 1)
        for r, c in env.obstacles:
            nx = 0.10 + (c / max(1, cols - 1)) * 0.80
            ny = 0.12 + (r / max(1, rows - 1)) * 0.76
            px, py = cx1 + nx * cw, cy1 + ny * ch
            rx, ry = cell_w * 0.36, cell_h * 0.36
            draw.rectangle(
                [px - rx, py - ry, px + rx, py + ry],
                fill=(51, 65, 85),
                outline=(100, 116, 139),
                width=2,
            )
            draw.text((px - 24, py - 10), "WALL", fill=FG_MUTED, font=fonts["edge"])

    # Draw Edges
    drawn_pairs = set()
    for u, neighbors in env.edges.items():
        for v, cost in neighbors.items():
            pair = tuple(sorted((u, v)))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            x1 = cx1 + env.nodes[u][0] * cw
            y1 = cy1 + env.nodes[u][1] * ch
            x2 = cx1 + env.nodes[v][0] * cw
            y2 = cy1 + env.nodes[v][1] * ch
            draw.line([(x1, y1), (x2, y2)], fill=(51, 65, 85), width=4)

            if not env.grid_size:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                draw.rectangle([mx - 20, my - 13, mx + 20, my + 13], fill=BG_MAIN, outline=(51, 65, 85))
                draw.text((mx - 13, my - 10), str(cost), fill=FG_MUTED, font=fonts["edge"])

    # Draw Search Path So Far
    if step.search_step and len(step.search_step.path_so_far) > 1:
        spath = step.search_step.path_so_far
        for i in range(len(spath) - 1):
            u, v = spath[i], spath[i + 1]
            x1, y1 = cx1 + env.nodes[u][0] * cw, cy1 + env.nodes[u][1] * ch
            x2, y2 = cx1 + env.nodes[v][0] * cw, cy1 + env.nodes[v][1] * ch
            draw_dashed_line(draw, x1, y1, x2, y2, ACCENT_PURPLE, width=6)

    # Draw Planned Path `seq`
    if len(step.planned_state_path) > 1:
        ppath = step.planned_state_path
        for i in range(len(ppath) - 1):
            u, v = ppath[i], ppath[i + 1]
            x1, y1 = cx1 + env.nodes[u][0] * cw, cy1 + env.nodes[u][1] * ch
            x2, y2 = cx1 + env.nodes[v][0] * cw, cy1 + env.nodes[v][1] * ch
            draw_dashed_line(draw, x1, y1, x2, y2, ACCENT_BLUE, width=6)

    # Draw Traveled Edges
    for u, v in traveled_edges:
        x1, y1 = cx1 + env.nodes[u][0] * cw, cy1 + env.nodes[u][1] * ch
        x2, y2 = cx1 + env.nodes[v][0] * cw, cy1 + env.nodes[v][1] * ch
        draw_arrow_line(draw, x1, y1, x2, y2, ACCENT_EMERALD, width=8, node_radius=22.0)

    # Draw Nodes
    explored_set = set(step.search_step.explored_nodes) if step.search_step else set()
    frontier_set = set(step.search_step.frontier_nodes) if step.search_step else set()
    current_search_node = step.search_step.current_node if step.search_step else None
    node_r = 22

    for name, (nx, ny) in env.nodes.items():
        px, py = cx1 + nx * cw, cy1 + ny * ch
        fill_col = (30, 41, 59)
        out_col = (100, 116, 139)

        if name in frontier_set:
            fill_col = (124, 45, 18)
            out_col = ACCENT_ORANGE
        if name in explored_set:
            fill_col = (76, 29, 149)
            out_col = ACCENT_PURPLE
        if name in visited_states and len(traveled_edges) > 0:
            fill_col = (6, 78, 59)
            out_col = ACCENT_EMERALD
        if name == start_node:
            fill_col = (20, 83, 45)
            out_col = ACCENT_GREEN
        if name == goal_node:
            fill_col = (127, 29, 29)
            out_col = ACCENT_RED

        if name == current_search_node:
            draw.ellipse(
                [px - node_r - 8, py - node_r - 8, px + node_r + 8, py + node_r + 8],
                outline=ACCENT_GOLD,
                width=3,
            )

        draw.ellipse(
            [px - node_r, py - node_r, px + node_r, py + node_r],
            fill=fill_col,
            outline=out_col,
            width=4,
        )

        if not env.grid_size:
            h_val = env.heuristic(name, goal_node)
            tw = len(name) * 5.5
            draw.text((px - tw, py - node_r - 24), name, fill=FG_TEXT, font=fonts["node"])
            if algo_name == "A*":
                draw.text(
                    (px - 22, py + node_r + 5),
                    f"h={h_val:.0f}",
                    fill=ACCENT_BLUE,
                    font=fonts["heur"],
                )
        else:
            draw.text((px - 18, py - 9), name, fill=FG_TEXT, font=fonts["heur"])

    # Draw Moving Agent Marker
    ax = cx1 + agent_pos[0] * cw
    ay = cy1 + agent_pos[1] * ch
    draw.ellipse([ax - 29, ay - 29, ax + 29, ay + 29], outline=ACCENT_GOLD, width=4)
    draw.ellipse([ax - 17, ay - 17, ax + 17, ay + 17], fill=ACCENT_GOLD, outline=BG_MAIN, width=3)
    draw.rounded_rectangle([ax - 44, ay - 56, ax + 44, ay - 31], radius=5, fill=ACCENT_GOLD, outline=BG_MAIN, width=2)
    draw.text((ax - 30, ay - 52), "AGENT", fill=BG_MAIN, font=fonts["badge"])

    # Legend Bar at Bottom of Left Column
    leg_y = left_y2 - 36
    legends = [
        ("● Start", ACCENT_GREEN),
        ("● Goal", ACCENT_RED),
        ("● Explored", ACCENT_PURPLE),
        ("● Frontier", ACCENT_ORANGE),
        ("--- Planned (seq)", ACCENT_BLUE),
        ("━━ Traveled", ACCENT_EMERALD),
        ("★ Moving Agent", ACCENT_GOLD),
    ]
    lx = left_x1 + 20
    for l_txt, l_col in legends:
        draw.text((lx, leg_y), l_txt, fill=l_col, font=fonts["legend"])
        lx += len(l_txt) * 10 + 28

    # 4. Right Column: Slide 19 Pseudocode + Static Memory + Trace Log
    rx1, ry1, rx2, ry2 = 1158, 126, WIDTH - 16, HEIGHT - 16

    # 4A. Slide 19 Pseudocode Panel
    p1_y2 = ry1 + 382
    draw.rectangle([rx1, ry1, rx2, p1_y2], fill=BG_PANEL, outline=(51, 65, 85), width=2)
    draw.text(
        (rx1 + 16, ry1 + 12),
        "Slide 19: SIMPLE-PROBLEM-SOLVING-AGENT",
        fill=ACCENT_BLUE,
        font=fonts["section"],
    )
    draw.text(
        (rx2 - 220, ry1 + 12),
        "Offline ('Eyes Closed')",
        fill=ACCENT_GOLD,
        font=fonts["legend"],
    )

    code_box_y1 = ry1 + 46
    draw.rectangle([rx1 + 12, code_box_y1, rx2 - 12, p1_y2 - 12], fill=BG_CODE)
    cy_line = code_box_y1 + 8
    for idx, line in enumerate(SLIDE_PSEUDOCODE_LINES):
        if idx == step.line_index:
            draw.rectangle([rx1 + 16, cy_line - 2, rx2 - 16, cy_line + 22], fill=(30, 58, 138))
            draw.text((rx1 + 20, cy_line), f"▶ {line}", fill=ACCENT_GOLD, font=fonts["code"])
        else:
            draw.text((rx1 + 20, cy_line), f"  {line}", fill=FG_MUTED, font=fonts["code"])
        cy_line += 24

    # 4B. Static Memory & Variables Panel
    p2_y1 = p1_y2 + 12
    p2_y2 = p2_y1 + 310
    draw.rectangle([rx1, p2_y1, rx2, p2_y2], fill=BG_PANEL, outline=(51, 65, 85), width=2)
    draw.text(
        (rx1 + 16, p2_y1 + 12),
        "Agent Persistent (static) Memory & Variables",
        fill=ACCENT_GOLD,
        font=fonts["section"],
    )

    fields = [
        ("Current Percept:", str(step.percept), ACCENT_BLUE),
        ("state (UPDATE-STATE):", str(step.state) if step.state else "null", ACCENT_GREEN),
        ("goal (FORMULATE-GOAL):", str(step.goal) if step.goal else "null", ACCENT_RED),
        ("problem (FORMULATE):", step.problem_str, ACCENT_PURPLE),
        ("action (FIRST(seq)):", str(step.action) if step.action else "null", ACCENT_GOLD),
    ]
    fy = p2_y1 + 48
    for label_title, val_str, v_col in fields:
        draw.text((rx1 + 16, fy + 3), label_title, fill=FG_MUTED, font=fonts["var_lbl"])
        draw.rectangle([rx1 + 265, fy, rx2 - 14, fy + 28], fill=BG_MAIN)
        draw.text(
            (rx1 + 273, fy + 3),
            textwrap.shorten(val_str, width=44, placeholder="..."),
            fill=v_col,
            font=fonts["var_val"],
        )
        fy += 33

    draw.text(
        (rx1 + 16, fy + 4),
        "seq (Planned Action Sequence Queue — FIRST / REST):",
        fill=FG_MUTED,
        font=fonts["var_lbl"],
    )
    seq_box_y1 = fy + 28
    draw.rectangle([rx1 + 16, seq_box_y1, rx2 - 14, p2_y2 - 12], fill=BG_MAIN)
    if not step.seq:
        draw.text(
            (rx1 + 24, seq_box_y1 + 12),
            "[]  (empty action sequence)",
            fill=ACCENT_BLUE,
            font=fonts["var_val"],
        )
    else:
        first_s = f"FIRST: {step.seq[0]}"
        rest_s = textwrap.shorten(f"REST : {step.seq[1:]}", width=66, placeholder="...]")
        draw.text((rx1 + 24, seq_box_y1 + 6), first_s, fill=ACCENT_GOLD, font=fonts["var_val"])
        draw.text((rx1 + 24, seq_box_y1 + 30), rest_s, fill=ACCENT_BLUE, font=fonts["var_val"])

    # 4C. Step-by-Step Execution Trace Log Panel
    p3_y1 = p2_y2 + 12
    draw.rectangle([rx1, p3_y1, rx2, ry2], fill=BG_PANEL, outline=(51, 65, 85), width=2)
    draw.text(
        (rx1 + 16, p3_y1 + 12),
        "Step-by-Step Execution Trace Log",
        fill=ACCENT_BLUE,
        font=fonts["section"],
    )
    draw.text(
        (rx2 - 150, p3_y1 + 12),
        f"Step {step_idx + 1} / {len(microsteps)}",
        fill=ACCENT_GOLD,
        font=fonts["legend"],
    )

    log_box_y1 = p3_y1 + 44
    draw.rectangle([rx1 + 12, log_box_y1, rx2 - 12, ry2 - 12], fill=BG_CODE)

    # Show the last 4 steps in the log box
    start_log = max(0, step_idx - 3)
    ly = log_box_y1 + 8
    for i in range(start_log, step_idx + 1):
        m = microsteps[i]
        prefix = "👉 " if i == step_idx else "   "
        col = ACCENT_GOLD if i == step_idx else FG_TEXT
        line1 = f"{prefix}[{i + 1:02d}] {m.phase_name}"
        line2 = "     " + textwrap.shorten(m.explanation, width=65, placeholder="...")
        draw.text((rx1 + 18, ly), line1, fill=col, font=fonts["log"])
        draw.text((rx1 + 18, ly + 20), line2, fill=FG_MUTED, font=fonts["log"])
        ly += 44

    return img


def append_episode_frames(
    ffmpeg_proc: subprocess.Popen,
    env: MapEnvironment,
    algo_name: str,
    hold_frames_normal: int,
    move_anim_frames: int,
    fonts: Dict[str, ImageFont.FreeTypeFont],
):
    microsteps = generate_full_episode_microsteps(
        env=env,
        start_state=env.default_start,
        goal_state=env.default_goal,
        search_algorithm=algo_name,
        include_search_expansion_steps=True,
    )

    traveled_edges: List[Tuple[str, str]] = []
    visited_states: List[str] = [env.default_start]

    for idx, step in enumerate(microsteps):
        if step.move_from and step.move_to:
            start_xy = env.nodes[step.move_from]
            end_xy = env.nodes[step.move_to]
            traveled_edges.append((step.move_from, step.move_to))
            visited_states.append(step.move_to)

            # Smooth animation of the agent moving along the road/grid edge
            for f in range(move_anim_frames):
                t = (f + 1) / move_anim_frames
                smooth_t = t * t * (3.0 - 2.0 * t)
                nx = start_xy[0] + (end_xy[0] - start_xy[0]) * smooth_t
                ny = start_xy[1] + (end_xy[1] - start_xy[1]) * smooth_t
                frame_img = render_frame(
                    env,
                    algo_name,
                    microsteps,
                    idx,
                    (nx, ny),
                    traveled_edges,
                    visited_states,
                    fonts,
                )
                ffmpeg_proc.stdin.write(frame_img.tobytes())

            # Pause briefly after arriving at the node
            final_frame = render_frame(
                env,
                algo_name,
                microsteps,
                idx,
                end_xy,
                traveled_edges,
                visited_states,
                fonts,
            )
            raw_bytes = final_frame.tobytes()
            for _ in range(hold_frames_normal // 2):
                ffmpeg_proc.stdin.write(raw_bytes)
        else:
            agent_xy = env.nodes[step.agent_location]
            frame_img = render_frame(
                env,
                algo_name,
                microsteps,
                idx,
                agent_xy,
                traveled_edges,
                visited_states,
                fonts,
            )
            raw_bytes = frame_img.tobytes()
            count = hold_frames_normal * 2 if idx == len(microsteps) - 1 else hold_frames_normal
            for _ in range(count):
                ffmpeg_proc.stdin.write(raw_bytes)


def main():
    output_project_path = "/Users/macbookshop/Desktop/AI project/agent/AI_Agent_Simulation_Demo.mp4"
    output_desktop_path = "/Users/macbookshop/Desktop/AI_Agent_Simulation_Demo.mp4"

    fonts = load_fonts()

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "18",
        output_project_path,
    ]

    print("Rendering 1080p Full HD simulation video...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Part 1: Classic AIMA Romania Road Map (Arad -> Bucharest with A*)
    romania_env = create_romania_environment()
    append_episode_frames(
        ffmpeg_proc=proc,
        env=romania_env,
        algo_name="A*",
        hold_frames_normal=28,  # ~0.93s per step
        move_anim_frames=26,    # Smooth ~0.87s road movement animation
        fonts=fonts,
    )

    # Part 2: 2D Grid Maze World ((0,0) -> (5,7) with A*)
    grid_env = create_grid_environment()
    append_episode_frames(
        ffmpeg_proc=proc,
        env=grid_env,
        algo_name="A*",
        hold_frames_normal=16,  # Faster pacing for second demo
        move_anim_frames=16,
        fonts=fonts,
    )

    proc.stdin.close()
    stderr_out = proc.stderr.read().decode("utf-8", errors="ignore")
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr_out}")

    shutil.copy2(output_project_path, output_desktop_path)
    print(f"Video saved successfully to:\n  1) {output_project_path}\n  2) {output_desktop_path}")


if __name__ == "__main__":
    main()
