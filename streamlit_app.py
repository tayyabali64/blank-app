import random
from collections import deque

import numpy as np
import streamlit as st

# ==========================================
# PAGE + MOBILE-FIRST STYLING
# ==========================================
st.set_page_config(
    page_title="Grid Escape — Can You Outsmart a Q-Learning AI?",
    page_icon="🧩",
    layout="centered",
)

# Meta description for link previews / SEO (Streamlit has no native API for this,
# so it is injected into the parent document's <head> from a zero-height component)
import streamlit.components.v1 as components

_META_DESCRIPTION = (
    "Race a reinforcement-learning AI through a hard 10x10 maze. "
    "It trained itself with Q-learning and plays a perfect game — match its path, "
    "peek at its Q-table, and prove you can still beat the machine."
)
components.html(
    f"""
    <script>
    const doc = window.parent.document;
    const setMeta = (attr, name, content) => {{
        let el = doc.querySelector(`meta[${{attr}}="${{name}}"]`);
        if (!el) {{
            el = doc.createElement('meta');
            el.setAttribute(attr, name);
            doc.head.appendChild(el);
        }}
        el.setAttribute('content', content);
    }};
    setMeta('name', 'description', {_META_DESCRIPTION!r});
    setMeta('property', 'og:title', 'Grid Escape — Can You Outsmart a Q-Learning AI?');
    setMeta('property', 'og:description', {_META_DESCRIPTION!r});
    </script>
    """,
    height=0,
)

st.markdown(
    """
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 520px;}

    /* Keep columns side-by-side on phones (Streamlit stacks them by default),
       without overriding each column's weighted width */
    div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: 0.4rem !important;}
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {min-width: 0 !important;}

    /* Pin the paddle row's column widths on every screen size
       (Streamlit's mobile media query would otherwise stretch each to 100%) */
    div[data-testid="stHorizontalBlock"]:has(.st-key-left) > div[data-testid="stColumn"]:nth-child(1),
    div[data-testid="stHorizontalBlock"]:has(.st-key-right) > div[data-testid="stColumn"]:nth-child(3) {
        flex: 0 0 13% !important; width: 13% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.st-key-left) > div[data-testid="stColumn"]:nth-child(2) {
        flex: 1 1 auto !important; width: auto !important;
    }

    /* Big touch-friendly buttons */
    div.stButton > button {
        width: 100%;
        min-height: 3.1rem;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 14px;
    }

    /* Tighter vertical rhythm so the move buttons hug the board */
    div[data-testid="stVerticalBlock"] {gap: 0.5rem;}

    /* Tall side paddles flanking the maze */
    .st-key-left button, .st-key-right button {
        height: clamp(240px, 68vw, 370px);
        padding: 0;
    }
    .st-key-up button, .st-key-down button {min-height: 2.8rem;}

    /* Hide the keyboard-shortcut hint (the little arrow) inside buttons */
    div.stButton button kbd {display: none;}

    /* Subtle, quiet move buttons */
    .st-key-up button, .st-key-down button, .st-key-left button, .st-key-right button {
        background: rgba(128, 128, 128, 0.2) !important;
        color: #8A97A5 !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        box-shadow: none;
        font-size: 1.15rem;
        font-weight: 400;
        line-height: 1;
    }
    .st-key-up button:hover, .st-key-down button:hover,
    .st-key-left button:hover, .st-key-right button:hover {
        background: rgba(128, 128, 128, 0.14) !important;
        color: #5D6B77 !important;
    }
    .st-key-up button:active, .st-key-down button:active,
    .st-key-left button:active, .st-key-right button:active {
        background: rgba(128, 128, 128, 0.22) !important;
    }
    .st-key-up button:disabled, .st-key-down button:disabled,
    .st-key-left button:disabled, .st-key-right button:disabled {
        opacity: 0.35;
    }

    .scorecard {
        display: flex; gap: 0.4rem; margin: 0.6rem 0 0.8rem 0;
    }
    .stat {
        flex: 1 1 0; text-align: center; padding: 0.5rem 0.2rem;
        border-radius: 12px; background: rgba(128,128,128,0.12);
    }
    .stat .label {font-size: 0.72rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.04em;}
    .stat .value {font-size: 1.35rem; font-weight: 800; line-height: 1.3;}

    .maze {
        display: grid;
        grid-template-columns: repeat(10, minmax(0, 1fr));
        gap: 3px;
        width: 100%;
        margin: 0 auto;
    }
    .maze .cell {
        aspect-ratio: 1;
        min-width: 0;
        overflow: hidden;
        border-radius: 5px;
        display: flex; align-items: center; justify-content: center;
        font-size: clamp(12px, 4vw, 24px);
        background: #2E4053;
    }
    .maze .wall {
        background: linear-gradient(145deg, #93A1AC, #5D6B77);
        box-shadow: inset 0 2px 3px rgba(255,255,255,0.35), inset 0 -3px 4px rgba(0,0,0,0.35);
        color: #C0392B;
        font-weight: 900;
    }
    .maze .start  {background: #196F3D;}
    .maze .goal   {background: #78281F;}
    .maze .player {background: #D4AC0D;}
    /* The runner emoji faces left by default — mirror it so it runs toward the goal */
    .maze .player.flip {transform: scaleX(-1);}
    .maze .trail {
        background: #3D5A76;
        font-size: clamp(9px, 2.6vw, 13px);
        font-weight: 700;
        color: #D5DBDF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🧩 Grid Escape")
st.caption("Beat the Q-learning AI!")

# ==========================================
# 1. ENVIRONMENT
# ==========================================
GRID_SIZE = 10
NUM_STATES = GRID_SIZE * GRID_SIZE
START_STATE = 0
GOAL_STATE = NUM_STATES - 1
NUM_OBSTACLES = 30  # hard difficulty

ACTION_EFFECTS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right


def get_next_state(state, action, obstacles):
    row, col = divmod(state, GRID_SIZE)
    dr, dc = ACTION_EFFECTS[action]
    new_row, new_col = row + dr, col + dc
    if 0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE:
        next_state = new_row * GRID_SIZE + new_col
        if next_state not in obstacles:
            return next_state
    return state


def bfs_shortest(obstacles):
    """Shortest path length from start to goal, or -1 if blocked."""
    queue = deque([(START_STATE, 0)])
    seen = {START_STATE}
    while queue:
        s, d = queue.popleft()
        if s == GOAL_STATE:
            return d
        for a in range(4):
            ns = get_next_state(s, a, obstacles)
            if ns not in seen:
                seen.add(ns)
                queue.append((ns, d + 1))
    return -1


def generate_solvable_maze():
    """Roll random wall layouts until one has a path from start to goal."""
    valid_tiles = [i for i in range(NUM_STATES) if i not in (START_STATE, GOAL_STATE)]
    while True:
        obstacles = set(random.sample(valid_tiles, NUM_OBSTACLES))
        par = bfs_shortest(obstacles)
        if par != -1:
            return obstacles, par


# ==========================================
# 2. Q-LEARNING AGENT
# ==========================================
ALPHA = 0.25         # learning rate
GAMMA = 0.95         # discount factor
MAX_EPISODES = 2000  # training stops at the FIRST policy that solves the maze
MAX_STEPS = 150
EPS_MIN = 0.05
EPS_DECAY = 0.98


def greedy_rollout(q_table, obstacles):
    """Follow the greedy policy from start; returns the path or None if it loops."""
    state = START_STATE
    path = [state]
    seen = {state}
    while state != GOAL_STATE and len(path) <= NUM_STATES * 2:
        action = int(np.argmax(q_table[state]))
        state = get_next_state(state, action, obstacles)
        if state in seen:
            return None
        seen.add(state)
        path.append(state)
    return path if state == GOAL_STATE else None


def train_q_agent(obstacles):
    """
    Tabular Q-learning with epsilon-greedy exploration, always starting
    episodes from the start cell (no exploring starts). After every episode
    the greedy policy is tested, and training stops at the FIRST policy that
    can solve the maze — the agent gets no time to polish its route, so it
    is honestly trained but not always optimal.

    Returns (q_table, greedy path or None, episodes trained).
    """
    q_table = np.zeros((NUM_STATES, 4))
    epsilon = 1.0

    for episode in range(1, MAX_EPISODES + 1):
        state = START_STATE
        for _ in range(MAX_STEPS):
            if random.random() < epsilon:
                action = random.randrange(4)
            else:
                action = int(np.argmax(q_table[state]))
            next_state = get_next_state(state, action, obstacles)
            reward = 100.0 if next_state == GOAL_STATE else -1.0
            target = reward + GAMMA * q_table[next_state].max()
            q_table[state, action] += ALPHA * (target - q_table[state, action])
            state = next_state
            if state == GOAL_STATE:
                break
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

        path = greedy_rollout(q_table, obstacles)
        if path is not None:
            return q_table, path, episode

    return q_table, None, MAX_EPISODES


# ==========================================
# 3. GAME STATE
# ==========================================
def new_game():
    with st.spinner("🤖 Training Q-learning agent on a fresh maze..."):
        obstacles, par = generate_solvable_maze()
        q_table, ai_path, episodes_trained = train_q_agent(obstacles)
    st.session_state.obstacles = obstacles
    st.session_state.par = par
    st.session_state.q_table = q_table
    st.session_state.ai_path = ai_path
    st.session_state.ai_steps = len(ai_path) - 1 if ai_path else -1
    st.session_state.episodes_trained = episodes_trained
    restart_run()


def restart_run():
    st.session_state.player_state = START_STATE
    st.session_state.moves_count = 0
    st.session_state.trail = set()
    st.session_state.game_over = False
    st.session_state.autopilot_used = False


def make_move(action):
    if st.session_state.game_over:
        return
    prev = st.session_state.player_state
    nxt = get_next_state(prev, action, st.session_state.obstacles)
    if nxt != prev:
        st.session_state.trail.add(prev)
    st.session_state.player_state = nxt
    st.session_state.moves_count += 1
    if nxt == GOAL_STATE:
        st.session_state.game_over = True


def run_autopilot():
    path = st.session_state.ai_path
    if not path:
        return
    st.session_state.trail = set(path[:-1])
    st.session_state.player_state = GOAL_STATE
    st.session_state.moves_count = len(path) - 1
    st.session_state.autopilot_used = True
    st.session_state.game_over = True


if "obstacles" not in st.session_state:
    new_game()

# ==========================================
# 4. SCORECARD
# ==========================================
ai_display = str(st.session_state.ai_steps) if st.session_state.ai_steps != -1 else "—"
st.markdown(
    f"""
    <div class="scorecard">
      <div class="stat"><div class="label">Your moves</div><div class="value">🏃 {st.session_state.moves_count}</div></div>
      <div class="stat"><div class="label">AI (Q-learning)</div><div class="value">🤖 {ai_display}</div></div>
      <div class="stat"><div class="label">Best possible</div><div class="value">⭐ {st.session_state.par}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 5. Q-TABLE & HYPERPARAMETERS VIEWER
# ==========================================
with st.expander("🧠 Show Q-table and hyperparameters"):
    st.markdown(
        f"""
        **Hyperparameters**

        | Parameter | Value |
        |---|---|
        | Learning rate (α) | {ALPHA} |
        | Discount factor (γ) | {GAMMA} |
        | Exploration (ε) | 1.0 → {EPS_MIN} (×{EPS_DECAY} per episode) |
        | Training budget | stops at the **first** policy that solves the maze (max {MAX_EPISODES} episodes) |
        | Episodes trained (this maze) | {st.session_state.episodes_trained} |
        | Max steps per episode | {MAX_STEPS} |
        | Reward | +100 goal · −1 per step |
        """
    )

    st.markdown("**Q-table** — learned value of each action per cell (best action highlighted)")
    import pandas as pd

    free_states = [s for s in range(NUM_STATES) if s not in st.session_state.obstacles and s != GOAL_STATE]
    q = st.session_state.q_table
    df = pd.DataFrame(
        {
            "Cell (row, col)": [f"({s // GRID_SIZE}, {s % GRID_SIZE})" for s in free_states],
            "⬆️ Up": q[free_states, 0].round(1),
            "⬇️ Down": q[free_states, 1].round(1),
            "⬅️ Left": q[free_states, 2].round(1),
            "➡️ Right": q[free_states, 3].round(1),
        }
    )
    action_cols = ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right"]
    st.dataframe(
        df.style.highlight_max(axis=1, subset=action_cols, props="background-color:#196F3D;color:white;").format("{:.1f}", subset=action_cols),
        hide_index=True,
        height=400,
    )

# ==========================================
# 6. MAZE RENDER + DIRECTIONAL PADDLES
#    (up above, down below, left/right flanking the board)
# ==========================================
disabled = st.session_state.game_over

cells_html = []
for cell_id in range(NUM_STATES):
    css, icon = "", ""
    if cell_id == st.session_state.player_state:
        css, icon = ("player", "🎉") if cell_id == GOAL_STATE else ("player flip", "🏃")
    elif cell_id == GOAL_STATE:
        css, icon = "goal", "🏆"
    elif cell_id in st.session_state.obstacles:
        css, icon = "wall", "✕"
    elif cell_id in st.session_state.trail:
        # Reveal the learned Q-value (best action's score) of each box you left
        css = "trail"
        icon = f"{st.session_state.q_table[cell_id].max():.0f}"
    elif cell_id == START_STATE:
        css = "start"
    cells_html.append(f"<div class='cell {css}'>{icon}</div>")

if st.button("∧", key="up", disabled=disabled, width="stretch", shortcut="ArrowUp"):
    make_move(0)
    st.rerun()

side_l, board_col, side_r = st.columns([1.3, 7.4, 1.3], vertical_alignment="center")
with side_l:
    if st.button("❮", key="left", disabled=disabled, width="stretch", shortcut="ArrowLeft"):
        make_move(2)
        st.rerun()
with board_col:
    st.markdown(f"<div class='maze'>{''.join(cells_html)}</div>", unsafe_allow_html=True)
with side_r:
    if st.button("❯", key="right", disabled=disabled, width="stretch", shortcut="ArrowRight"):
        make_move(3)
        st.rerun()

if st.button("∨", key="down", disabled=disabled, width="stretch", shortcut="ArrowDown"):
    make_move(1)
    st.rerun()

# ==========================================
# 6. RESULT BANNER
# ==========================================
if st.session_state.game_over:
    moves = st.session_state.moves_count
    ai_steps = st.session_state.ai_steps
    if st.session_state.autopilot_used:
        st.info(f"🤖 The Q-learning agent escaped in {moves} moves (best possible: {st.session_state.par}).")
    elif ai_steps != -1 and moves < ai_steps:
        st.balloons()
        st.success(f"🏆 You BEAT the AI! {moves} moves vs its {ai_steps} — it should have trained longer.")
    elif ai_steps != -1 and moves == ai_steps:
        st.balloons()
        st.success(f"🤝 Dead heat — you matched the AI's {ai_steps} moves!")
    else:
        st.warning(f"🏁 Escaped in {moves} moves. The AI did it in {ai_steps}. Tap Restart to try again!")

# ==========================================
# 7. ACTION BUTTONS
# ==========================================
st.write("")

act_l, act_r = st.columns(2)
with act_l:
    if st.button("↺ Restart", key="restart", width="stretch"):
        restart_run()
        st.rerun()
with act_r:
    if st.button("🤖 Watch AI", key="autopilot", width="stretch", disabled=st.session_state.ai_steps == -1):
        restart_run()
        run_autopilot()
        st.rerun()

if st.button("🎲 New maze (retrains AI)", key="newmaze", type="primary", width="stretch"):
    new_game()
    st.rerun()

# ==========================================
# 8. THE Q-SCORE FUNCTION
# ==========================================
st.markdown("---")
st.markdown("**The Q-score function** — after every step, the AI nudges its table entry for that cell and move:")
st.latex(r"Q(s,a)\ \leftarrow\ Q(s,a) + \alpha\,\bigl[\,r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)\,\bigr]")
st.caption(
    "s: current cell · a: move taken · r: reward (+100 goal, −1 per step) · "
    "s′: next cell · α: learning rate · γ: discount factor. "
    "The numbers revealed on your trail are each cell's best Q-score, maxₐ Q(s, a)."
)
