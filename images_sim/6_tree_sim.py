import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Affine2D
 
# === Environment setup ===
MAP_W, MAP_H = 70, 70
robot_pos = np.array([60.0, 60.0])
warehouse_pos = np.array([60.0, 60.0])
 
# Dynamic obstacles (animals)
NUM_MOVS = 3
movs = [np.array([random.uniform(20, 60), random.uniform(20, 60)]) for _ in range(NUM_MOVS)]
mov_vel = [np.random.uniform(-1, 1, size=2) * 0.3 for _ in range(NUM_MOVS)]  # fixed typo
 
# Palm trees (static obstacles)
static_obstacles = []
for i in range(10, 60, 22):
    static_obstacles.append(np.array([i, 20]))
    static_obstacles.append(np.array([i, 40]))
 
# Human starts at a random palm
human_index = np.random.randint(len(static_obstacles))
human_pos = static_obstacles[human_index].copy()
# print("human_pos:", human_pos)
 
# === Additional static obstacles: Rocks & Ponds ===
rocks = [np.array([22, 30]), np.array([60, 50])]
ponds = [np.array([40, 50]), np.array([45, 30])]
 
# Merge all static obstacles for potential field computation
all_static_obs = static_obstacles + rocks + ponds
 
# === Potential field parameters ===
k_att = 1.0
k_rep_static = 120.0
k_rep_dynamic = 250.0
obs_influence = 5.0
max_speed = 1.0
goal_tolerance = 1.0
 
# === Helpers ===
def replace_robot_image(new_img):
    """Replace the current robot AnnotationBbox with a new image."""
    global robot_ab
    try:
        robot_ab.remove()
    except Exception:
        pass
    new_box = OffsetImage(new_img, zoom=0.05)
    robot_ab = AnnotationBbox(new_box, robot_pos, frameon=False)
    ax.add_artist(robot_ab)
 
def attractive_force(pos, goal):
    diff = goal - pos
    dist = np.linalg.norm(diff)
    if dist < 1e-5:
        return np.zeros(2)
    return k_att * diff / dist
 
def repulsive_force(pos, obstacles, k_rep, influence):
    f = np.zeros(2)
    for obs in obstacles:
        diff = pos - obs
        dist = np.linalg.norm(diff)
        if dist < influence and dist > 1e-3:
            f += k_rep * (1.0 / dist - 1.0 / influence) * (1.0 / dist**3) * diff
    return f
 
def update_movs(dt=0.1):
    global movs, mov_vel
    for i in range(NUM_MOVS):
        movs[i] += mov_vel[i] * dt * 10.0
        for j in range(2):
            if movs[i][j] < 10 or movs[i][j] > 60:
                mov_vel[i][j] *= -1
 
def move_human_to_new_palm():
    global human_index, human_pos
    new_index = human_index
    while new_index == human_index:
        new_index = np.random.randint(len(static_obstacles))
    human_index = new_index
    human_pos = static_obstacles[human_index].copy()
    print(f"👤 Human moved to new palm at {human_pos}")
 
# === Mission control ===
mission_stage = "idle"
 
def send_to_human():
    global mission_stage
    print("🧭 Command: Go to HUMAN.")
    mission_stage = "to_human"
 
def send_to_warehouse():
    global mission_stage
    print("🧭 Command: Go to WAREHOUSE.")
    mission_stage = "to_warehouse"
 
# === Visualization ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor("#f5deb3")
ax.set_xlim(0, MAP_W)
ax.set_ylim(0, MAP_H)
ax.set_aspect("equal")
ax.set_title("Mazr3aPorter: Potential Field Navigation")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
 
# 🦾 Robot images
robot_img_empty = mpimg.imread("husky.png")
robot_img_full = mpimg.imread("husky_dates.png")
robot_imagebox = OffsetImage(robot_img_empty, zoom=0.05)
robot_ab = AnnotationBbox(robot_imagebox, robot_pos, frameon=False)
ax.add_artist(robot_ab)
 
# 🏭 Warehouse image (rotated)
warehouse_img = mpimg.imread("warehouse.png")
warehouse_imagebox = OffsetImage(warehouse_img, zoom=0.15)
warehouse_ab = AnnotationBbox(warehouse_imagebox, warehouse_pos, frameon=False)
ax.add_artist(warehouse_ab)
 
# 🐕 Dynamic animals
image_folder = "dynamic_objects"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
movs_abs = []
for mov in movs:
    random_image_path = os.path.join(image_folder, random.choice(image_files))
    movs_img = mpimg.imread(random_image_path)
    movs_imagebox = OffsetImage(movs_img, zoom=0.05)
    movs_ab = AnnotationBbox(movs_imagebox, mov, frameon=False)
    ax.add_artist(movs_ab)
    movs_abs.append(movs_ab)
 
# 🪨 Rocks
rock_img = mpimg.imread("rock.png")
rock_imagebox = OffsetImage(rock_img, zoom=0.05)
for r in rocks:
    rock_ab = AnnotationBbox(rock_imagebox, r, frameon=False)
    ax.add_artist(rock_ab)
 
# 💧 Ponds / Holes
pond_img = mpimg.imread("hole.png")
pond_imagebox = OffsetImage(pond_img, zoom=0.05)
for p in ponds:
    pond_ab = AnnotationBbox(pond_imagebox, p, frameon=False)
    ax.add_artist(pond_ab)
 
# 🌴 Palm trees
palm_img = mpimg.imread("date_palm.png")
palm_imagebox = OffsetImage(palm_img, zoom=0.15)
palm_abs = []
for p in static_obstacles:
    palm_ab = AnnotationBbox(palm_imagebox, p, frameon=False)
    ax.add_artist(palm_ab)
    palm_abs.append(palm_ab)
 
# 👨 Human
# human_img = mpimg.imread("farmer.png")
# human_imagebox = OffsetImage(human_img, zoom=0.05)
# human_ab = AnnotationBbox(human_imagebox, human_pos, frameon=False)
# ax.add_artist(human_ab)
(human_dot,) = ax.plot([human_pos[0]], [human_pos[1]], "rs", markersize=8, label="Farmer", zorder=10)
 
 
# --- Path trace ---
(path_line,) = ax.plot([], [], "r--", linewidth=1, zorder=10)
path_x, path_y = [], []
 
ax.legend(loc="upper left")
 
# === Step function ===
def step(frame):
    global robot_pos, mission_stage, human_pos, robot_ab
 
    update_movs()
    # human_ab.xy = human_pos
    human_dot.set_data([human_pos[0]], [human_pos[1]])
 
    if mission_stage == "to_human":
        goal = human_pos
        replace_robot_image(robot_img_empty)
    elif mission_stage == "to_warehouse":
        goal = warehouse_pos
        replace_robot_image(robot_img_full)
    else:
        for i, mov_ab in enumerate(movs_abs):
            mov_ab.xy = movs[i]
        return [robot_ab, *movs_abs, path_line, human_dot]
 
    # --- Potential field ---
    F_att = attractive_force(robot_pos, goal)
    F_rep_static = repulsive_force(robot_pos, all_static_obs, k_rep_static, obs_influence)
    F_rep_dynamic = repulsive_force(robot_pos, movs, k_rep_dynamic, obs_influence)
    F_total = F_att + F_rep_static + F_rep_dynamic
 
    # --- Motion update ---
    norm = np.linalg.norm(F_total)
    if norm > max_speed:
        F_total = F_total / norm * max_speed
    robot_pos += F_total * 0.2
 
    path_x.append(robot_pos[0])
    path_y.append(robot_pos[1])
    robot_ab.xy = robot_pos
 
    for i, mov_ab in enumerate(movs_abs):
        mov_ab.xy = movs[i]
    path_line.set_data(path_x, path_y)
 
    # --- Goal check ---
    if np.linalg.norm(robot_pos - goal) < goal_tolerance:
        if mission_stage == "to_human":
            print("✅ Arrived at HUMAN (under palm). Waiting for load...")
            mission_stage = "idle"
        elif mission_stage == "to_warehouse":
            print("✅ Arrived at WAREHOUSE. Dropping dates...")
            mission_stage = "idle"
            move_human_to_new_palm()
 
            replace_robot_image(robot_img_empty)
 
    return [robot_ab, *movs_abs, path_line, human_dot]
 
ani = animation.FuncAnimation(fig, step, frames=1000, interval=100, blit=False, repeat=True)
 
def on_key(event):
    if event.key == "h":
        send_to_human()
    elif event.key == "w":
        send_to_warehouse()
 
fig.canvas.mpl_connect("key_press_event", on_key)
 
print("Press 'h' to send the robot to the HUMAN.")
print("Press 'w' to send the robot to the WAREHOUSE.")
plt.show()