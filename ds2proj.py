import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

st.set_page_config(layout="centered")
st.title("Maze Generator & Solver")

# Controls 
N = st.slider("Maze Size", 5, 51, 21, step=2)
algo_choice = st.selectbox("Maze Generation Algorithm", ["Prim", "Kruskal"])
solver_choice = st.selectbox("Pathfinding Algorithm", ["DFS", "BFS"])

# DFS
def dfs(maze, start, goal):
    stack = [start]
    visited = set()
    parent = {}
    while stack:
        current = stack.pop()
        if current == goal:
            break
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
                if (nx, ny) not in visited:
                    stack.append((nx, ny))
                    parent[(nx, ny)] = current
    # Reconstruct path
    path = []
    node = goal
    while node in parent:
        path.append(node)
        node = parent[node]
    if node == start:
        path.append(start)
        return path[::-1]
    return []

# BFS
def bfs(maze, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = current
                    queue.append((nx, ny))
    # Reconstruct path
    path = []
    node = goal
    while node in parent:
        path.append(node)
        node = parent[node]
    if node == start:
        path.append(start)
        return path[::-1]
    return []

# Prim's Algorithm
def generate_maze_prim(n):
    maze = np.ones((n, n), dtype=int)
    start = (1, 1)
    maze[start] = 0
    walls = []

    def add_walls(x, y):
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx < n-1 and 1 <= ny < n-1 and maze[nx][ny] == 1:
                walls.append((nx, ny, x, y))

    add_walls(*start)

    while walls:
        wx, wy, px, py = walls.pop(random.randint(0, len(walls)-1))
        if maze[wx][wy] == 1:
            mx, my = (wx + px)//2, (wy + py)//2
            maze[wx][wy] = 0
            maze[mx][my] = 0
            add_walls(wx, wy)

    return maze

# Kruskal's Algorithm
def generate_maze_kruskal(n):
    maze = np.ones((n, n), dtype=int)

    # Each cell is its own set
    parent = {}
    def find(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell
    def union(a, b):
        parent[find(a)] = find(b)

    # Initialize sets and walls
    cells = [(i, j) for i in range(1, n, 2) for j in range(1, n, 2)]
    for cell in cells:
        parent[cell] = cell
        maze[cell] = 0

    walls = []
    for x, y in cells:
        for dx, dy in [(2, 0), (0, 2)]:
            nx, ny = x + dx, y + dy
            if nx < n and ny < n:
                walls.append(((x, y), (nx, ny)))

    random.shuffle(walls)

    for a, b in walls:
        if find(a) != find(b):
            mx, my = (a[0] + b[0])//2, (a[1] + b[1])//2
            maze[mx][my] = 0
            union(a, b)

    return maze

# generate maze 
if algo_choice == "Prim":
    maze = generate_maze_prim(N)
else:
    maze = generate_maze_kruskal(N)

# solve and Visualize
start, goal = (1, 1), (N-2, N-2)

if solver_choice == "DFS":
    path = dfs(maze, start, goal)
else:
    path = bfs(maze, start, goal)

fig, ax = plt.subplots()
ax.imshow(maze, cmap='binary')
if path:
    y, x = zip(*path)
    ax.plot(x, y, color='red', linewidth=2)
ax.scatter(start[1], start[0], c='green', label='Start')
ax.scatter(goal[1], goal[0], c='blue', label='Goal')
ax.set_xticks([]); ax.set_yticks([])
st.pyplot(fig)
