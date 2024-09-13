
# Sorting Algorithms

# 1. Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            yield arr  # Yielding for visualization purposes

# 2. Merge Sort
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
        yield arr

# Searching Algorithms

# 3. Binary Search (works only on sorted arrays)
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1  # Target not found

# 4. Linear Search
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1  # Target not found

# Pathfinding Algorithms (for graphs)

# 5. Dijkstra's Algorithm (for shortest paths in a weighted graph)
import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while len(pq) > 0:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# 6. A* Search Algorithm
def a_star(graph, start, goal, h):
    open_set = set([start])
    came_from = {}

    g_score = {node: float('infinity') for node in graph}
    g_score[start] = 0

    f_score = {node: float('infinity') for node in graph}
    f_score[start] = h(start)

    while open_set:
        current = min(open_set, key=lambda x: f_score[x])

        if current == goal:
            return reconstruct_path(came_from, current)

        open_set.remove(current)

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + graph[current][neighbor]

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + h(neighbor)

                if neighbor not in open_set:
                    open_set.add(neighbor)

    return False

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Example heuristic function for A*
def heuristic(node, goal):
    # This can be a simple Euclidean distance or Manhattan distance based on the use case
    return abs(node - goal)

# Optimization Algorithms

# 7. Dynamic Programming - Knapsack Problem
def knapsack(weights, values, W):
    n = len(values)
    dp = [[0 for x in range(W+1)] for x in range(n+1)]

    for i in range(n+1):
        for w in range(W+1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][W]

# 8. Greedy Algorithm - Coin Change Problem
def coin_change(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        while amount >= coin:
            amount -= coin
            count += 1
    return count if amount == 0 else -1  # -1 means no solution
