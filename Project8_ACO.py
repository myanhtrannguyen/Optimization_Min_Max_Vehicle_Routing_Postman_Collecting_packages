import sys
import random as rd
import time

# Input data
def Input():
    N, K = map(int, input().split())
    d = [[int(x) for x in sys.stdin.readline().split()] for _ in range(N + 1)]
    for _ in range(K - 1):
        d.append(d[0]) # create virtual depot
    return N, K, d

# Initialize pheromone
def InitPheromone(N, K):
    return [[1.0 for _ in range(N + K)] for _ in range(N + K)]

# Select the next node based on pheromone and heuristic
def FindNext(unvisited, position, pheromone, route_lengths, alpha, beta, k=0, A=100):
    probs = {}
    total = 0.0
    for i in unvisited:
        #pheromone component
        τ = pheromone[position][i] ** alpha
        # Heuristic information
        if i > N:
            #heuristic for virtual depot
            if route_lengths[k] == 0:
                η = 0
            elif route_lengths[k] < A:
                η = (1 / (A - route_lengths[k])) ** beta
            else:
                η = (1 / (A / route_lengths[k])) ** beta
        else:
            #heuristic for real pickup point
            η = (1 / (d[position][i] + 1e-6)) ** beta

        prob = τ * η
        probs[i] = prob
        total += prob
    # Roulette-wheel selection
    r = rd.uniform(0, total)
    s = 0.0
    for j, p in probs.items():
        s += p
        if s >= r:
            return j

# Route construction version 0: uses virtual depots to balance load across postmen
def FindRoutesv0(N, K, d, pheromone, alpha=1.0, beta=3.0, A=100.0):
    unvisited = list(range(1, N + K))
    routes = [[] for _ in range(K)]
    route_lengths = [0 for _ in range(K)]
    
    # Start each route at a unique virtual depot
    for a in range(K):
        routes[a].append(0 if a == 0 else a + N)

    position = 0
    k = 0

    while unvisited:
        chosen = FindNext(unvisited, position, pheromone, route_lengths, alpha, 2*beta, k, A)
        if chosen > N:
            k = chosen - N
        else:
            routes[k].append(chosen)
            route_lengths[k] += d[position][chosen]
        position = chosen
        unvisited.remove(chosen)

    # Reassign nodes from overloaded routes to empty routes
    while 0 in route_lengths:
        k = route_lengths.index(0)
        count = 0
        while True:
            count += 1
            k1 = max(range(K), key=lambda x: route_lengths[x])
            if len(routes[k1]) >= 3:
                a = routes[k1].pop()
                cur_k = routes[k][-1]
                routes[k].append(a)
                b, c = routes[k1][-2], routes[k1][-1]
                route_lengths[k] += d[cur_k][a]
                route_lengths[k1] -= d[b][c]
            else:
                break
            if route_lengths[k] >= max(route_lengths):
                break

    return routes, route_lengths

# Route construction version 1: greedy selection based on shortest route length
def FindRoutesv1(N, K, d, pheromone, alpha=1.0, beta=3.0):
    unvisited = list(range(1, N + 1))
    routes = [[] for _ in range(K)]
    route_lengths = [0 for _ in range(K)]

    for i in range(K):
        routes[i].append(0 if i == 0 else i + N)

    position = 0
    k = 0

    while unvisited:
        chosen = FindNext(unvisited, position, pheromone, route_lengths, alpha, beta)
        routes[k].append(chosen)
        route_lengths[k] += d[position][chosen]
        k = min(range(K), key=lambda x: route_lengths[x])
        position = routes[k][-1]
        unvisited.remove(chosen)

    return routes, route_lengths

# Update pheromone values based on best solutions found
def UpdatePheromone(N, pheromone, best_routes, best_lengths, evaporation, m):
    for i in range(N + K):
        for j in range(N + K):
            pheromone[i][j] = max(0.07, pheromone[i][j] * evaporation)

    for k in range(K):
        u = 0 if k == 0 else best_routes[k][0]
        for v in best_routes[k]:
            if v == 0:
                continue
            delta = m / (1 + best_lengths[k])
            pheromone[u][v] = min(20, pheromone[u][v] + delta)
            pheromone[v][u] = pheromone[u][v]
            u = v

# Main Ant Colony Optimization algorithm
def Solve(N, K, d, num_ants=20, iterations=100, alpha_base=1, beta_base=4, evap_base=0.8, f=2, time_limit=29):
    start_time = time.time()
    pheromone = InitPheromone(N, K)
    # Start with greedy construction
    best_routes, best_lengths = FindRoutesv1(N, K, d, pheromone)
    best_max_length = max(best_lengths)
    A = sum(best_lengths) / K
    best_A = A

    for i in range(iterations):
        alpha = alpha_base + 0.5 * i / iterations
        beta = beta_base - 1.5 * i / iterations
        evaporation = evap_base - 0.2 * i / iterations
        max_len_in_i = 1e10

        for j in range(num_ants):
            if j <= num_ants * (0.2 - 0.15 * i / iterations):
                routes, lengths = FindRoutesv1(N, K, d, pheromone, alpha, beta)
            else:
                routes, lengths = FindRoutesv0(N, K, d, pheromone, alpha, beta, A)

            max_len = max(lengths)
            average = sum(lengths) / K

            if max_len < best_max_length or (max_len == best_max_length and best_A >= average):
                best_max_length = max_len
                best_routes = routes
                best_lengths = lengths
                best_A = average

            if max_len < max_len_in_i:
                max_len_in_i = max_len
                best_routes_in_i = routes
                best_lengths_in_i = lengths
                A = average

            if time.time() - start_time > time_limit:
                return best_routes

        # Update pheromones based on good iteration or best of iteration
        ratio = best_max_length / max_len_in_i
        if ratio >= 0.9 and best_A > average:
            UpdatePheromone(N, pheromone, best_routes_in_i, best_lengths_in_i, evaporation, ratio ** 5 * f * best_max_length)
        elif ratio >= 1 or ratio < 0.95:
            UpdatePheromone(N, pheromone, best_routes, best_lengths, evaporation, f * best_max_length)

    return best_routes

def PrintSolution(K, best_routes):
    print(K)
    for route in best_routes:
        print(len(route))
        print("0", *route[1:])

N, K, d = Input()
# Hyperparameter tuning based on problem size
if N <= 50:
    best_routes = Solve(N, K, d, num_ants=100, iterations=180, alpha_base=1.2, beta_base=5.5, evap_base=0.75, f=1.5)
elif N <= 100:
    best_routes = Solve(N, K, d, num_ants=80, iterations=150, alpha_base=1.1, beta_base=5.0, evap_base=0.6, f=1.5)
elif N <= 200:
    best_routes = Solve(N, K, d, num_ants=80, iterations=100, alpha_base=1.0, beta_base=5.0, evap_base=0.55, f=2)
elif N <= 500:
    best_routes = Solve(N, K, d, num_ants=25, iterations=50, alpha_base=0.8, beta_base=5.5, evap_base=0.5, f=2.5)
elif N <= 800:
    best_routes = Solve(N, K, d, num_ants=20, iterations=30, alpha_base=0.8, beta_base=5.5, evap_base=0.5, f=3.0)
else:
    best_routes = Solve(N, K, d, num_ants=15, iterations=20, alpha_base=0.8, beta_base=5.5, evap_base=0.5, f=3.5)
PrintSolution(K, best_routes)