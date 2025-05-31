import random

def Input(): # Input data
    N, K = map(int, input().split())
    d = [list(map(int, input().split())) for _ in range(N + 1)]
    return N, K, d

def InitialSol(N, K): # Generate initial Solution
    nodes = list(range(1, N + 1))
    random.shuffle(nodes)
    routes = [[] for _ in range(K)]
    for index, node in enumerate(nodes):
        routes[index % K].append(node)
    return [[0] + route for route in routes]

def Evaluate(routes, d): # Return max distance between all postmen
    cost = []
    for route in routes:
        total_dis = 0
        for i in range(1, len(route)):
            total_dis += d[route[i - 1]][route[i]]
        cost.append(total_dis)
    return max(cost)

def CostChange(route_from, route_to, i_from, i_to, d): # Cost diff when move a node from a postman to another
    if len(route_from) <= 1:
        return 1e9  # Can't remove if only depot + one point

    # node to move
    b = route_from[i_from]
    
    # Removal cost from route_from
    a = route_from[i_from - 1]
    if i_from + 1 < len(route_from):
        c = route_from[i_from + 1]
        remove_cost = d[a][b] + d[b][c] - d[a][c]
    else:
        remove_cost = d[a][b]

    # Insert cost to route_to
    x = route_to[i_to - 1]
    if i_to < len(route_to):
        y = route_to[i_to]
        insert_cost = d[x][b] + d[b][y] - d[x][y]
    else:
        insert_cost = d[x][b]

    return insert_cost - remove_cost

def NeighborsGen(routes, d, MAX_NEIGHBORS = 100): # Generate 100 neighbors
    neighbors = []
    K = len(routes)
    count = 0

    while count < MAX_NEIGHBORS:
        i = random.randint(0, K - 1)
        j = random.randint(0, K - 1)

        if i == j or len(routes[i]) <= 1:
            continue

        i_from = random.randint(1, len(routes[i]) - 1)
        i_to = random.randint(1, len(routes[j]))

        delta = CostChange(routes[i], routes[j], i_from, i_to, d)
        neighbors.append([delta, [i, j, i_from, i_to]])
        count += 1

    neighbors.sort()
    return neighbors
 
def Moving(routes, move): # Move node
    i, j, i_from, i_to = move
    new_routes = [r[:] for r in routes]  # deep copy
    node = new_routes[i][i_from]
    del new_routes[i][i_from]
    new_routes[j].insert(i_to, node)
    return new_routes

def TabuSearch(N, K, d, MAX_LOOP=300, TABU=10):
    current = InitialSol(N, K)
    best = [r[:] for r in current]
    best_cost = Evaluate(best, d)

    tabu_list = []
    loop = 0

    while loop < MAX_LOOP:
        neighbors = NeighborsGen(current, d, MAX_NEIGHBORS=100)

        for neighbor in neighbors:
            move = neighbor[1]
            move_key = move[:3]

            found_in_tabu = False
            for t in tabu_list:
                if move_key == t[:3]:
                    found_in_tabu = True
                    break

            candidate = Moving(current, move)
            candidate_cost = Evaluate(candidate, d)

            if found_in_tabu and candidate_cost >= best_cost:
                continue

            current = candidate
            if candidate_cost < best_cost:
                best = [r[:] for r in candidate]
                best_cost = candidate_cost

            tabu_list.append(move + [loop + TABU])
            break

        # Remove expired tabus
        tabu_list = [t for t in tabu_list if t[4] > loop]
        loop += 1

    return best

def main():
    N, K, d = Input()
    Sol = TabuSearch(N, K, d, MAX_LOOP=300, TABU=10)

    print(K)
    for route in Sol:
        print(len(route))
        print(" ".join(str(x) for x in route))

if __name__ == "__main__":
    main()
