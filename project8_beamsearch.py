import sys
import numpy as np

def read_input():
    lines = sys.stdin.read().strip().split('\n')
    N, K = map(int, lines[0].split())
    dist_matrix = [list(map(int, line.split())) for line in lines[1:]]
    return N, K, np.array(dist_matrix)

# Divide N points into K clusters using a greedy distance-based approach
def greedy_distance_cluster(N, K, dist_matrix):
    unassigned = set(range(1, N+1)) 
    clusters = []

    cluster_size = N // K
    remainder = N % K

    while unassigned:
        start = unassigned.pop()
        cluster = [start]

        distances = [(dist_matrix[start][j], j) for j in unassigned]
        distances.sort()

        cur_cluster_size = cluster_size + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1

        for _, point in distances:
            if len(cluster) >= cur_cluster_size:
                break
            cluster.append(point)
            unassigned.remove(point)

        clusters.append(cluster)

    return clusters

# Evaluate function to return the distance cost of a route
def route_cost(route, dist_matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += dist_matrix[route[i]][route[i+1]]
    return cost

# Beam Search through the clusters to find the best route
def beam_search_route(cluster, dist_matrix, beam_width=5):
    beam = [([0], set())]  # start at depot

    while True:
        new_beam = []
        for path, visited in beam:
            if len(visited) == len(cluster):
                new_beam.append((path + [0], visited))  
                continue
            for p in cluster:
                if p not in visited:
                    new_path = path + [p]
                    new_visited = visited | {p}
                    new_beam.append((new_path, new_visited))

        new_beam = sorted(new_beam, key=lambda x: route_cost(x[0], dist_matrix))[:beam_width]
        if all(len(v) == len(cluster) for _, v in new_beam):
            break
        beam = new_beam

    best_path = min(new_beam, key=lambda x: route_cost(x[0], dist_matrix))[0]
    return best_path


def main():
    N, K, dist_matrix = read_input()

    clusters = greedy_distance_cluster(N, K, dist_matrix)
    routes = [beam_search_route(cluster, dist_matrix, beam_width=5) for cluster in clusters]

    print(K)
    for route in routes:
        print(len(route))
        print(" ".join(map(str, route)))

if __name__ == "__main__":
    main()
