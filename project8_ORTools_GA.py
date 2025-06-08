import sys
import time
import random
import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import heapq

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

class BalancedVRPSolver:
    def __init__(self, N: int, K: int, distance_matrix: List[List[int]]):
        self.N = N
        self.K = K
        self.distance_matrix = distance_matrix
        self.start_time = time.time()
        self.time_limit = 18.0  # Reduced to 18 seconds for safety
        
        # Calculate balanced load distribution
        self.target_load = N // K
        self.remainder = N % K
        self.route_sizes = [self.target_load + (1 if i < self.remainder else 0) for i in range(K)]
        
        # Precompute nearest neighbors for faster access
        self.nearest_neighbors = {}
        self.precompute_nearest_neighbors()
        
    def precompute_nearest_neighbors(self):
        """Precompute k nearest neighbors for each node"""
        k = min(20, self.N)  # Limit to 20 nearest neighbors
        
        for i in range(self.N + 1):
            distances = []
            for j in range(self.N + 1):
                if i != j:
                    distances.append((self.distance_matrix[i][j], j))
            
            distances.sort()
            self.nearest_neighbors[i] = [node for _, node in distances[:k]]
    
    def get_remaining_time(self):
        return max(0, self.time_limit - (time.time() - self.start_time))
    
    def solve(self):
        """Main solving method that selects strategy based on problem size"""
        if self.N < 50:
            return self.solve_small()
        elif self.N <= 150:
            return self.solve_medium()
        elif self.N <= 400:
            return self.solve_large()
        elif self.N <= 900:
            return self.solve_very_large()
        else:
            return self.solve_ultra_large()  # New tier for 900 < N <= 1000
    
    def solve_small(self):
        """Tier 1: Small instances (N < 50) - Use OR-Tools if available"""
        if ORTOOLS_AVAILABLE and self.get_remaining_time() > 10:
            solution = self.solve_with_ortools()
            if solution:
                return solution
        
        # Fallback to balanced greedy + local search
        return self.solve_with_balanced_greedy()
    
    def solve_medium(self):
        """Tier 2: Medium instances (50 ≤ N ≤ 150)"""
        if ORTOOLS_AVAILABLE and self.get_remaining_time() > 8:
            solution = self.solve_with_ortools()
            if solution:
                return solution
        
        # Balanced construction + local search
        solution = self.solve_with_balanced_greedy()
        if self.get_remaining_time() > 3:
            solution = self.local_search_improvement(solution)
        return solution
    
    def solve_large(self):
        """Tier 3: Large instances (150 < N ≤ 400) - Genetic Algorithm"""
        return self.solve_with_genetic_algorithm()
    
    def solve_very_large(self):
        """Tier 4: Very large instances (400 < N ≤ 900) - Multi-phase approach"""
        return self.solve_with_clustering()
    
    def solve_ultra_large(self):
        """Tier 5: Ultra large instances (900 < N ≤ 1000) - Speed optimized"""
        return self.solve_with_ultra_fast_heuristic()
    
    def solve_with_ultra_fast_heuristic(self):
        """Ultra-fast heuristic for instances with N > 900"""
        # Phase 1: Quick geographic partitioning (2 seconds)
        routes = self.quick_geographic_partition()
        
        # Phase 2: Fast nearest neighbor improvement (3 seconds)
        if self.get_remaining_time() > 10:
            routes = self.fast_nearest_neighbor_improvement(routes)
        
        # Phase 3: Limited local search (remaining time)
        if self.get_remaining_time() > 5:
            routes = self.limited_local_search(routes)
        
        return routes, self.calculate_max_distance(routes)[1]
    
    def quick_geographic_partition(self):
        """Ultra-fast geographic partitioning using polar coordinates"""
        # Convert to polar coordinates for fast clustering
        customers = []
        for i in range(1, self.N + 1):
            # Use distance to depot as radius
            radius = self.distance_matrix[0][i]
            # Use index-based angle for distribution
            angle = (i * 2 * math.pi) / self.N
            customers.append((i, radius, angle))
        
        # Sort by angle for geographic distribution
        customers.sort(key=lambda x: x[2])
        
        # Assign to routes in round-robin fashion with balance
        routes = [[] for _ in range(self.K)]
        route_loads = [0] * self.K
        
        customer_idx = 0
        while customer_idx < len(customers):
            for k in range(self.K):
                if customer_idx >= len(customers):
                    break
                
                # Check if this route can take more customers
                if route_loads[k] < self.route_sizes[k]:
                    routes[k].append(customers[customer_idx][0])
                    route_loads[k] += 1
                    customer_idx += 1
        
        # Add depot to each route
        for k in range(self.K):
            routes[k] = [0] + routes[k]
        
        return routes
    
    def fast_nearest_neighbor_improvement(self, routes):
        """Fast nearest neighbor based route improvement"""
        improved_routes = []
        
        for route in routes:
            if len(route) <= 2:  # Only depot or depot + 1 customer
                improved_routes.append(route)
                continue
            
            # Quick nearest neighbor reordering
            new_route = [0]  # Start with depot
            unvisited = set(route[1:])  # All customers except depot
            current = 0
            
            while unvisited and self.get_remaining_time() > 0.1:
                # Find nearest unvisited customer
                nearest = min(unvisited, key=lambda x: self.distance_matrix[current][x])
                new_route.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            
            # Add any remaining customers (shouldn't happen)
            new_route.extend(list(unvisited))
            
            improved_routes.append(new_route)
        
        return improved_routes
    
    def limited_local_search(self, routes):
        """Very limited local search for ultra-large instances"""
        max_iterations = 5  # Very limited iterations
        iteration = 0
        
        while iteration < max_iterations and self.get_remaining_time() > 2:
            improved = False
            
            # Only try intra-route 2-opt on longest routes
            route_distances = [(i, self.calculate_route_distance(routes[i])) 
                             for i in range(len(routes))]
            route_distances.sort(key=lambda x: x[1], reverse=True)
            
            # Improve only top 3 longest routes
            for i, _ in route_distances[:3]:
                if len(routes[i]) > 3 and self.get_remaining_time() > 1:
                    old_distance = self.calculate_route_distance(routes[i])
                    new_route = self.fast_2opt(routes[i])
                    new_distance = self.calculate_route_distance(new_route)
                    
                    if new_distance < old_distance:
                        routes[i] = new_route
                        improved = True
            
            if not improved:
                break
            
            iteration += 1
        
        return routes
    
    def fast_2opt(self, route):
        """Very fast 2-opt with limited moves"""
        if len(route) <= 3:
            return route
        
        best_route = route[:]
        best_distance = self.calculate_route_distance(route)
        
        # Limit the number of 2-opt moves for speed
        max_moves = min(10, len(route) * 2)
        moves_tried = 0
        
        for i in range(1, len(route) - 2):
            if moves_tried >= max_moves or self.get_remaining_time() < 0.1:
                break
                
            for j in range(i + 2, len(route)):
                if moves_tried >= max_moves:
                    break
                
                # Create new route by reversing segment
                new_route = route[:]
                new_route[i:j+1] = reversed(new_route[i:j+1])
                
                new_distance = self.calculate_route_distance(new_route)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    break  # Take first improvement for speed
                
                moves_tried += 1
        
        return best_route
    
    def solve_with_ortools(self):
        """Use OR-Tools VRP solver with balanced constraints"""
        if not ORTOOLS_AVAILABLE:
            return None
            
        try:
            # Create routing model
            manager = pywrapcp.RoutingIndexManager(self.N + 1, self.K, 0)
            routing = pywrapcp.RoutingModel(manager)
            
            # Distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return self.distance_matrix[from_node][to_node]
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # Add capacity constraints for balanced loads
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return 1 if from_node != 0 else 0
            
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                [size for size in self.route_sizes],  # vehicle maximum capacities
                True,  # start cumul to zero
                'Capacity')
            
            # Minimize the maximum route distance
            routing.SetFixedCostOfAllVehicles(0)
            
            # Search parameters
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.seconds = int(self.get_remaining_time())
            
            # Solve
            solution = routing.SolveWithParameters(search_parameters)
            
            if solution:
                return self.extract_ortools_solution(manager, routing, solution)
            
        except Exception as e:
            print(f"OR-Tools error: {e}", file=sys.stderr)
        
        return None
    
    def extract_ortools_solution(self, manager, routing, solution):
        """Extract solution from OR-Tools"""
        routes = []
        max_distance = 0
        
        for vehicle_id in range(self.K):
            route = [0]  # Start from depot
            route_distance = 0
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node != 0:
                    route.append(node)
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            routes.append(route)
            max_distance = max(max_distance, route_distance)
        
        return routes, max_distance
    
    def solve_with_balanced_greedy(self):
        """Balanced greedy construction using modified savings algorithm"""
        # Calculate savings for all pairs (limited for speed)
        savings = []
        sample_size = min(self.N * 5, 1000)  # Limit savings calculations
        
        customers = list(range(1, self.N + 1))
        if len(customers) > 50:
            # Sample customers for large instances
            sample_customers = random.sample(customers, min(50, len(customers)))
        else:
            sample_customers = customers
        
        for i in sample_customers:
            for j in self.nearest_neighbors.get(i, [])[:10]:  # Use nearest neighbors
                if j > i and j != 0:  # Avoid duplicates and depot
                    save = self.distance_matrix[0][i] + self.distance_matrix[0][j] - self.distance_matrix[i][j]
                    savings.append((save, i, j))
        
        savings.sort(reverse=True)  # Sort by savings in descending order
        
        # Initialize routes - each customer starts in its own route
        routes = [[0, i] for i in range(1, self.N + 1)]
        route_loads = [1] * self.N
        
        # Merge routes based on savings while respecting balance
        for save, i, j in savings:
            if self.get_remaining_time() < 2:
                break
                
            route_i = route_j = -1
            
            # Find which routes contain i and j
            for idx, route in enumerate(routes):
                if i in route:
                    route_i = idx
                if j in route:
                    route_j = idx
            
            # If in different routes and can merge without exceeding capacity
            if (route_i != route_j and route_i != -1 and route_j != -1 and
                route_loads[route_i] + route_loads[route_j] <= max(self.route_sizes)):
                
                # Simple merge (append route_j to route_i)
                routes[route_i].extend(routes[route_j][1:])  # Skip depot in second route
                route_loads[route_i] += route_loads[route_j]
                routes.pop(route_j)
                route_loads.pop(route_j)
        
        # Balance the routes
        routes = self.balance_routes_quickly(routes)
        
        return routes, self.calculate_max_distance(routes)[1]
    
    def balance_routes_quickly(self, routes):
        """Quick route balancing"""
        # If we have too many routes, merge smallest ones
        while len(routes) > self.K and self.get_remaining_time() > 1:
            # Find two smallest routes
            route_sizes = [(i, len(routes[i]) - 1) for i in range(len(routes))]
            route_sizes.sort(key=lambda x: x[1])
            
            if len(route_sizes) >= 2:
                i, j = route_sizes[0][0], route_sizes[1][0]
                if route_sizes[0][1] + route_sizes[1][1] <= max(self.route_sizes):
                    routes[i].extend(routes[j][1:])  # Merge
                    routes.pop(j)
        
        # If we have too few routes, split largest ones
        while len(routes) < self.K and self.get_remaining_time() > 1:
            # Find largest route
            largest_idx = max(range(len(routes)), key=lambda i: len(routes[i]))
            
            if len(routes[largest_idx]) > 3:  # Can split
                route = routes[largest_idx]
                mid = len(route) // 2
                routes[largest_idx] = route[:mid]
                routes.append([0] + route[mid:])
            else:
                # Add empty route
                routes.append([0])
        
        return routes
    
    def solve_with_genetic_algorithm(self):
        """Genetic algorithm for large instances"""
        population_size = 30  # Reduced for speed
        generations = 50      # Reduced for speed
        mutation_rate = 0.15
        elite_size = 3
        
        # Initialize population with balanced solutions
        population = []
        for _ in range(population_size):
            solution = self.generate_random_balanced_solution()
            population.append(solution)
        
        best_solution = min(population, key=lambda x: self.calculate_max_distance(x)[1])
        best_fitness = self.calculate_max_distance(best_solution)[1]
        
        generation = 0
        while generation < generations and self.get_remaining_time() > 3:
            # Evaluate fitness
            fitness_scores = []
            for solution in population:
                _, max_dist = self.calculate_max_distance(solution)
                fitness_scores.append(1.0 / (1.0 + max_dist))
            
            # Selection and reproduction
            new_population = []
            
            # Keep elite
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
            for idx in elite_indices:
                new_population.append(population[idx][:])
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child = self.balanced_crossover(parent1, parent2)
                
                if random.random() < mutation_rate:
                    child = self.balanced_mutation(child)
                
                new_population.append(child)
            
            population = new_population
            
            # Update best solution
            for solution in population:
                _, max_dist = self.calculate_max_distance(solution)
                if max_dist < best_fitness:
                    best_solution = solution[:]
                    best_fitness = max_dist
            
            generation += 1
        
        return best_solution, best_fitness
    
    def solve_with_clustering(self):
        """Multi-phase clustering approach for very large instances"""
        # Phase 1: Quick geographic clustering
        clusters = self.quick_balanced_clustering()
        
        # Phase 2: Convert to routes
        routes = self.assign_clusters_to_vehicles(clusters)
        
        # Phase 3: Quick route optimization
        for i in range(len(routes)):
            if len(routes[i]) > 3 and self.get_remaining_time() > 2:
                routes[i] = self.fast_2opt(routes[i])
        
        return routes, self.calculate_max_distance(routes)[1]
    
    def quick_balanced_clustering(self):
        """Quick balanced clustering using distance-based approach"""
        # Use depot distances for quick clustering
        customers_by_distance = []
        for i in range(1, self.N + 1):
            customers_by_distance.append((self.distance_matrix[0][i], i))
        
        customers_by_distance.sort()
        
        # Distribute customers across clusters in balanced manner
        clusters = [[] for _ in range(self.K)]
        
        # Round-robin assignment with balance constraints
        for idx, (_, customer) in enumerate(customers_by_distance):
            cluster_idx = idx % self.K
            
            # Check if cluster has room
            if len(clusters[cluster_idx]) < self.route_sizes[cluster_idx]:
                clusters[cluster_idx].append(customer)
            else:
                # Find cluster with minimum load
                min_load = min(len(cluster) for cluster in clusters)
                for k in range(self.K):
                    if len(clusters[k]) == min_load and len(clusters[k]) < self.route_sizes[k]:
                        clusters[k].append(customer)
                        break
        
        return clusters
    
    def assign_clusters_to_vehicles(self, clusters):
        """Convert clusters to routes"""
        routes = []
        for cluster in clusters:
            if cluster:
                route = [0] + cluster  # Add depot at start
                routes.append(route)
            else:
                routes.append([0])  # Empty route
        
        return routes
    
    def generate_random_balanced_solution(self):
        """Generate a random solution with balanced loads"""
        customers = list(range(1, self.N + 1))
        random.shuffle(customers)
        
        routes = []
        start_idx = 0
        
        for k in range(self.K):
            route_size = self.route_sizes[k]
            end_idx = start_idx + route_size
            
            route = [0] + customers[start_idx:end_idx]
            routes.append(route)
            start_idx = end_idx
        
        return routes
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection for genetic algorithm"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[winner_idx][:]
    
    def balanced_crossover(self, parent1, parent2):
        """Order crossover that preserves balance"""
        # Preserve route structure from parent1
        child = [[] for _ in range(len(parent1))]
        
        # Copy route sizes from parent1
        for i in range(len(parent1)):
            child[i] = [0]  # Start with depot
        
        # Create order from parent2
        order = []
        for route in parent2:
            order.extend(route[1:])  # Skip depot
        
        # Fill child routes according to parent2 order
        order_idx = 0
        for i in range(len(child)):
            target_size = len(parent1[i]) - 1  # -1 for depot
            
            while len(child[i]) - 1 < target_size and order_idx < len(order):
                if order[order_idx] not in [customer for route in child for customer in route]:
                    child[i].append(order[order_idx])
                order_idx += 1
        
        return child
    
    def balanced_mutation(self, solution):
        """Mutation that preserves balance"""
        solution = [route[:] for route in solution]  # Deep copy
        
        if random.random() < 0.7:
            # Intra-route mutation (2-opt) - faster
            route_idx = random.randint(0, len(solution) - 1)
            if len(solution[route_idx]) > 3:
                solution[route_idx] = self.fast_2opt(solution[route_idx])
        else:
            # Inter-route swap
            route1_idx = random.randint(0, len(solution) - 1)
            route2_idx = random.randint(0, len(solution) - 1)
            
            if (route1_idx != route2_idx and 
                len(solution[route1_idx]) > 1 and len(solution[route2_idx]) > 1):
                
                # Random swap
                pos1 = random.randint(1, len(solution[route1_idx]) - 1)
                pos2 = random.randint(1, len(solution[route2_idx]) - 1)
                
                solution[route1_idx][pos1], solution[route2_idx][pos2] = \
                    solution[route2_idx][pos2], solution[route1_idx][pos1]
        
        return solution
    
    def local_search_improvement(self, routes):
        """Local search improvement with balanced moves"""
        improved = True
        iteration = 0
        max_iterations = 20  # Reduced for speed
        
        while improved and self.get_remaining_time() > 2 and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try 2-opt within each route (limited)
            for i in range(len(routes)):
                if len(routes[i]) > 3 and self.get_remaining_time() > 1:
                    new_route = self.fast_2opt(routes[i])
                    if self.calculate_route_distance(new_route) < self.calculate_route_distance(routes[i]):
                        routes[i] = new_route
                        improved = True
            
            # Try limited swaps between routes
            if self.get_remaining_time() > 3:
                for i in range(min(len(routes), 5)):  # Limit route pairs
                    for j in range(i + 1, min(len(routes), i + 6)):
                        if len(routes[i]) > 1 and len(routes[j]) > 1:
                            # Try one swap only
                            pos1 = random.randint(1, len(routes[i]) - 1)
                            pos2 = random.randint(1, len(routes[j]) - 1)
                            
                            old_dist_i = self.calculate_route_distance(routes[i])
                            old_dist_j = self.calculate_route_distance(routes[j])
                            old_max = max(old_dist_i, old_dist_j)
                            
                            # Swap
                            routes[i][pos1], routes[j][pos2] = routes[j][pos2], routes[i][pos1]
                            
                            new_dist_i = self.calculate_route_distance(routes[i])
                            new_dist_j = self.calculate_route_distance(routes[j])
                            new_max = max(new_dist_i, new_dist_j)
                            
                            if new_max < old_max:
                                improved = True
                                break
                            else:
                                # Swap back
                                routes[i][pos1], routes[j][pos2] = routes[j][pos2], routes[i][pos1]
                    
                    if improved:
                        break
        
        return routes
    
    def calculate_route_distance(self, route):
        """Calculate total distance for a single route"""
        if len(route) <= 1:
            return 0
        
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i]][route[i + 1]]
        
        # Return to depot
        distance += self.distance_matrix[route[-1]][0]
        
        return distance
    
    def calculate_max_distance(self, routes):
        """Calculate maximum distance among all routes"""
        max_distance = 0
        for route in routes:
            distance = self.calculate_route_distance(route)
            max_distance = max(max_distance, distance)
        
        return routes, max_distance
    
    def validate_solution(self, routes):
        """Validate solution correctness and balance"""
        # Check all customers are visited exactly once
        visited = set()
        for route in routes:
            for customer in route[1:]:  # Skip depot
                if customer in visited:
                    return False, "Customer visited multiple times"
                visited.add(customer)
        
        if len(visited) != self.N:
            return False, f"Missing customers: expected {self.N}, got {len(visited)}"
        
        # Check balance
        sizes = [len(route) - 1 for route in routes]  # -1 for depot
        if max(sizes) - min(sizes) > 1:
            return False, f"Unbalanced routes: sizes {sizes}"
        
        return True, "Valid solution"

def main():
    # Read input
    N, K = map(int, input().split())
    
    distance_matrix = []
    for i in range(N + 1):
        row = list(map(int, input().split()))
        distance_matrix.append(row)
    
    # Solve
    solver = BalancedVRPSolver(N, K, distance_matrix)
    routes, max_distance = solver.solve()
    
    # Validate solution
    is_valid, message = solver.validate_solution(routes)
    if not is_valid:
        print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)
    
    # Output
    print(K)
    for route in routes:
        print(len(route))
        print(' '.join(map(str, route)))

if __name__ == "__main__":
    main()