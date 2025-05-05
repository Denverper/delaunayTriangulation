import random
from matplotlib import pyplot as plt
import numpy as np
import persistentDCEL as DCEL
import random
from pympler import asizeof
import time

class HistoryDAG:
    """
    Class to implement the history DAG for the DCEL.
    """
    __slots__ = ('root')
    def __init__(self):
        self.root = HistoryDAGNode(None, None) # List to store history of changes
        
    def __str__(self):
        def print_tree(node, depth=0):
            result = "  " * depth + f"Node(face={node.face}, version={node.face.outer_edge.version})\n"
            for child in node.children:
                result += print_tree(child, depth + 1)
            return result
        
        return f"HistoryDAG:\n{print_tree(self.root)}"

class HistoryDAGNode:
    """
    Class to represent a node in the history DAG.
    """
    __slots__ = ('face', 'children', 'version')
    def __init__(self, face, version):
        self.face = face  # The face associated with this node, a triple of vertices
        self.children = []  # List to store child nodes, at most 3
        self.version = version

    def add_children(self, children):
        if len(children) > 3:
            raise Exception("Cannot add more than 3 children to a node.")
        for child in children:
            self.children.append(child)
        
    def __str__(self):
        return f"Node(face={self.face})"

    def __repr__(self):
        return f"Node(face={self.face})"

class DelaunayIncremental:
    """
    Class to implement the Delaunay triangulation algorithm using the DCEL data structure.
    This structure is made partiallypersistent by versioning the history DAG and the DCEL, while maintaining linear space.
    """
    __slots__ = ('dcel', 'points', 'historyDAG', 'global_version', 'version_mapping')

    def __init__(self, points):
        self.dcel = DCEL.DCEL()
        self.points = points ## shuffle the points, rand permutation
        random.shuffle(self.points) ## assume points are in the form of ((x,y), p)
        self.points = self.batch_by_priority(self.points) ## batch the points by priority
        self.historyDAG = HistoryDAG()
        self.global_version = -1
        self.version_mapping = [] ## list of tuples (priotity, version)
        self.create_supertriangle() ## create the supertriangle
        self.global_version+=1
        curr_p = self.points[0][1] ## get the priority of the first point, the highest
        for point in self.points:
            if point[1] != curr_p: ## next batch reached, increment version and add to version mapping
                self.version_mapping.append((curr_p, self.global_version))
                curr_p = point[1]
                self.global_version+=1
            self.insert_point(point[0])
        
        self.version_mapping.append((curr_p, self.global_version)) ## add the last version mapping
        
    @staticmethod 
    def batch_by_priority(points):
        """
        Batch the points by priority from increasing to decreasing.
        
        Args:
            points: List of points to batch.
            
        Returns:
            List of points in the order of priority.
        """
        return sorted(points, key=lambda p: -p[1]) ## sort by priority, highest first, stable sort so points are in permuted order within batches
        
    @staticmethod
    def _pointer_check(children, version): ## ensure all children of the node are not more recent than the version given
        for child in children:
            if child.face.outer_edge.version > version:
                return False
        return True
    
    def create_supertriangle(self):
        """
        Create a supertriangle that adaptively surrounds all input points.
        """
        # Extract all x and y coordinates
        xs = [p[0][0] for p in self.points]
        ys = [p[0][1] for p in self.points]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Calculate width and height of bounding box
        width = max_x - min_x
        height = max_y - min_y
        
        padding = max(width, height) * 10  # 10x bigger than biggest dimension
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        p1 = (center_x - 2 * padding, center_y - padding)
        p2 = (center_x + 2 * padding, center_y - padding)
        p3 = (center_x, center_y + 2 * padding)

        super_triangle_endpoints = [
            DCEL.Vertex(p1),
            DCEL.Vertex(p2),
            DCEL.Vertex(p3)
        ]

        face = self.dcel.create_face(super_triangle_endpoints, self.global_version)
        self.historyDAG.root.face = face
        face.dag_node = self.historyDAG.root
        self.historyDAG.root.version = self.global_version

    def insert_point(self, point):
        """
        Insert a point into the triangulation.
        
        Args:
            point: The point to insert.
        """
        # Find the triangle that contains the point at the current version
        triangle = self.find_triangle(point)
        point = DCEL.Vertex(point)
        if triangle is None:
            raise ValueError(f"No triangle found for the {point}.")
        
        # Split the triangle into three new triangles
        verts = self.dcel.get_face_vertices(triangle.face)
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            new_face = self.dcel.create_face([v1, v2, point], self.global_version)
            new_node = HistoryDAGNode(new_face, version=self.global_version)
            new_face.dag_node = new_node
            triangle.children.append(new_node) ## add to the history DAG
            self.legalize_edge(point, new_face.outer_edge, new_node) ## check for illegal edge from v1, v2

    def find_triangle(self, point, version=None):
        """
        Find the triangle that contains the point in version v.
        
        Args:
            point: The point to check.
            version: The version to check against. If None, use the current version.
            
        Returns:
            HistoryDAGNode: The triangle that contains the point in the History DAG.
        """
        if version is None:
            version = self.global_version
            
        curr = self.historyDAG.root
        while len(curr.children) != 0 and self._pointer_check(curr.children, version):
            found = False
            for child in curr.children:
                if self.dcel.contains_point(child.face, point):
                    curr = child
                    found = True
                    break
            if not found:
                return None
        return curr
    
    def legalize_edge(self, p, edge: DCEL.Edge, current_node: HistoryDAGNode): ## only called in conctruction, so current version
        """
        Legalize the edge by checking if the point p is in the circumcircle of the triangle formed by the give edge and its twin.
        This function is only ever called in the construction phase, so the version is always the current version.

        Args:
            p (DCEL.Vertex): the vertex that was added
            edge (DCEL.Edge): the edge to check
            current_node (HistoryDAGNode): the node of the edge in the history DAG
        """
        if edge.get_current_twin() is None or edge.get_current_twin().face is None:
            return

        twin_edge = edge.get_current_twin()
        twin_node = twin_edge.face.dag_node

        if twin_node is None:
            return

        if self.dcel.in_circumcircle(twin_edge.face, p):
            a = edge.start
            b = edge.end
            d = twin_edge.next.end

            self.dcel.remove_face(edge.face, self.global_version)
            self.dcel.remove_face(twin_edge.face, self.global_version)

            new_face1 = self.dcel.create_face([a, d, p], self.global_version) 
            new_face2 = self.dcel.create_face([d, b, p], self.global_version)

            new_node1 = HistoryDAGNode(new_face1, self.global_version)
            new_node2 = HistoryDAGNode(new_face2, self.global_version)
            
            new_face1.dag_node = new_node1
            new_face2.dag_node = new_node2

            current_node.add_children([new_node1, new_node2])
            twin_node.add_children([new_node1, new_node2])

            # Recursively legalize, this is not working, maybe wrong edges being passed
            self.legalize_edge(p, new_face1.outer_edge, new_node1)
            self.legalize_edge(p, new_face2.outer_edge, new_node2)
       
    def closest_point(self, point, tau):
        """
        Find the closest point to the given point with priority >= tau.

        Args:
            point (tuple(float, float)): the point to find the nearest neighbor of priority >= tau
            tau (int): the priority threshold

        Returns:
            DCEL.Vertex: the closest point found in the triangulation
        """
        query_vertex = DCEL.Vertex(point)
        version = self.get_version(tau) 
        triangle = self.find_triangle(point, version)
        
        # Initialize with vertices of the triangle
        closest = None
        min_distance = float('inf')
        
        for vertex in self.dcel.get_face_vertices(triangle.face):
            d = self.distance(query_vertex, vertex)
            if d < min_distance:
                min_distance = d
                closest = vertex

        # Now simulate insertions along the edges
        for edge in self.dcel.get_face_edges(triangle.face):
            cand_vertex, cand_distance = self.simulate_insertion(query_vertex, edge, min_distance, version)
            if cand_vertex is not None and cand_distance < min_distance:
                min_distance = cand_distance
                closest = cand_vertex

        return closest

    def simulate_insertion(self, p, edge: DCEL.Edge, best_distance, version):
        """
        Simulate the insertion process and find a closer vertex recursively. Finds all the edges that would be created if the point were inserted.
        Does not modify the triangulation.

        Args:
            p: DCEL.Vertex (query point)
            edge: DCEL.Edge (edge to check)
            v: version
            best_distance: current best known distance

        Returns:
            (closest_vertex, distance): the best vertex found during the simulation
        """    
        if edge.get_versioned_twin(version) is None or edge.get_versioned_twin(version).face is None:
            return (None, best_distance)

        twin_edge = edge.get_versioned_twin(version)
        if twin_edge is None or twin_edge.face is None:
            return (None, best_distance)
        
        new_face = twin_edge.face
        
        if self.dcel.in_circumcircle(new_face, p):
            a = edge.start
            b = edge.end
            d = twin_edge.next.end
            
            closest_vertex = None
            
            dist = self.distance(p, d)
            if dist < best_distance:
                best_distance = dist
                closest_vertex = d
            
            e1 = self.dcel.get_versioned_edge(a, d, version)
            e2 = self.dcel.get_versioned_edge(d, b, version)

            cand1, dist1 = self.simulate_insertion(p, e1, best_distance, version)
            if dist1 < best_distance:
                best_distance = dist1
                closest_vertex = cand1

            
            cand2, dist2 = self.simulate_insertion(p, e2, best_distance, version)
            if dist2 < best_distance:
                best_distance = dist2
                closest_vertex = cand2
            return (closest_vertex, best_distance)
        else:
            return (None, best_distance)

    @staticmethod
    def distance(p: DCEL.Vertex, q: DCEL.Vertex):
        """
        Calculate the Euclidean distance between two points.

        Args:
            p (DCEL.Vertex): point 1
            q (DCEL.Vertex): point 2

        Returns:
            float: the distance between the two points
        """
        return ((p.x - q.x) ** 2 + (p.y - q.y) ** 2) ** 0.5
    
    def get_version(self, tau):
        """
        Perform a binary search on self.version_mapping to find the version
        where version_mapping[0] <= tau. This returns the version that corresponds to the given priority.

        Args:
            tau: The priority value to search for.

        Returns:
            The version corresponding to the found priority.
        """
        if tau <= self.version_mapping[-1][0]:
            return self.version_mapping[-1][1]
        if tau >= self.version_mapping[0][0]:
            return self.version_mapping[0][1]
        left, right = 0, len(self.version_mapping) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.version_mapping[mid][0] >= tau:  # Compare priority
                if mid == len(self.version_mapping) - 1 or self.version_mapping[mid + 1][0] < tau:
                    return self.version_mapping[mid][1]  # Return version
                else:
                    left = mid + 1
            else:
                right = mid - 1
        return -1  # Return -1 if no valid version is found
        
def brute_force_closest(points, query, priority):
    """
    Brute force method to find the closest point to the query point with priority >= tau.
    
    Args:
        points: List of points to check.
        query: The query point.
        
    Returns:
        The closest point to the query point.
    """
    closest = None
    min_distance = float('inf')
    
    for point, p in points:
        if p >= priority:
            d = DelaunayIncremental.distance(DCEL.Vertex(query), DCEL.Vertex(point))
            if d < min_distance:
                min_distance = d
                closest = point
                
    return closest

def analyze_complexity_var_points(point_vals, b_rep, q_rep):
    """
    Function to analyze the complexity of the Delaunay triangulation algorithm with varying number of points and a fixed number of priorities.
    It measures the build time, query time, and space allocation for different numbers of points.
    It generates plots for build time, query time, and space allocation as a function of the number of points.

    Args:
        point_vals (list[int]): the numbers of points to test
        b_rep (int): the number of times to build the triangulation
        q_rep (int): the number of queries to perform on each build
    """
    ## buffers for overall complexity
    space = []
    query = []
    build = []
    
    for num_points in point_vals: 
        total_q = 0
        total_b = 0
        total_space = 0
        
        for _ in range(b_rep): ## build it b_rep times
            test_data = []
            ## generate random points with priorities
            for _ in range(num_points):  
                x = random.uniform(-num_points, num_points)
                y = random.uniform(-num_points, num_points)
                priority = random.randint(-100, 100)
                test_data.append(((x, y), priority))
            
            ## build D(S) and time construction time
            start_b = time.perf_counter()
            triang = DelaunayIncremental(test_data)
            end_b = time.perf_counter()
            total_b +=(end_b -start_b)

            ## check space allocation
            size = asizeof.asizeof(triang)
            total_space += size
        
            ## check query time for random points on each built D(S)
            for _ in range(q_rep):
                point_to_check = (random.uniform(-num_points, num_points), random.uniform(-num_points, num_points))
                priority = random.randint(-100, 100)
                start_q = time.time()
                triang.closest_point(point_to_check, priority)
                end_q = time.time()
                total_q += (end_q-start_q)
        
        avg_q = total_q/(q_rep*b_rep)
        avg_b = total_b/b_rep
        avg_space = total_space/b_rep
        
        query.append(avg_q)
        build.append(avg_b)
        space.append(avg_space)
        
        print(f"{num_points} Points, {b_rep} Build Iterations, {q_rep * b_rep} Query Iterations, Build: {avg_b:.6f}, Query: {avg_q:.6f}, Space: {avg_space:.2f} bytes, Total Time: {(total_b + total_q):.6f} seconds, completion time: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    
    with open("./ModifiedRICAlgo/TmpRuntimeData/complexity.txt", "w") as f:
        f.write(f"Points, Space (Bytes), Build (s), Query (s)\n")
        for i in range(len(point_vals)):
            f.write(f"{point_vals[i]}, {space[i]}, {build[i]}, {query[i]}\n")
            
    plt.plot(point_vals, space, marker='o')
    plt.title("Space Allocation (Bytes) vs Number of Points for Building Delaunay")
    plt.xlabel("Number of Points")
    plt.ylabel("Average Space in Bytes")
    plt.grid()
    plt.savefig("./spacePers.png")
    plt.show()
    
    plt.plot(point_vals, build, marker='o')
    plt.title("Build Time vs Number of Points for Building Delaunay")
    plt.xlabel("Number of Points")
    plt.ylabel("Average Build Time (s)")
    plt.grid()
    plt.savefig("./buildPers.png")
    plt.show()
    
    plt.plot(point_vals, query, marker='o')
    plt.title("Query Time vs Number of Points for Building Delaunay")
    plt.xlabel("Number of Points")
    plt.ylabel("Average Query Time (s)")
    plt.grid()
    plt.savefig("./queryPers.png")
    plt.show()

def analyze_complexity_var_p(p_vals, points, b_rep, q_rep):
    """
    Function to analyze the complexity of the Delaunay triangulation algorithm with varying number of priorities and a fixed number of points.
    It measures the build time, query time, and space allocation for different numbers of priorities.
    It generates plots for build time, query time, and space allocation as a function of the number of priorities.
    
    Args:
        p_vals (list[int]): values of priorities to test
        points (int): The number of points to test with each priority
        b_rep (int): the number of times to build the triangulation
        q_rep (int): the number of queries to perform on each build
    """
    ## buffers for overall complexity
    space = []
    query = []
    build = []
    print(f"{points} Points, {b_rep} Build Iterations, {q_rep * b_rep} Query Iterations, {len(p_vals)} Priorities")
    for p in p_vals:
        total_q = 0
        total_b = 0
        total_space = 0
        
        for _ in range(b_rep): ## build it b_rep times
            test_data = []
            ## generate random points with priorities
            for _ in range(points):  
                x = random.uniform(-points, points)
                y = random.uniform(-points, points)
                priority = random.randint(0, p)
                test_data.append(((x, y), priority))
            
            ## build D(S) and time construction time
            start_b = time.perf_counter()
            triang = DelaunayIncremental(test_data)
            end_b = time.perf_counter()
            total_b +=(end_b -start_b)

            ## check space allocation
            size = asizeof.asizeof(triang)
            total_space += size
        
            ## check query time for random points on each built D(S)
            for _ in range(q_rep):
                point_to_check = (random.uniform(-points, points), random.uniform(-points, points))
                priority = random.randint(0, p)
                start_q = time.time()
                triang.closest_point(point_to_check, priority)
                end_q = time.time()
                total_q += (end_q-start_q)
        
        avg_q = total_q/(q_rep*b_rep)
        avg_b = total_b/b_rep
        avg_space = total_space/b_rep
        
        query.append(avg_q)
        build.append(avg_b)
        space.append(avg_space)
        
        print(f"{p} Priorities, Build: {avg_b:.6f}, Query: {avg_q:.6f}, Space: {avg_space:.2f} bytes, Total Time: {(total_b + total_q):.6f} seconds, completion time: {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    print(f"Full results for {points} points with differing priorities:")
    print("Priorities, Space, Build, Query")
    for i in range(len(p_vals)):
        print(f'{p_vals[i]}, {space[i]}, {build[i]}, {query[i]}')

def benchmark_and_verify(triang, test_data, n_tests=1000):
    """
    Function to benchmark the Delaunay triangulation algorithm and verify its correctness against a brute-force method.
    Generates random points and priorities to test the closest point method.
    It compares the results with a brute-force method and prints any mismatches.
    It also prints the time taken for each test and the brute-force method.

    Args:
        triang (DelaunayIncremental): the triangulation object to test
        test_data (list[int]): the list of points to test against, needed for brute force
        n_tests (int, optional): Number of random tests to run. Defaults to 1000.
    """
    for i in range(n_tests):
        # Pick a random point to query
        point_to_check = (random.uniform(-200, 200), random.uniform(-200, 200))
        priority = random.randint(-5, 5)

        # Time the non-brute-force method
        start_time = time.perf_counter()
        result = triang.closest_point(point_to_check, priority)
        elapsed = time.perf_counter() - start_time

        # Compare with brute force
        b_start_time = time.perf_counter()
        expected = brute_force_closest(test_data, point_to_check, priority)
        b_elapsed = time.perf_counter() - b_start_time

        # Verify
        if result is None or expected is None:
            if result != expected:
                print(f"Mismatch at test {i}:")
                print(f"Point: {point_to_check}, Priority: {priority}")
                print(f"Expected: {expected}, Got: {result}")
                break
        elif result.coordinates != expected:
            print(f"Mismatch at test {i}:")
            print(f"Point: {point_to_check}, Priority: {priority}")
            print(f"Expected: {expected} - Dist ({DelaunayIncremental.distance(DCEL.Vertex(point_to_check), DCEL.Vertex(expected))}), Got: {result.coordinates} - Dist ({DelaunayIncremental.distance(DCEL.Vertex(point_to_check), result)})")
            break

        print(f"Test {i}: OK, time: {elapsed:.6f} seconds, brute force time: {b_elapsed:.6f} seconds")

if __name__ == "__main__":
    p_vals = np.unique(np.round(np.logspace(np.log10(1), np.log10(10000), 15)).astype(int)).tolist()
    analyze_complexity_var_p(p_vals, 10000, b_rep=100, q_rep=1000)
