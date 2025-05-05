import random
import BaseRICAlgo.DCEL as DCEL
import time
import matplotlib.pyplot as plt

class HistoryDAG:
    """
    Class to implement the history DAG for the DCEL.
    """

    def __init__(self):
        self.root = HistoryDAGNode(None) # List to store history of changes
        
    def __str__(self):
        def print_tree(node, depth=0):
            result = "  " * depth + f"Node(face={node.face}, old={node.old})\n"
            for child in node.children:
                result += print_tree(child, depth + 1)
            return result
        
        return f"HistoryDAG:\n{print_tree(self.root)}"


class HistoryDAGNode:
    """
    Class to represent a node in the history DAG.
    """

    def __init__(self, face):
        self.face = face  # The face associated with this node, a triple of vertices, points back at node
        self.children = []  # List to store child nodes, at most 3
        self.old = False  # Flag to indicate if this node is old

    def add_children(self, children):
        if len(children) > 3:
            raise Exception("Cannot add more than 3 children to a node.")
        for child in children:
            self.children.append(child)
    
    def set_old(self):
        self.old = True
        
    def __str__(self):
        return f"Node(face={self.face}, old={self.old})"

    def __repr__(self):
        return f"Node(face={self.face}, old={self.old})"

class DelaunayIncremental:
    """
    Class to implement the regular Delaunay triangulation algorithm using the DCEL data structure.
    """

    def __init__(self, points):
        """
        Initialize the Delaunay triangulation with a list of points.
        Inserts the points into the triangulation in random order and holds the triangulation in a DCEL.

        Args:
            points (list[tuple]): The points to triangulate.
        """
        self.dcel = DCEL.DCEL()
        self.points = points ## shuffle the points, rand permutation
        random.shuffle(self.points)
        self.historyDAG = HistoryDAG()
        self.create_supertriangle() ## create the supertriangle
        for point in self.points:
            self.insert_point(point)
        
        
    def create_supertriangle(self):
        """
        Create a supertriangle that dynamically surrounds all input points.
        """
        # Extract all x and y coordinates
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        
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

        face = self.dcel.create_face(super_triangle_endpoints)
        self.historyDAG.root.face = face
        face.dag_node = self.historyDAG.root

    
    def insert_point(self, point):
        """
        Insert a point into the triangulation.
        
        Args:
            point: The point to insert.
        """
        # Find the triangle that contains the point
        triangle = self.find_triangle(point)
        point = DCEL.Vertex(point)
        if triangle is None:
            raise ValueError(f"No triangle found for the {point}.")
        
        # Split the triangle into three new triangles
        verts = self.dcel.get_face_vertices(triangle.face)
        triangle.set_old()  # Mark the triangle as old
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            new_face = self.dcel.create_face([v1, v2, point])
            new_node = HistoryDAGNode(new_face)
            new_face.dag_node = new_node
            triangle.children.append(new_node) ## add to the history DAG
            self.legalize_edge(point, new_face.outer_edge, new_node) ## check for illegal edge from v1, v2
                

    def find_triangle(self, point):
        """
        Find the triangle that contains the point.
        
        Args:
            point: The point to check.
            
        Returns:
            Triangle: The triangle that contains the point in the History DAG.
        """
        curr = self.historyDAG.root
        while curr.old:
            found = False
            for child in curr.children:
                if self.dcel.contains_point(child.face, point):
                    curr = child
                    found = True
                    break
            if not found:
                return None
        return curr
    
    def legalize_edge(self, p, edge: DCEL.Edge, current_node: HistoryDAGNode):
        """
        Legalize the edge by checking if the point is in the circumcircle of the triangle. If so, flip the edge. and recurse

        Args:
            p (DCEL.Vertex): the point to check
            edge (DCEL.Edge): the edge to check
            current_node (HistoryDAGNode): the current node of the face of the edge in the history DAG
        """
        if edge.twin is None or edge.twin.face is None:
            return

        twin_edge = edge.twin
        twin_node = twin_edge.face.dag_node

        if twin_node is None:
            return

        if self.dcel.in_circumcircle(twin_edge.face, p):
            # Save the vertices
            a = edge.start
            b = edge.end
            d = twin_edge.next.end

            # Remove old faces
            self.dcel.remove_face(edge.face)
            self.dcel.remove_face(twin_edge.face)

            # Create two new faces
            new_face1 = self.dcel.create_face([a, d, p])
            new_face2 = self.dcel.create_face([d, b, p])

            new_node1 = HistoryDAGNode(new_face1)
            new_node2 = HistoryDAGNode(new_face2)
            
            new_face1.dag_node = new_node1
            new_face2.dag_node = new_node2

            current_node.add_children([new_node1, new_node2])
            twin_node.add_children([new_node1, new_node2])

            current_node.set_old()
            twin_node.set_old()
            # Recursively legalize
            self.legalize_edge(p, new_face1.outer_edge, new_node1)
            self.legalize_edge(p, new_face2.outer_edge, new_node2)
     
       
    def closest_point(self, point):
        """
        Find the closest point in the triangulation to the given point. 
        Finds the triangle that contains the point and simulates the insertion process to find the closest vertex, as a subgraph of the Delaunay trianguilaton is the nearest neightbor graph..

        Args:
            point (_type_): _description_

        Returns:
            _type_: _description_
        """
        query_vertex = DCEL.Vertex(point)
        triangle = self.find_triangle(point)
        
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
            cand_vertex, cand_distance = self.simulate_insertion(query_vertex, edge, min_distance)
            if cand_vertex is not None and cand_distance < min_distance:
                min_distance = cand_distance
                closest = cand_vertex

        return closest, min_distance

    def simulate_insertion(self, p, edge, best_distance):
        """
        Simulate the insertion process and find a closer vertex recursively.

        Args:
            p: DCEL.Vertex (query point)
            edge: DCEL.Edge
            v: version threshold
            best_distance: current best known distance

        Returns:
            (closest_vertex, distance): the best vertex found during the simulation
        """    
        if edge.twin is None or edge.twin.face is None:
            return (None, best_distance)

        twin_edge = edge.twin
        if twin_edge is None or twin_edge.face is None:
            return (None, best_distance)
        
        new_face = twin_edge.face
        
        if self.dcel.in_circumcircle(new_face, p):
            # Vertices around the flipped edge
            a = edge.start
            b = edge.end
            d = twin_edge.next.end
            
            closest_vertex = None
            
            # Check the new neighbor vertices
            dist = self.distance(p, d)
            if dist < best_distance:
                best_distance = dist
                closest_vertex = d
            
            # Recurse into the two new edges after flip
            e1 = self.dcel.get_edge(a, d)
            e2 = self.dcel.get_edge(d, b)

            cand1, dist1 = self.simulate_insertion(p, e1, best_distance)
            if dist1 < best_distance:
                best_distance = dist1
                closest_vertex = cand1

            
            cand2, dist2 = self.simulate_insertion(p, e2, best_distance)
            if dist2 < best_distance:
                best_distance = dist2
                closest_vertex = cand2
            return (closest_vertex, best_distance)
        else:
            return (None, best_distance)
            

    @staticmethod
    def distance(p: DCEL.Vertex, q: DCEL.Vertex):
        return ((p.x - q.x) ** 2 + (p.y - q.y) ** 2) ** 0.5
    
    def brute_force_closest(self, point):
        """
        Brute force method to find the closest point in the triangulation.
        
        Args:
            point: The point to check.
            
        Returns:
            Vertex: The closest vertex found.
        """
        min_distance = float('inf')
        point = DCEL.Vertex(point)
        closest_vertex = None
        for vertex in self.dcel.vertices:
            if vertex == point:
                continue
            d = self.distance(vertex, point)
            if d < min_distance:
                min_distance = d
                closest_vertex = vertex
        return closest_vertex, min_distance

def time_query(rep):
    """
    Time the query time of the Delaunay triangulation algorithm for closest point search.

    Args:
        rep (int): the number of repetitions for each point.
    """
    times = []
    point_nums = list(range(10000, 100001, 5000))
    
    for num in point_nums:
        total_q = 0
        test_data = []
        for _ in range(num): 
            x = random.uniform(-200, 200)
            y = random.uniform(-200, 200)
            test_data.append(((x, y)))
        
        triang = DelaunayIncremental(test_data)
        for _ in range(rep):
            point = (random.uniform(-200,200), random.uniform(-200,200))
            start_q = time.time()
            triang.closest_point(point)[1]
            end_q = time.time()
            total_q += (end_q-start_q)
        avg_q = total_q/rep
        times.append(avg_q)
        if total_q > 2 and rep > 1:
            rep //= 2
        print(f"{num} Points, {rep} Iterations, Query: {avg_q:3f}, Total: {total_q:3f}")

    plt.plot(point_nums, times, marker='o')
    plt.title("Query Time vs Number of Points for Building Delaunay")
    plt.xlabel("Number of Points")
    plt.ylabel("Average Query Time (s)")
    plt.grid()
    plt.savefig("./buildRIC.png")
    plt.show()
       
def time_build(rep):
    """
    Time the build time of the Delaunay triangulation algorithm.

    Args:
        rep (int): the number of repetitions for each point.
    """
    times = []
    point_nums = list(range(1000, 10000, 1000))
    
    for num in point_nums:    
        total_b = 0
        for _ in range(rep):
            test_data = []
            for _ in range(num): 
                x = random.uniform(-1000, 1000)
                y = random.uniform(-1000, 1000)
                test_data.append(((x, y)))
                
            start_build = time.time()
            DelaunayIncremental(test_data)
            end_build = time.time()
            total_b += (end_build-start_build)
        
        avg_b = total_b/rep
        times.append(avg_b)
        print(f"{num} Points, {rep} Iterations, Build: {avg_b:3f}, Total: {total_b:3f}")
        
    plt.plot(point_nums, times, marker='o')
    plt.title("Build Time vs Number of Points for Building Delaunay")
    plt.xlabel("Number of Points")
    plt.ylabel("Average Build Time (s)")
    plt.grid()
    plt.show()
    plt.savefig("./buildRIC.png")

def testplot(points):
    """
    Test the Delaunay triangulation algorithm by plotting the triangulation.

    Args:
        points (int): the number of points to generate.
    """
    test_data = []
    for _ in range(points): 
        x = random.uniform(-1000, 1000)
        y = random.uniform(-1000, 1000)
        test_data.append(((x, y)))
    triang = DelaunayIncremental(test_data)
    triang.dcel.plot(zoom_radius=500)
    
    
testplot(100)