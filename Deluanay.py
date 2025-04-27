import random
import DCEL

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
        self.face = face  # The face associated with this node, a triple of vertices
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

class DeluanayIncremental:
    """
    Class to implement the Deluanay triangulation algorithm using the DCEL data structure.
    """

    def __init__(self, points):
        self.dcel = DCEL.DCEL()
        self.points = points ## shuffle the points, rand permutation
        random.shuffle(self.points)
        self.historyDAG = HistoryDAG()
        self.create_supertirangle() ## create the supertriangle
        for point in self.points:
            # print(f"Inserting point: {point}")
            self.insert_point(point)
        
        # self.dcel.plot()
        
    def create_supertirangle(self):
        """
        Create a supertriangle that contains all the points in the set.
        """
        
        # Create a supertriangle that contains all the points
        super_triangle_endpoints = [DCEL.Vertex((-10000, -10000)), DCEL.Vertex((10000, -10000)), DCEL.Vertex((0, 10000))]
        face = self.dcel.create_face(super_triangle_endpoints)
        self.historyDAG.root.face = face
    
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
            raise ValueError("No triangle found for the point.")
        
        # Split the triangle into three new triangles
        verts = self.dcel.get_face_vertices(triangle.face)
        triangle.set_old()  # Mark the triangle as old
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            new_face = self.dcel.create_face([v1, v2, point])
            new_node = HistoryDAGNode(new_face)
            triangle.children.append(new_node) ## add to the history DAG
            self.legalize_edge(point, new_face.outer_edge) ## check for illegal edge from v1, v2
                

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
    
    def legalize_edge(self, p, edge: DCEL.Edge):
        if edge.twin is None or edge.twin.face is None:
            return  # Nothing to legalize

        new_face = edge.twin.face

        if self.dcel.in_circumcircle(new_face, p):
            twin_edge = edge.twin
            
            # Save the vertices before removing the faces
            a = edge.start
            b = edge.end
            c = edge.next.end
            d = twin_edge.next.end

            current_node = self.find_triangle(((a.x + b.x + c.x)/3, (a.y + b.y + c.y)/3))
            twin_node = self.find_triangle(((b.x + a.x + d.x)/3, (b.y + a.y + d.y)/3))
            
            self.dcel.remove_face(edge.face)
            self.dcel.remove_face(twin_edge.face)
            
            # Create the two new faces after flipping
            new_face1 = self.dcel.create_face([c, a, d])
            new_face2 = self.dcel.create_face([d, b, c])
            
            new_node1 = HistoryDAGNode(new_face1)
            new_node2 = HistoryDAGNode(new_face2)

            current_node.add_children([new_node1, new_node2])
            twin_node.add_children([new_node1, new_node2])
            current_node.set_old()
            twin_node.set_old()
            
            # Recursively legalize the new edges
            self.legalize_edge(p, new_face1.outer_edge.prev)
            self.legalize_edge(p, new_face2.outer_edge.prev)

        
DeluanayIncremental([(random.random()*150, random.random()*150) for _ in range(100)])