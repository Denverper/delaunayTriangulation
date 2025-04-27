import random
from matplotlib import pyplot as plt
import numpy as np
import tools

class DCEL:
    def __init__(self):
        """
        Initializes the DCEL object, stores, our vertices, edges, and faces
        """
        self.vertices = set()  # List of vertices
        self.edges = {}  # Hash table for edges (key: (start, end), value: Edge)
        self.faces = set()  # set of faces

    def get_or_create_vertex(self, v):
        """
        Function to get or create a vertex in the DCEL, if one already exists

        Args:
            v (Vertex): The vertex to try and find

        Returns:
            Vertex: The Vertex, either new or already in DCEL
        """
        if v in self.vertices:
            return v
        self.vertices.add(v)
        return v

    def create_face(self, points):
        """
        Make a new face from a list of points (vertices).
        Creates new half edges for this face, and uses twins if they already exist
        if not, make them.

        Args:
            points (list[Vertex]): the ordered list of vertices that make up the face

        Raises:
            ValueError: not enough points to make a face

        Returns:
            Face: A new face in the DCEL
        """
        if len(points) < 3:
            raise ValueError("A face must have at least 3 vertices.")

        ## Ensure we work with our DCEL vertices
        vertices = [self.get_or_create_vertex(p) for p in points]
        new_edges = []
        
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            
            new_edge = Edge(v1, v2) ## make new edge
            
            twin_key = (v2.coordinates, v1.coordinates) ## check if there already is a twin, if not, make one
            if twin_key in self.edges:
                twin_edge = self.edges[twin_key]
                new_edge.twin = twin_edge
                twin_edge.twin = new_edge
            else:
                twin_edge = Edge(v2, v1)
                new_edge.twin = twin_edge
                twin_edge.twin = new_edge
            
            self.edges[(v1.coordinates, v2.coordinates)] = new_edge
            self.edges[(v2.coordinates, v1.coordinates)] = twin_edge
            new_edges.append(new_edge)
        
        for i, edge in enumerate(new_edges):
            edge.next = new_edges[(i + 1) % len(new_edges)]
            edge.prev = new_edges[i - 1]

        face = Face(new_edges[0])
        for edge in new_edges:
            edge.face = face

        self.faces.add(face)
        return face

                
    def remove_face(self, face):
        """
        Remove a face from the DCEL, and remove the edges that belong to the face, if the twin is None, remove bothe edges, if not, set its half edge to None.

        Args:
            face Face: the face to remove
        """
        edge = face.outer_edge
        first_edge = edge
        self.faces.discard(face)  # Remove face from set   
        
        while True:
            twin_edge = edge.twin
            if (edge.start.coordinates, edge.end.coordinates) in self.edges:
                self.edges[(edge.start.coordinates, edge.end.coordinates)].face = None  # Remove edge from hash table
            
                # Check if the twin has a face, if not, remove it
                if twin_edge.face is None:
                    del self.edges[(edge.end.coordinates, edge.start.coordinates)]
                    del self.edges[(edge.start.coordinates, edge.end.coordinates)]
                            
            edge = edge.next
            if edge == first_edge:
                break  # Loop through face complete
        
    def get_face_vertices(self, face):
        """
        Helper method to get the vertices for a given face

        Args:
            face (Face): The face to get vertices for

        Returns:
            list[Vertex]: the ordered list of vertices from the face
        """
        verts = []
        edge = face.outer_edge
        first_edge = edge
        while True:
            verts.append(edge.start)
            edge = edge.next
            if edge == first_edge:
                break
        return verts

    def get_face_edges(self, face):
        """
        Helper method to get the edges defining a face.

        Args:
            face (Face): The face to get edges for

        Returns:
            list[Edge]: the ordered list of edges from the face
        """
        edges = []
        edge = face.outer_edge
        first_edge = edge
        while True:
            edges.append(edge)
            edge = edge.next
            if edge == first_edge:
                break
        return edges

    def contains_point(self, face, point):
        """
        Check if a point is inside a triangle (face) using the orientation test.

        Args:
            face (Face): The face to check
            point (Vertex): The point to check

        Returns:
            bool: True if the point is inside the face, False otherwise
        """
        vertices = self.get_face_vertices(face)
        n = len(vertices)
        if type(point) is Vertex:
            point = tools.Point(point.x, point.y)
        else:
            point = tools.Point(point[0], point[1])

        
        for i in range(n):
            v1 = tools.Point(vertices[i].x, vertices[i].y)
            v2 = tools.Point(vertices[(i + 1) % n].x, vertices[(i + 1) % n].y)
            
            if tools.orient(v1, v2, point) == -1:
                return False
        return True
    
    def in_circumcircle(self, face, point):
        """
        Check if a point is inside the circumcircle of a triangle (face).

        Args:
            face (Face): The face to check
            point (Vertex): The point to check

        Returns:
            bool: True if the point is inside the circumcircle, False otherwise
        """
        vertices = self.get_face_vertices(face)
        if len(vertices) != 3:
            raise ValueError("Face must have exactly three vertices.")

        a, b, c = vertices
        d = point

        # Matrix determinant approach
        mat = [
            [a.x - d.x, a.y - d.y, (a.x - d.x)**2 + (a.y - d.y)**2],
            [b.x - d.x, b.y - d.y, (b.x - d.x)**2 + (b.y - d.y)**2],
            [c.x - d.x, c.y - d.y, (c.x - d.x)**2 + (c.y - d.y)**2],
        ]

        # Compute the determinant
        det = (
            mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2])
            - mat[0][1] * (mat[1][0] * mat[2][2] - mat[2][0] * mat[1][2])
            + mat[0][2] * (mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1])
        )

        # If det > 0, point lies inside the circumcircle
        return det > 0

    def plot(self, zoom=True, zoom_center=(0,0), zoom_radius=80.0):
        """
        Plot the current state of the DCEL in 2D with optional zoom functionality.

        Args:
            zoom (bool): Whether to enable zooming.
            zoom_center (tuple): The (x, y) coordinates of the zoom center.
            zoom_radius (float): The radius around the zoom center to display.
        """
        fig, ax = plt.subplots()
        
        # Plot edges
        for edge in self.edges.values():
            x_coords = [edge.start.x, edge.end.x]
            y_coords = [edge.start.y, edge.end.y]
            ax.plot(x_coords, y_coords, 'b-', alpha=0.7)  # Blue lines for edges
        
        # Plot vertices
        for vertex in self.vertices:
            ax.plot(vertex.x, vertex.y, 'ro')  # Red dots for vertices
            ax.text(vertex.x, vertex.y, f"({vertex.x:.1f}, {vertex.y:.1f})", fontsize=8, color='black')
        
        if zoom and zoom_center:
            x_min = zoom_center[0] - zoom_radius
            x_max = zoom_center[0] + zoom_radius
            y_min = zoom_center[1] - zoom_radius
            y_max = zoom_center[1] + zoom_radius
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        ax.set_aspect('equal', adjustable='datalim')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('DCEL State')
        plt.show()

    
    def __repr__(self): ## for degbugging
        ret = f'DCEL: {len(self.vertices)} vertices, {len(self.edges)} edges, {len(self.faces)} faces'
        ret += '\nVertices:\n' + ',\n'.join([f"{v.coordinates}" for v in self.vertices])
        ret += '\nEdges:\n' + ',\n'.join([f"{str(e.start.coordinates)}->{str(e.end.coordinates)}" for e in self.edges.values()])
        ret += '\nFaces: ' + ', '.join([str(f) for f in self.faces])
        return ret

    def __str__(self):
        return self.__repr__()

class Edge: 
    """
    Edge class to be held in the DCEL
    """
    def __init__(self, start, end, face = None, twin = None , next = None, prev = None):
        """
        Initialize availible variables

        Args:
            start (Vertex): _description_
            end (Vertex): _description_
            face (Face, optional): The incident face. Defaults to None.
            twin (Edge, optional): the twin of the half edge. Defaults to None.
            next (Edge, optional): the next edge inident on the same face. Defaults to None.
            prev (Edge, optional): the previous edge incident on the same face. Defaults to None.
        """
        self.id = random.randint(0, 1000) # Unique identifier
        self.start = start
        self.end = end
        self.twin = None  # The opposite edge
        self.next = None  # Next edge in the face cycle
        self.prev = None  # Previous edge in the face cycle
        self.face = None  # The face that this edge belongs to
        
    def __repr__(self): ## for degbugging
        return f'Edge: ({self.start}, {self.end}) id: {self.id}'
    
    def __str__(self):
        return f'Edge: ({self.start}, {self.end})'

class Vertex:
    """
    Vertex class to be held in the DCEL
    """
    def __init__(self, coordinates):
        """
        Initialize Vertex to hold a reference to an edge incident on it.
        Args:
            coordinates (Vertex): the position of the vertex
        """
        self.coordinates = coordinates
        self.x, self.y = coordinates
        self.edge = None
        
    def __repr__(self): ## for degbugging
        return f'Vertex: ({self.coordinates})'
    
    def __str__(self):
        return f'Vertex: ({self.coordinates})'

class Face:
    """
    Face class to be held in the DCEL
    """
    def __init__(self, edge):
        """
        Initialize the face to hold a edge incident on the face, any edge that refers to this face as its face

        Args:
            edge (Edge): the edge that is indicent on the face
        """
        self.outer_edge = edge
        
    def get_centroid(self):
        """
        Get the centroid of the face by averaging the coordinates of its vertices.

        Returns:
            tuple: The centroid coordinates (x, y)
        """
        vertices = []
        edge = self.outer_edge
        first_edge = edge
        while True:
            vertices.append(edge.start)
            edge = edge.next
            if edge == first_edge:
                break
        x = sum(v.x for v in vertices) / len(vertices)
        y = sum(v.y for v in vertices) / len(vertices)
        return (x, y)
        
    def __repr__(self): ## for degbugging
        return f'Face: {self.outer_edge})'
    
    def __str__(self):
        return f'Face: {self.outer_edge})'
    
    
