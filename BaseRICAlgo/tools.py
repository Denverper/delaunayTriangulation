from typing import List

class Vector: ## 2D vector
    def __init__(self, arr: List[int]):
        self.vec = tuple([i for i in arr]) ##make into tuple, immutable
        self.size = len(arr)
        
    def __add__(self, b):
        temp = []
        for j in range(len(self.vec)):
            temp.append(self.vec[j] + b.vec[j])
        return Vector(temp)
    
    def scale(self, val):
        return Vector([num*val for num in self.vec])
    
    def __sub__(self, b):
        ## a - b == a + (-b)
        neg_b = b.scale(-1)
        return self + neg_b
    
    def __str__(self):
        return self.vec.__str__()
    

class Point: ## 2D point
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __sub__(self, p2):
        return Vector([self.x - p2.x, self.y - p2.y])
    
    def __str__(self):
        return f'({self.x},{self.y})'
    
    def __eq__(self, p2):
        return self.x == p2.x and self.y == p2.y
    

def dot(vec1: Vector, vec2 : Vector):
    ## > 0: less than pi/2
    ## < 0: larger than pi/2
    ## == 0: perpendictlar 
    if vec1.size == vec2.size:
        val = 0
        for i in range(vec1.size):
            val += (vec1.vec[i] * vec2.vec[i])
        return val
    return None

def cross(a: Vector, b : Vector):
    ## > 0: a is clockwise from b
    ## < 0: b is clockwise from a
    ## == 0: parallel
    if a.size == b.size:
        return a.vec[0]*b.vec[1] - a.vec[1]*b.vec[0]
    return None

def collinear(p:Point,q:Point,r:Point):
    return True if cross(q-p, r-p) == 0 else False

def between(p:Point,q:Point,r:Point):
    if not collinear(p,q,r) or (p == q and q != r):
        return False
    if p.x != q.x:
        return min(p.x,q.x) <= r.x and r.x <= max(p.x,q.x)
    return min(p.y,q.y) <= r.y and r.y <= max(p.y,q.y)

def orient(p:Point,q:Point,r:Point):
    val = cross(q-p, r-p)
    if val == 0: ##collinear
        return 0
    return 1 if val > 0 else -1 ## 1 if left, -1 if right

