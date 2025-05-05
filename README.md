# delaunayTriangulation

This is a Python implementation of Randomized Incremental Delaunay triangulation aglorithm, as well as a modification of the algorithm to support priority based nearest neighbor search, as described in the paper: Efficient Partially Persistent Data Structures for Priority-Based
Nearest Neighbor Queries. 

## Project Structure
```
delaunayTriangulation/
├── BaseRICAlgo
│   ├── Delaunay.py
│   ├── DCEL.py
│   ├── tools.py
├── ModifiedRICAlgo
│   ├── Delaunay.py
│   ├── persistentDCEL.py
│   ├── tools.py
├── README.md
├── .gitignore
├── .devcontainer
│   ├── devcontainer.json
│   ├── Dockerfile
│   ├── requirements.txt
```

## BaseRICAlgo

This directory contains the implementation of the base randomized incremental Delaunay triangulation algorithm. The main components are:
- `Delaunay.py`: The main class that implements the Delaunay triangulation algorithm.
- `DCEL.py`: The data structure used to represent the triangulation. It includes the half-edge data structure and methods for manipulating it.
- `tools.py`: Utility functions for handling geometric operations, such as checking the orientation of points and calculating the circumcircle of triangles.

## ModifiedRICAlgo
This directory contains the modified version of the randomized incremental Delaunay triangulation algorithm. The main components are:
- `Delaunay.py`: The main class that implements the modified Delaunay triangulation algorithm with priority based nearest neighbor search.
- `persistentDCEL.py`: The data structure used to represent the triangulation. It includes the half-edge data structure and methods for manipulating it. It is made persistent to support the modified algorithm.
- `tools.py`: Utility functions for handling geometric operations, such as checking the orientation of points and calculating the circumcircle of triangles. 

## Usage
To use the Delaunay triangulation algorithm, you can import the `DeluanayIncremental` class from either `BaseRICAlgo` or `ModifiedRICAlgo` and create an instance of it on a set of points. On instance creation, the triangulation will be performed. If using the modified version, each point must be assigned a priority. Both implementations support the `closest_point()` method to retrieve the closest point to a given point in the triangulation. The modified version requires the `priority` parameter to be passed to the `closest_point()` method, and it will return the closest point with priority >= the passed priority.



