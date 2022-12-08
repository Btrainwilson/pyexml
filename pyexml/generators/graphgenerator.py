import numpy as np
import networkx 

def add_edge(graph, i, j):
    graph[i, j] = 1
    graph[j, i] = 1

def generateRect(n, m):

    graph = networkx.grid_2d_graph(n, m)
    return networkx.to_numpy_array(graph)

def generateKings(n, m, laplacian = False):

    square_graph = networkx.to_numpy_array(networkx.grid_2d_graph(n, m))

    #Bulk
    for i in range(1, n - 1):
        for j in range(1, m - 1):

            v = i*m + j
            v_next = (i+1)*m + j
            v_prev = (i-1)*m + j

            add_edge(square_graph, v, v_next + 1)
            add_edge(square_graph, v, v_next - 1)

            add_edge(square_graph, v, v_prev + 1)
            add_edge(square_graph, v, v_prev - 1)

    add_edge(square_graph, 1, m)
    add_edge(square_graph, m-2, 2*m-1)
    add_edge(square_graph, (n - 2)*m, (n - 1)*m + 1)
    add_edge(square_graph, (n - 1)*m + m - 2, (n - 2)*m + m - 1)

    if laplacian:
        for i in range(square_graph.shape[0]):
            square_graph[i,i] = -np.sum(square_graph[i,:])
    

    return square_graph