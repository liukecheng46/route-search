from enum import Enum
from queue import PriorityQueue,Queue
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from bresenham import bresenham
from matplotlib import pyplot as plt
import networkx as nx

# north and west in origin file is confusing, so i change it to x and y 
# according to the file, the obstacle is a cube 
def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum x coordinates
    x_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    x_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum y coordinates
    y_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    y_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    x_size = int(np.ceil(x_max - x_min))
    y_size = int(np.ceil(y_max - y_min))

    # Initialize an empty grid
    grid = np.zeros((x_size, y_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        x, y, alt, d_x, d_y, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(x - d_x - safety_distance - x_min, 0, x_size-1)),
                int(np.clip(x + d_x + safety_distance - x_min, 0, x_size-1)),
                int(np.clip(y - d_y - safety_distance - y_min, 0, y_size-1)),
                int(np.clip(y + d_y + safety_distance - y_min, 0, y_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
    return grid, int(x_min), int(y_min), int(x_max), int(y_max)


# Assume all actions cost the same.
# 只能上下左右走，不能走对角
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions

# given by teacher 
# delete visited set
# visited set is extra, we can use branch to know if this node is visited
def a_star(grid, h, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))

    # node:(g_cost,parent_node,parent_action)
    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in branch:                               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

# question 1
# ida*
# the key point is last iteration's smallest f value should be next iteration's boundry, and each 
# iteartion is a dfs
def iterative_astar(grid, h, start, goal):
    max_iteration = 100000
    first_boundry = h(start,goal)
    
    # node:(g_cost,parent_node,parent_action)
    path_dic = {}
    
    boundry = first_boundry
    path = []
    
    while(True):
        max_iteration -= 1
        next_boundry = ida_one_itreation(grid,h,path_dic,0,start,goal,boundry)
        
        # reach the goal
        if(next_boundry== -1):
            print('Find path')
            path_cost = path_dic[goal][0]
            path.append(goal)
            n = goal
            while path_dic[n][1] != start:
                path.append(path_dic[n][1])
                n = path_dic[n][1]
            path.append(start)
            return path,path_cost
        
        # reach the max iteration time
        if(max_iteration < 0):
            print('Failed to find a path!')
            return [],next_boundry
        
        # set next boundry before next loop
        boundry = next_boundry
                   
# ida*一次迭代(dfs)
def ida_one_itreation(grid,h,path_dic,g_cost,current_node,goal,boundry):
    current_f = g_cost + h(current_node,goal)
    if(current_f > boundry):
        return current_f
    if(current_node == goal):
        return -1
    
    # min value in this iteration
    Min = np.inf
    for action in valid_actions(grid, current_node):
        da = action.delta
        next_node = (current_node[0] + da[0], current_node[1] + da[1])
        if next_node not in path_dic: 
            next_g_cost = g_cost + action.cost
            
            # dfs: add new status
            path_dic[next_node] = (next_g_cost,current_node,action)
            
            # dfs:deeper loop
            next_f = ida_one_itreation(grid,h,path_dic,next_g_cost,next_node,goal,boundry)
            if(next_f == -1):
                return -1
            if(next_f < Min):
                Min = next_f
                
            # dfs:backtrace  
            path_dic.pop(next_node)
            
    return Min
              
            
#question 2 
# I dont think i completely know what this question is about, in this problem, four diretion action 's costs are all equal to 1,in this case,the ucs is actually a random bfs. and we dont have the information to design a new cost function for g. So i think maybe what means by cost function is h. but in this case, the usc is actually greedy search.
# I choose to implement the second one: set g = 0, and degisn 曼哈顿 切比雪夫 余弦距离 as cost function( see at last)
def ucs(grid, h, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))

    # node:(g_cost,parent_node,parent_action)
    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = 0
                queue_cost = h(next_node, goal)
                
                if next_node not in branch:                               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost
    
    
# question 3
# choose the visit sequence of fixed_points and invoke a* algorithm for 4 times
def fixed_a_star(grid, h, start, goal, fixed_points):
    # first we determine the visit sequence of the fixed_points
    # sequence of visit fixed_points
    fixed_path = []
    while (len(fixed_points) != 0):
        # min h from start to fixed points
        Min = np.inf
        Min_point = (0,0)
        for point in fixed_points:
            if (h(start,point) < Min):
                Min = h(start,point)
                Min_point = point
        # choose the nearst points(by h) in fixed_points 
        fixed_path.append(Min_point)
        fixed_points.remove(Min_point)
    
    # then we use the sequence to invoke a star algoriothm 4 times and join the path and cost
    return_path = []
    return_cost = 0
    fixed_path.append(goal)
    for point in fixed_path:
        path,cost = a_star(grid, h, start, point)
        return_path += path
        return_cost += cost
        start = point
    return return_path,return_cost

#question4 load data and construct the graph

#  record obstacle centres and create a Voronoi graph around those points
def create_grid_and_edges(data, drone_altitude, safety_distance):
    
    
#      """
#      Returns a grid representation of a 2D configuration space
#      along with Voronoi graph edges given obstacle data and the
#      drone's altitude.
#      """
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))
    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))
    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])

    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
             int(north - d_north - safety_distance - north_min_center),
             int(north + d_north + safety_distance - north_min_center),
             int(east - d_east - safety_distance - east_min_center),
             int(east + d_east + safety_distance - east_min_center),
             ]
            grid[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3]] = 1
    
            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])
    # location of obstacle centres
    graph = Voronoi(points)
    edges = []
    for edge in graph.ridge_vertices:
        point1 = graph.vertices[edge[0]]
        point2 = graph.vertices[edge[1]]

        cells = list(bresenham(int(point1[0]), int(point1[1]), int(point2
    [0]), int(point2[1])))
        infeasible = False
        for cell in cells:
            if np.amin(cell) < 0 or cell[0] >= grid.shape[0] or cell[1] >= grid.shape[1]:
                infeasible = True
                break
            if grid[cell[0], cell[1]] == 1:
                infeasible = True
                break
        if infeasible == False:
            point1 = (point1[0], point1[1])
            point2 = (point2[0], point2[1])
            edges.append((point1,point2))
           
    return grid, edges

# question4 bfs by graph
def bfs_graph(graph, start, goal):
    path= []
    queue = Queue()
    queue.put((0,start))
    # node:(depth,parent_node)
    path_dic = {}
    found = False

    while not queue.empty():
        current_node = queue.get()[1]
        if current_node == start:
            current_depth = 0.0
        else:              
            current_depth = path_dic[current_node][0]

        if current_node == goal:      
            print('Found a path.')
            found = True
            break
        else:
            #use networkx api to find a node's neighbor nodes
            for node in graph.neighbors(current_node):
                next_node = node
                if next_node not in path_dic:
                    queue.put((current_depth+1,next_node))
                    path_dic[next_node] = (current_depth+1,current_node)

    if found:
        n = goal
        path_cost = path_dic[n][0]
        path.append(goal)
        while path_dic[n][1] != start:
            path.append(path_dic[n][1])
            n = path_dic[n][1]
        path.append(start)
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')

    return path[::-1], path_cost

### 欧氏距离
def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

### 曼哈顿距离
def m_heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position),ord=1)

### 切比雪夫距离
def q_heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position),ord=np.inf)

### 余弦距离
def cos_heuristic(position, goal_position):
    vector1 = np.array(position)
    vector2 = np.array(goal_position)
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))




    

