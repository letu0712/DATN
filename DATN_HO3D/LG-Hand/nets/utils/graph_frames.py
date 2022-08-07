import numpy as np
import numpy as np
import random
import torch


import torch
import numpy as np 
import random

class Graph():
    """ The Graph to model the skeletons of human body/hand

    Args:
        strategy (string): must be one of the follow candidates
        - spatial: Clustered Configuration


        layout (string): must be one of the follow candidates

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout,          #fphab_gt
                 strategy,
                 pad=1,
                 max_hop=1,
                 dilation=1):

        self.max_hop = max_hop
        self.dilation = dilation
        self.seqlen = 2*pad+1       #Number of frames T
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)

        # get distance of each node to center
        self.dist_center = self.get_distance_to_center(layout)
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout):
        """

        :return: get the distance of each node to center
        
        Node center is wrist
        """
        dist_center = np.zeros(self.num_node)
        if layout == 'ho3d':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+6] = [0, 1, 1, 1, 1, 1]
                dist_center[index_start+6 : index_start+11] = [2, 2, 2, 2, 2]
                dist_center[index_start+11 : index_start+16] = [3, 3, 3, 3, 3]
                dist_center[index_start+16 : index_start+21] = [4, 4, 4, 4, 4]
        #dist_center: [0. 1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 3. 3. 3. 3. 3. 4. 4. 4. 4. 4. 0. 1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 3. 3. 3. 3. 3. 4. 4. 4. 4. 4. 0. 1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 3. 3. 3. 3. 3. 4. 4. 4. 4. 4.]
        return dist_center


    def __str__(self):
        return self.A

    def graph_link_between_frames(self,base):
        """
        calculate graph link between frames given base nodes and seq_ind
        :param base:
        :return:
        """
        return [((front - 1) + i*self.num_node_each, (back - 1)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]


    def basic_layout(self,neighbour_base):
        """
        for generating basic layout time link selflink etc.
        neighbour_base: neighbour link per frame

        :return: link each node with itself
        """
        self.num_node = self.num_node_each * self.seqlen
        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                     for j in range(self.num_node_each)]
        #time_link: [(0, 21), (1, 22), (2, 23), (3, 24), (4, 25), (5, 26), (6, 27), (7, 28), (8, 29), (9, 30), (10, 31), (11, 32), (12, 33), (13, 34), (14, 35), (15, 36), (16, 37), (17, 38), (18, 39), (19, 40), (20, 41), (21, 42), (22, 43), (23, 44), (24, 45), (25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54), (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62)]

        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1)
                                  for j in range(self.num_node_each)]
        #self.time_link_forward: [(0, 21), (1, 22), (2, 23), (3, 24), (4, 25), (5, 26), (6, 27), (7, 28), (8, 29), (9, 30), (10, 31), (11, 32), (12, 33), (13, 34), (14, 35), (15, 36), (16, 37), (17, 38), (18, 39), (19, 40), (20, 41), (21, 42), (22, 43), (23, 44), (24, 45), (25, 46), (26, 47), (27, 48), (28, 49), (29, 50), (30, 51), (31, 52), (32, 53), (33, 54), (34, 55), (35, 56), (36, 57), (37, 58), (38, 59), (39, 60), (40, 61), (41, 62)]
        
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]
        #self.time_link_back: [(21, 0), (22, 1), (23, 2), (24, 3), (25, 4), (26, 5), (27, 6), (28, 7), (29, 8), (30, 9), (31, 10), (32, 11), (33, 12), (34, 13), (35, 14), (36, 15), (37, 16), (38, 17), (39, 18), (40, 19), (41, 20), (42, 21), (43, 22), (44, 23), (45, 24), (46, 25), (47, 26), (48, 27), (49, 28), (50, 29), (51, 30), (52, 31), (53, 32), (54, 33), (55, 34), (56, 35), (57, 36), (58, 37), (59, 38), (60, 39), (61, 40), (62, 41)]
        
        self_link = [(i, i) for i in range(self.num_node)]
        #self_link: [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25), (26, 26), (27, 27), (28, 28), (29, 29), (30, 30), (31, 31), (32, 32), (33, 33), (34, 34), (35, 35), (36, 36), (37, 37), (38, 38), (39, 39), (40, 40), (41, 41), (42, 42), (43, 43), (44, 44), (45, 45), (46, 46), (47, 47), (48, 48), (49, 49), (50, 50), (51, 51), (52, 52), (53, 53), (54, 54), (55, 55), (56, 56), (57, 57), (58, 58), (59, 59), (60, 60), (61, 61), (62, 62)]
        
        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        return self_link, time_link

    def get_edge(self, layout):   #return all connect pair of joints
        """
        get edge link of the graph

        cb: center bone
        """
        if layout == 'ho3d':
            self.num_node_each = 21


            neighbour_base = [(1, 2), (7, 2), (12, 7), (17, 12), (3, 1), (8, 3),
                              (13, 8), (18, 13), (4, 1), (9, 4), (14, 9),
                              (19, 14), (5, 1), (10, 5), (15, 10), (20, 15),
                              (6,1), (11,6), (16,11), (21,16)
                              ]

            self_link, time_link = self.basic_layout(neighbour_base)

            self.T = [6,11,16]
            self.I = [7,12,17]
            self.M = [8,13,18]
            self.R = [9,14,19]
            self.P = [10,15,20]
            self.goc = [0,1,2,3,4,5]
            
            self.part = [self.goc,self.T,self.I,self.M,self.R,self.P]
            # print("self.part: ", self.part)


            self.edge = self_link + self.neighbour_link_all +  time_link
            # print("self.edge: ",self.edge)
            # center node of body/hand
            self.center = 0

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):

        valid_hop = range(0, self.max_hop + 1, self.dilation)   #range(0,2)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        # print("Adjacency matrix: ",adjacency.shape)
        normalize_adjacency = normalize_digraph(adjacency)



        if strategy == 'spatial':
            A = []
            for hop in valid_hop:   
                a_root = np.zeros((self.num_node, self.num_node))    #num_node = num_each_node * seq = 3*21 if seq = 3
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_forward = np.zeros((self.num_node, self.num_node))
                a_back = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                          if (j,i) in self.time_link_forward:
                              a_forward[j, i] = normalize_adjacency[j, i]

                          elif (j,i) in self.time_link_back:
                              a_back[j, i] = normalize_adjacency[j, i]

                          elif self.dist_center[j] == self.dist_center[i]:
                              a_root[j, i] = normalize_adjacency[j, i]

                          elif self.dist_center[j] > self.dist_center[i]:
                              a_close[j, i] = normalize_adjacency[j, i]

                          else:
                              a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)    #Matrix of root node
                
                else:
                    A.append(a_close)
                    A.append(a_further)
                    
                    if self.seqlen > 1:
                        A.append(a_forward)
                        A.append(a_back)         
            A = np.stack(A)
            self.A = A
            # print("self.A.shape: ",self.A.shape)
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]# GET [I,A]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1): # preserve A(i,j) = 1 while A(i,i) = 0
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)       #Calculate sum of each Column
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)          #Diagonal matrix
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

import torch
graph = Graph(layout="ho3d", strategy="spatial", pad=1)

# print(graph.get_distance_to_center(layout="fphab_gt"))
#A = graph.get_adjacency(strategy="spatial")

A = torch.tensor(graph.A, dtype=torch.float32)
#print(A.shape)



# print(graph.get_edge(layout="fphab_gt"))

# A = np.array([[0,1,0,1],
#               [1,0,1,1],
#               [0,1,0,1],
#               [1,1,1,0]], dtype=np.float64)

# Dl = np.sum(A, 0)
# num_node = A.shape[0]
# Dn = np.zeros((num_node, num_node))

# for i in range(num_node):
#     if Dl[i] > 0:
#         Dn[i,i] = Dl[i]**(-1)
# print(Dl)
# print(Dn)
# AD = np.dot(A, Dn)
# print(AD)



# for i in range(num_node):
#     if Dl[i] > 0:
#         Dn[i,i] = Dl[i]**(-0.5)

# DAD = np.dot(np.dot(Dn,A), Dn)
# print(DAD)

