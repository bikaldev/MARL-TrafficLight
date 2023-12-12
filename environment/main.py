import sumolib
import random

from edge import Edge, EdgePosition
from node import Node
from simulation_env import SimulationEnv
from q_network import node_initialization, Q_NETWORK

EPOCHS = 10
SIGMA = 0.99
EPSILON = 1
C1 = 300
EPISODE_LENGTH = 5000

# This function creates a logical graph from the provided .net.xml file of a road network
# It returns a tuple (nodes, edges) where nodes is a list of objects of type nodes 
# and edges is list of objects of type edges

def init_network(filename):
    nodes = {}
    edges = {}
    edge_list = list()
    node_list = list()
    node_to_edge = dict()
    node_neighbourhood = dict()
    network = sumolib.net.readNet(filename)
    for item in network.getEdges():
        send_node = item.getFromNode()
        recv_node = item.getToNode()
        edge = Edge(item.getID(), send_node.getID(), recv_node.getID())
        if(recv_node.getType() == "traffic_light"):
            edge_list.append(item.getID())
            if(recv_node.getID() not in node_list):
                node_list.append(recv_node.getID())
            if(recv_node.getID() not in node_to_edge):
                node_to_edge[recv_node.getID()] = [item.getID()]
            else:
                node_to_edge[recv_node.getID()].append(item.getID())
        
        if(send_node.getType() == "traffic_light" and recv_node.getType() == "traffic_light"):
            if(recv_node.getID() not in node_neighbourhood):
                node_neighbourhood[recv_node.getID()] = [send_node.getID()]
            else:
                node_neighbourhood[recv_node.getID()].append(send_node.getID())
        

        (send_x, send_y) = send_node.getCoord()
        (recv_x, recv_y) = recv_node.getCoord()
        if(abs(send_x - recv_x) < 0.000001):
            if(send_y - recv_y > 0):
                edge.set_position(EdgePosition.TOP)
            else:
                edge.set_position(EdgePosition.BOTTOM)
        elif(abs(send_y - recv_y)<0.000001):
            if(send_x - recv_x > 0):
                edge.set_position(EdgePosition.RIGHT)
            else:
                edge.set_position(EdgePosition.LEFT)
        else:
            slope = (send_y - recv_y)/(send_x- recv_x)
            if(abs(slope) < 1):
                if(send_x - recv_x > 0):
                    edge.set_position(EdgePosition.RIGHT)
                else:
                    edge.set_position(EdgePosition.LEFT)
            else:
                if(send_y - recv_y > 0):
                    edge.set_position(EdgePosition.TOP)
                else:
                    edge.set_position(EdgePosition.BOTTOM)
                
        for lane in item.getLanes():
            edge.set_lane(lane.getID())
        edges[item.getID()] = edge
    
    
    for item in network.getNodes():
        nodes[item.getID()] = Node(item.getID(), item.getType() == "traffic_light")


    return ((nodes, edges), edge_list, node_list, node_to_edge, node_neighbourhood)

def get_e_greedy_action(q_val, e = EPSILON):
    if(random.random() < e):
        if(q_val[0:0] > q_val[0:1]):
            return 0
        else:
            return 1
    else:
        return random.randint(0,1)

def get_joint_action(action_space, node_list, q_val_list):
    joint_action = dict()
    for node in action_space:
        action = get_e_greedy_action(q_val_list[node_list.index(node)])
        joint_action[node] = action
    
    return joint_action
    

# Runs the simulation
def run(graph, edge_list, node_list, node_to_edge, node_neighbourhood):
    env = SimulationEnv()
    q_net = Q_NETWORK(graph,edge_list, node_list, node_to_edge, node_neighbourhood)
    tar_q_net = Q_NETWORK(graph, edge_list, node_list, node_to_edge, node_neighbourhood)
    env.set_graph(graph)
    for epoch in range(EPOCHS):
        env.start()
        sim_step = 0
        time_step = 0
        state_list = []
        while(sim_step < EPISODE_LENGTH):
            action_space = env.get_action_space()
            if(env.get_action_space()):
                graph, reward_list = env.observe()
                if(time_step != 0):
                    #Store transition(inputs, q_val_list, joint_action, reward_list) on replay buffer
                    pass

                inputs = node_initialization(graph, edge_list, node_list)
                q_val_list, state_list = q_net.call(inputs, state_list)
                joint_action = get_joint_action(action_space, node_list, q_val_list)
                env.apply_action(joint_action)
        
        env.stop()

        for c in range(C1):
            # sample a minibatch of transition in continous interval
            # calc q_val with tar_q_net
            # calc y for every agent
            # gradient descent on loss
            pass



    


if __name__ == "__main__":
    graph, edge_list, node_list, node_to_edge, node_neighbourhood = init_network('../sumo/road_network.net.xml')
    run(graph, edge_list, node_list, node_to_edge, node_neighbourhood)