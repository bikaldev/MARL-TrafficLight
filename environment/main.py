import sumolib
import random
from copy import deepcopy

from edge import Edge, EdgePosition
from node import Node
from simulation_env import SimulationEnv
from q_network import node_initialization, Q_NETWORK
from replay_buffer import ReplayBuffer

import tensorflow as tf
# from tensorflow import tf_agents

EPOCHS = 2
SIGMA = 0.99
EPSILON = 0.9
C1 = 20
EPISODE_LENGTH = 3500
BUFFER_SIZE = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

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
        if(q_val[:,0] > q_val[:,1]):
            return 0
        else:
            return 1
    else:
        return random.randint(0,1)
    
def get_greedy_action_value(q_val):
    if(q_val[:,0] > q_val[:,1]):
        return q_val[:,0]
    else:
        return q_val[:,1]

def get_joint_action(action_space, node_list, q_val_list):
    joint_action = dict()
    for node in action_space:
        action = get_e_greedy_action(q_val_list[node_list.index(node)])
        joint_action[node] = action
    
    return joint_action

def compute_loss(minibatch, node_list, q_net: Q_NETWORK):
    loss = 0.0
    state_list = []
    for transition in minibatch[0:-1]:
        inputs, joint_action, _, y_val_list = transition
        q_val_list, state_list = q_net.call(inputs, state_list)
        for node_idx in range(len(node_list)):
            node_id = node_list[node_idx]
            loss += (y_val_list[node_idx] - q_val_list[node_idx][0] if node_idx not in joint_action else q_val_list[node_idx][joint_action[node_id]]) ** 2
    
    return loss

def print_list(lt):
    for item in lt:
        print(item)

# Runs the simulation
def run(graph, edge_list, node_list, node_to_edge, node_neighbourhood):
    env = SimulationEnv()
    q_net = Q_NETWORK(graph,edge_list, node_list, node_to_edge, node_neighbourhood)
 
    tar_q_net = Q_NETWORK(graph, edge_list, node_list, node_to_edge, node_neighbourhood)
    env.set_graph(graph)
    avg_reward_list = []
    for epoch in range(EPOCHS):
        env.start()
        sim_step = 0
        time_step = 0
        state_list = []
        buffer = ReplayBuffer(BUFFER_SIZE)
        
        #Initialize environment variables
        inputs = ()
        joint_action = {}
        q_val_list = []
        reward_list = []
       
        print("Epoch: "+str(epoch))
        if(epoch >= 1):
            avg_reward = sum(avg_reward_list)/len(avg_reward_list)
            print("avg reward of epoch "+str(epoch) + " is "+str(avg_reward))

        while(sim_step < EPISODE_LENGTH):
            if(time_step % 25 != 0 or time_step == 0):
                action_space = env.get_action_space()
                if(action_space):
                    graph, reward_list = env.observe(node_list)
                    if(time_step != 0):
                        avg_reward_list.append(sum(reward_list))
                        print("Time Step: "+str(time_step) + " Sim step: "+str(sim_step) +" Total Reward : "+str(sum(reward_list)))
                        print_list(q_val_list)
                        # print("inputs: ")
                        # print_list(inputs)
                        # print("joint_action: ")
                        # print(joint_action)
                        # print("reward_list: ")
                        # print(reward_list)
                        #Store transition(inputs, joint_action, reward_list) on replay buffer
                        buffer.push((inputs, joint_action, reward_list))
                        

                    inputs = node_initialization(graph, edge_list, node_list)
                    q_val_list, state_list = q_net.call(inputs, state_list)
                    joint_action = get_joint_action(action_space, node_list, q_val_list)
                    env.apply_action(joint_action)
                    time_step += 1
                    
                
                sim_step += 1
                env.next_step()
            else:

        # env.stop()

                for c in range(C1):
                    with tf.GradientTape() as tape:
                        minibatch = buffer.sample(BATCH_SIZE)
                        state_list = []
                        for trans_idx in range(len(minibatch)-1):
                            transition_t = minibatch[trans_idx]
                            transition_t_pl_1 = minibatch[trans_idx+1]
                            q_tar_val_list,state_list =  tar_q_net.call(transition_t_pl_1[0], state_list)
                            y_val_list = list()
                            for node_idx in range(len(node_list)):
                                if(node_list[node_idx] in transition_t_pl_1[1]):
                                    y_val = transition_t[2][node_idx] + EPSILON * get_greedy_action_value(q_tar_val_list[node_idx])
                                else:
                                    y_val = transition_t[2][node_idx] + EPSILON * q_tar_val_list[node_idx][0]
                                y_val_list.append(y_val)
                            
                            minibatch[trans_idx] = minibatch[trans_idx] + (y_val_list,)
                        
                        loss = compute_loss(minibatch, node_list, q_net)
                    
                    grads = tape.gradient(loss, q_net.trainable_variables)
                    # print(q_net.trainable_variables)
                    # print(grads)
                    # return None
                    print(len(grads), len(q_net.trainable_variables))
                    # print(grads)
                    OPTIMIZER.apply_gradients(zip(grads, q_net.trainable_variables))
                    # return None
        
                # print(q_net.trainable_variables)
                q_net.save_weights('q_net.weights.h5')
                tar_q_net = deepcopy(q_net)
                tar_q_net.save_weights('tar_q_net.weights.h5')

                time_step += 1
        
        env.stop()




    

if __name__ == "__main__":
    graph, edge_list, node_list, node_to_edge, node_neighbourhood = init_network('../config_3/road_network.net.xml')
    print(node_list)
    print(edge_list)
    run(graph, edge_list, node_list, node_to_edge, node_neighbourhood)