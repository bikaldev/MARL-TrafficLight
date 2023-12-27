import tensorflow as tf
import random

from q_network import Q_NETWORK, node_initialization
from main import init_network

graph, edge_list, node_list, node_to_edge, node_neighbourhood= init_network('../sumo/road_network.net.xml')

# Generate test input tuple
edge_vec_list = list()
for idx in range(len(edge_list)):
    edge_vec_list.append(tf.random.uniform((1,6), minval=0, maxval=1))

phase_list = list()
for idx in range(len(node_list)):
    phase_list.append(random.choice([0,2]))
# print(phase_list)

q_net = Q_NETWORK(graph, edge_list, node_list, node_to_edge, node_neighbourhood)

def get_loss(y_vec_list):
    loss = 0.0
    for y_vec in y_vec_list:
        loss += tf.reduce_sum(y_vec)
    return loss

with tf.GradientTape() as tape:
    y_val_list, net_list = q_net.call((edge_vec_list, phase_list), [])
    print(y_val_list)
    loss = get_loss(y_val_list)
    print(type(loss))

grads = tape.gradient(loss, q_net.trainable_variables)
# print(grads)