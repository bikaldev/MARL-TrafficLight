import tensorflow as tf
import numpy as np

NUM_OF_LANES_PER_EDGE = 2
EDGE_INPUT_SHAPE = (3*NUM_OF_LANES_PER_EDGE,1)
EDGE_OUTPUT_SHAPE = (EDGE_INPUT_SHAPE[0] * 4,1)
NODE_INPUT_SHAPE = (EDGE_OUTPUT_SHAPE[0] + 4, 1)
NODE_OUTPUT_SHAPE = (NODE_INPUT_SHAPE[0],1)
 # for each edge
# NORMALIZATION = [
#                 keras.layers.Normalization(mean=28.20, variance=958.17),
#                 keras.layers.Normalization(mean=4.51, variance=15.56),
#                 keras.layers.Normalization(mean=3.87, variance=41.65)
#                 ]

def print_list(lt):
    for item in lt:
        print(item)


def node_initialization(graph, edge_list, node_list):
    edges = graph[1]
    edge_vec_list = list()
    phase_vec = list()
    for edge in edge_list:
        edge_vector = np.zeros(EDGE_OUTPUT_SHAPE)
        lanes = edges[edge].lanes
        params = ['q_len', 'n_of_vehs', 'avg_speed']
        idx = (edges[edge].position.value - 1) * EDGE_INPUT_SHAPE[0]
        for param in params:
            for lane in lanes:
                edge_vector[idx] = edges[edge].lane_data[lane][param]
                idx += 1
        # edge_vector[edges[edge].position.value + EDGE_INPUT_SHAPE[0] - 5] = 1
        edge_vec_list.append(tf.convert_to_tensor(edge_vector.T))
        # print(edge_vec_list[0].shape)

    for node in node_list:
        phase_vec.append(graph[0][node].phase)
    
    return (edge_vec_list, phase_vec)

class Q_NETWORK(tf.keras.Model):
    def __init__(self, graph:tuple ,edge_list: list, node_list: list, node_to_edge: dict, node_neighbourhood: dict):
        super().__init__()
        self.graph = graph
        self.edge_list = edge_list
        self.node_list = node_list
        self.node_to_edge = node_to_edge
        self.node_neighbourhood = node_neighbourhood

        #Edge and Node observation data Encoders
        self.edge_encode_l1 = tf.keras.layers.Dense(EDGE_OUTPUT_SHAPE[0], activation='relu')
        self.edge_encode_l2 =  tf.keras.layers.Dense(EDGE_OUTPUT_SHAPE[0], activation='relu')

        self.node_encode = tf.keras.layers.Dense(NODE_OUTPUT_SHAPE[0], activation='relu')

        self.lstm_layer_list = []
        for node in node_list:
            self.lstm_layer_list.append(tf.keras.layers.LSTMCell(NODE_OUTPUT_SHAPE[0]))
                                        

        self.attention_layer = tf.keras.layers.Dense(1, use_bias=False, activation=None)
        self.attention_act = tf.keras.layers.LeakyReLU()

        # self.q_val_1 = tf.keras.layers.Dense(30, activation='relu')
        self.q_val_1 = tf.keras.layers.Dense(15, activation='relu')
        # self.q_val_3 = tf.keras.layers.Dense(5, activation='relu')
        self.q_val_2 = tf.keras.layers.Dense(2, activation='relu')

    
    def node_init(self, inputs: tuple[list[tf.Tensor], list[int]]):
        edge_vec_list = inputs[0]
        phase_vec = inputs[1]
        node_vec_list = list()
        idx = 0
        for node in self.node_list:
            # node_vector = tf.zeros(NODE_INPUT_SHAPE)
            temp = tf.zeros((1,EDGE_OUTPUT_SHAPE[0]))

            phase = np.zeros((1,4))
            phase[0,phase_vec[idx]] = 1.0
            phase = tf.convert_to_tensor(phase, dtype=tf.float32)

            for edge in self.node_to_edge[node]:
                edge_idx = self.edge_list.index(edge)
                # temp[:,0:EDGE_OUTPUT_SHAPE[0]] += self.edge_encode_l2(self.edge_encode_l1(edge_vec_list[edge_idx]))
                edge_vec = self.edge_encode_l2(self.edge_encode_l1(edge_vec_list[edge_idx]))
                temp = tf.math.add(temp, edge_vec)
            node_vector = tf.concat([temp,phase],axis = 1)
            tf.cast(node_vector, tf.double)
            node_vec_list.append(self.node_encode(node_vector))
            idx += 1

        return node_vec_list


    def lstm(self, node_vec_list, state_list=[]):
        empty_flag = len(state_list) == 0
        idx = 0
        hid_node_vec_list = list()
        if(not empty_flag):
            for node_vec in node_vec_list:
                    hid_node_vec, node_state = self.lstm_layer_list[idx](node_vec, states=state_list[idx])
                    state_list[idx] = node_state
                    hid_node_vec_list.append(hid_node_vec)
                    idx += 1
        else:
            temp = tf.convert_to_tensor(np.zeros(NODE_OUTPUT_SHAPE).T, dtype=tf.float32)
            for node_vec in node_vec_list:
                hid_node_vec, node_state = self.lstm_layer_list[idx](node_vec, states= [temp, temp])
                state_list.append(node_state)
                hid_node_vec_list.append(hid_node_vec)
                idx += 1
        
        return hid_node_vec_list, state_list
        

    def node_attention_layer(self, hid_node_vec_list):
        out_node_vec_list = list()
        for node_i in self.node_list:
            idx_i = self.node_list.index(node_i)
            agg_attention = tf.zeros((1, NODE_OUTPUT_SHAPE[0]))
            sum_softmax = 0.0
            for node_j in self.node_neighbourhood[node_i]:
                idx_j = self.node_list.index(node_j)
                temp_val = self.attention_layer(tf.concat([hid_node_vec_list[idx_i], hid_node_vec_list[idx_j]], axis = 1))
                temp_val = self.attention_act(temp_val)
                temp_val = tf.exp(temp_val)
                sum_softmax += temp_val
                agg_attention += temp_val * hid_node_vec_list[idx_j]
            
            out_node_vec_list.append(tf.concat([hid_node_vec_list[idx_i], agg_attention/sum_softmax], axis = 1))

        return out_node_vec_list

    def output_layer(self, node_vec_list, out_node_vec_list):
        q_val_list = list()
        for idx in range(len(node_vec_list)):
            # temp_vec = tf.concat([node_vec_list[idx], out_node_vec_list[idx]], axis = 1)
            temp_vec = out_node_vec_list[idx]
            # print("temp_vec in output_layer")
            # print_list(temp_vec)
            q_val = self.q_val_2(self.q_val_1(temp_vec))
            q_val_list.append(q_val)
        
        return q_val_list

            


    def call(self, inputs: tuple[list[tf.Tensor], list[int]], state_list: list[tf.Tensor]):
        node_vec_list = self.node_init(inputs)
        # print("node_vec_list:")
        # print_list(node_vec_list)
        hid_node_vec_list, state_list = self.lstm(node_vec_list, state_list)
        # print("hid_node_vec_list:")
        # print_list(hid_node_vec_list)
        out_node_vec_list = self.node_attention_layer(hid_node_vec_list)
        # print("out_node_vec_list:")
        # print_list(out_node_vec_list)
        q_net_list = self.output_layer(node_vec_list, out_node_vec_list)

        return (q_net_list, state_list)



