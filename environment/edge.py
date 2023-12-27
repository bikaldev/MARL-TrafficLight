from enum import Enum
''' 
    This class represents a unidirectional collection of lanes in the road network.
    Each edge contains it observation information received from lane detectors. 
    If edge id is 'E1' then the detector id is 'E1_0' and 'E1_1' so on depending on the number of detectors
 '''
class Edge:
    def __init__(self, id, send_node, rec_node):
        assert(type(id) == str and type(send_node) == str and type(rec_node) == str)
        self.id = id
        self.send_node = send_node
        self.rec_node = rec_node
        self.lanes = []
        self.lane_data = {}
    
    def set_lane_data(self, laneID, obs_data):
            assert(type(laneID) == str and type(obs_data) == dict)
            self.lane_data[laneID] = obs_data
    
    def set_lane(self, laneID):
        self.lanes.append(laneID)
    
    def set_position(self, pos):
        assert(type(pos) == EdgePosition)
        self.position = pos # has to be of type EdgePosition
    
    def set_obs_vector(self, vec):
         self.vector = vec


class EdgePosition(Enum):
    # defines position wrto the receiving node.
    TOP = 1
    BOTTOM = 2
    LEFT = 3
    RIGHT = 4
        