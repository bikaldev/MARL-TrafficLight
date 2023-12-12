'''
    This represents a junction in the road network. A node may or may not have a traffic light depending upon the 
    is_control flag.
'''

class Node:
    def __init__(self, id, is_control):
        assert(type(id) == str and type(is_control) == bool)
        self.id = id
        self.is_control = is_control
    
    def set_phase(self, phase):
        self.phase = phase

        