import os
import sys
import optparse
from sumolib import checkBinary  # Checks for the binary in environ vars
import traci
import math


class SimulationEnv:
    def __init__(self):
        # we need to import some python modules from the $SUMO_HOME/tools directory
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

    def __get_options(self):
        opt_parser = optparse.OptionParser()
        opt_parser.add_option("--nogui", action="store_true",
                            default=False, help="run the commandline version of sumo")
        options, args = opt_parser.parse_args()
        return options

    def start(self):
        # options = self.__get_options()

        # # check binary
        # if options.nogui:
        #     sumoBinary = checkBinary('sumo')
        # else:
        #     sumoBinary = checkBinary('sumo-gui')
        sumoBinary = checkBinary('sumo')
         # traci starts sumo as a subprocess and then this script connects and runs
        traci.start([sumoBinary, "-c", "../sumo/road_network_map_config.sumocfg",
                             "--tripinfo-output", "../sumo/tripinfo.xml"])
        
        traci.simulationStep()
        
# The graph value is expected to be tuple containing (nodes, edges).
    def set_graph(self, graph):
        assert(type(graph) == tuple and len(graph) == 2)
        self.graph = graph

# This function should apply the given action set in the simulation environment.
# Action set is a dictionary with each control node/agent as key and the value as 0 or 1 indication whether
# to switch to the next phase(1) or stay in the same phase(0).
    def apply_action(self, action_set):
        for node in action_set:
            if(action_set[node] == 1):
                traci.trafficlight.setPhase(self.graph[0][node].id, (self.graph[0][node].phase + 1) % 4)
            else:
                traci.trafficlight.setPhase(self.graph[0][node].id, (self.graph[0][node].phase))

# Moves the simulation forward    
    def next_step(self):
        traci.simulationStep()

# Stop the simulation
    def stop(self):
        traci.close()
    
# Checks if there is atleast one agent that has an action space of more that 1 i.e. at least one agent has to be able to change phase.
# If present, it will return the action_set as a dictionary with the key being node id and the value a tuple of actions.
    def get_action_space(self):
        action_set = dict()
        for node in self.graph[0] :
            if self.graph[0][node].is_control:
                node_obj = self.graph[0][node]
                
                # If a traffic light is in yellow phase it can't be changed, if it is in other phases it can be changed
                # But since the least timestep for a phase is defined as 20 timestep, thus only at the end of it
                # can the phases be changed.
                if(traci.trafficlight.getPhase(node) % 2 == 0 and traci.trafficlight.getNextSwitch(node) - traci.simulation.getTime() == 0):
                    # It has action available
                    action_set[node_obj.id] = (0,1)

        if(len(action_set) == 0):
            return None
        else:
            return action_set


# This function observes the data at each node and edge in the last interval, and also calculates reward for the previous action step
    def observe(self, node_list):

        # Store data for every edge in the graph
        for edge in self.graph[1]:
            if(self.graph[0][self.graph[1][edge].rec_node].is_control):
                for lane in self.graph[1][edge].lanes:
                    queue_length =  traci.lanearea.getLastIntervalMaxJamLengthInMeters(lane)
                    num_of_vehicles = traci.lanearea.getLastIntervalVehicleNumber(lane)
                    avg_speed = traci.lanearea.getLastIntervalMeanSpeed(lane)
                    self.graph[1][edge].set_lane_data(lane, {
                        'q_len': (queue_length - 28.20)/math.sqrt(958.17),
                        'n_of_vehs': (num_of_vehicles - 4.51)/math.sqrt(15.56),
                        'avg_speed': (avg_speed-3.87)/math.sqrt(41.65)
                    })
            
        # Store the traffic light phaseID for every traffic light.
        for node in self.graph[0]:
            if(self.graph[0][node].is_control):
                current_phase = traci.trafficlight.getPhase(self.graph[0][node].id)
                self.graph[0][node].set_phase(current_phase)

        # Calculate reward
        reward_list = list()
        for node in node_list:
            reward = 0.0
            for edge in self.graph[1]:
                if(self.graph[1][edge].rec_node == node):
                    reward -= sum(self.graph[1][edge].lane_data[lane]['q_len'] for lane in self.graph[1][edge].lanes)
            reward_list.append(reward)

        return (self.graph,reward_list) # for previous interval
