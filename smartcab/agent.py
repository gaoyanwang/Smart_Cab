import random
import os
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QTable(object):
    def __init__(self):
        self.table = dict()

    def get(self, state, action):
        key = (state, action)
        return self.table.get(key, None)

    def set(self, state, action, q):
        key = (state, action)
        self.table[key] = q

class QLearn(Agent):
    def __init__(self, epsilon=.05, alpha=.1, gamma=.9):
        self.Q = QTable()       # Q(s, a)
        self.epsilon = epsilon  # probability of doing random move
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor 
        
    def action(self, state):
        if random.random() < self.epsilon: # exploration action ==> random move
            action = random.choice(Environment.valid_actions)
        else: # base the decision on q
            q = [self.Q.get(state, a) for a in Environment.valid_actions]
            maxQ = max(q)
            # we have identical max q from Q, which one to choose?
            if q.count(maxQ) > 1: 
                # pick an action randomly from all max
                best_actions = [i for i in range(len(Environment.valid_actions)) if q[i] == maxQ]                       
                action_idx = random.choice(best_actions)

            else:
                action_idx = q.index(maxQ)
            action = Environment.valid_actions[action_idx]
        return action
            
    def learn(self, state, action, next_state, reward):
        q = [self.Q.get(next_state, a) for a in Environment.valid_actions]
        maxQ = max(q)         
        if maxQ is None:
            maxQ = 0.0

        q = self.Q.get(state, action)
        if q is None:
           q = 0.0
        else:
           q = q + self.alpha * (reward + self.gamma * maxQ - q);
        self.Q.set(state, action, q)

class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world by using Q-Learning"""

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self) # simple route planner to get next_waypoint
        self.learner = QLearn(epsilon=1, alpha=0.3, gamma=0.6)        

    def reset(self, destination=None):
        self.planner.route_to(destination)      

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self) 
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # TODO: Select action according to your policy
        action = self.learner.action(self.state)
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        next_inputs = self.env.sense(self) 
        self.future_next_way_out = self.planner.next_waypoint()
        next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], self.future_next_way_out)
        #next_state = (next_inputs[0], next_inputs[1], next_inputs[3], self.planner.next_waypoint())
        self.learner.learn(self.state, action, next_state, reward)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
