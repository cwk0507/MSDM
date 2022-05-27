from math import sqrt, log
import time
from random import choice
from check_for_done_2 import check_for_done_move
from improvement import find_children_subset, find_children_priority

class Para:
    exploration = 0.5 # exploration coefficient for calculating UCB
    resources = 10  # time limit for each search
    INF = float('inf')
    
class GameNode():
    def __init__(self,mat,parent,player,move):
        self.state = mat
        self.player = player
        self.parent = parent
        self.children = {}
        self.move = move # position of last stone placed
        self.N = 0 # visited count
        self.Q = 0 # reward
    
    def ucb(self,explore=Para.exploration):
        if self.N == 0:
            return Para.INF
        else:
            return self.Q / self.N + explore * sqrt(log(self.parent.N)/self.N)

    def is_terminal(self):
        done, win = check_for_done_move(self.state,self.move)
        return True if done else False
    
    def expand(self, movenumber):
        if self.is_terminal():
            return 
        state = self.state
        children_subset = find_children_priority(state,find_children_subset(state, movenumber),-self.player)
        for idx, pairs in enumerate(children_subset):
            temp_state = state.copy()
            temp_player = -self.player
            temp_state[pairs[0]][pairs[1]] = temp_player
            self.children[idx] = GameNode(temp_state,self,temp_player,pairs)
        return 

def monte_carlo_tree_search(root, movenumber):
    start_time = time.time()
    while time.time() - start_time < Para.resources:
        leaf = traverse(root, movenumber) # leaf = unvisited node
        simulation_result = rollout(leaf)
        backpropagate(leaf, simulation_result)
    return best_child(root)

def traverse(node, movenumber):
    if node.N == 0:
        node.expand(movenumber)
    while fully_expanded(node) and not check_for_done_move(node.state,node.move)[0]:
        node = best_ucb(node)
        if not bool(node.children):
            node.expand(movenumber)
    if check_for_done_move(node.state,node.move)[0]: # check if children is empty
        return node
    return choice([node.children[keys] for keys in node.children if node.children[keys].N==0])
                         
def fully_expanded(node):
    if not bool(node.children):
        return False
    return not 0 in [node.children[keys].N for keys in node.children]
                        
def rollout(node):
    if node.is_terminal():
        _ , win = check_for_done_move(node.state,node.move)
    else:
        win = rollout_policy(node)
    return win

def rollout_policy(node):
    active_state = node.state.copy()
    vacant_pos = [(i,j) for i in range(active_state.shape[0]) for j in range(active_state.shape[0]) if active_state[i][j]==0]
    done, win = check_for_done_move(active_state,node.move)
    active_player = node.player
    while not done:
        active_player = -active_player
        selected_pos = choice(vacant_pos)
        vacant_pos.remove(selected_pos) 
        active_state[selected_pos[0]][selected_pos[1]] = active_player
        done, win = check_for_done_move(active_state,selected_pos)
    return win

def backpropagate(node, result):
    node.N += 1
    if result == node.player:
        node.Q += 1
    elif result == 0:
        node.Q += 0.5
    if node.parent is None:
        return 
    backpropagate(node.parent, result)

def best_ucb(node):
    max_ucb = max([node.children[key].ucb() for key in node.children])
    max_ucb_children = [node.children[key] for key in node.children if node.children[key].ucb()==max_ucb]
    return choice(max_ucb_children)

def best_child(node):
    max_N = max([node.children[key].N for key in node.children])
    max_N_children = [node.children[key] for key in node.children if node.children[key].N==max_N]
    return choice(max_N_children)
