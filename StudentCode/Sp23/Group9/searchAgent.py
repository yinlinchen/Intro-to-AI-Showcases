

import time,os
import traceback
import sys
import random
import random, util, math
import AStarAlgorithm
import snakeGameData as sg_data

frame_size_x = 720
frame_size_y = 480

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from snakeGame.py) and
        must return an action from Directions.{'UP','DOWN','LEFT','RIGHT'}
        """


class SearchAgent(Agent):
    
    """
    Agent just gets an action from an algorithm and returns the best move
    Using Astar and Qlearning for the search agent
    """
    
    
    def __init__(self, algorithm):
        self.trainer = algorithm
        self.actionFn = lambda state: self.getLegalActions(state)
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        
    def get_state(self, game):
        #position of the snake, the food and the snake's body and score too
        self.state = "NotImplemented"
        return (sg_data.snake_pos, sg_data.snake_body, sg_data.food_pos, sg_data.score)
        
    
    def getAction(self, state, algorithm):
        nextMove = ['UP','DOWN','LEFT','RIGHT']
        #for move in nextMove:
        #    if SearchAgent.gameOverState(self, snake_pos = sg_data.snake_pos, snake_body = sg_data.snake_body) == False and SearchAgent.illegal_move(self, move) == False:
        #        return move
        
        res = AStarAlgorithm.aStarSearch(self.get_state)

        if(len(res) > 0):
            return res[0]
        
    
        #if SearchAgent.gameOverState(algorithm.snake_pos, algorithm.snake_body) == False and SearchAgent.illegal_move(algorithm.move) == False:
        # if SearchAgent.gameOverState(sg_data.snake_pos, sg_data.snake_body) == False and SearchAgent.illegal_move(algorithm.move) == False:
        #         return algorithm.move
        return nextMove[random.randint(0,3)]

    def gameOverState(self, snake_pos, snake_body):
    #Touching the walls of the frame
        if snake_pos[0] < 0 or snake_pos[0] > frame_size_x-10:
            return True
        if snake_pos[1] < 0 or snake_pos[1] > frame_size_y-10:
           return True
    # Touching the snake body
        for block in snake_body[1:]:
            if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                return True
        
        return False
    
    def illegal_move(self, nextMove):
        if nextMove == 'UP' and sg_data.direction == 'DOWN':
            return True
        if nextMove == 'DOWN' and sg_data.direction == 'UP':
            return True
        if nextMove == 'LEFT' and sg_data.direction == 'RIGHT':
            return True
        if nextMove == 'RIGHT' and sg_data.direction == 'LEFT':
            return True
        
        return False
    
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in ['DOWN', 'UP', 'LEFT', 'RIGHT']:
            x,y = state.snake_pos
            if action == 'DOWN':
                dx, dy = 0, -1
            if action == 'UP':
                dx, dy = 0, 1
            if action == 'LEFT':
                dx, dy = -1, -0
            if action == 'RIGHT':
                dx, dy = 1, 0
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.illegal_move(action):
                nextState = (nextx, nexty)
                cost = 1
                successors.append((nextState, action, cost))
        return successors
    
  