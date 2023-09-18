# -*- coding: utf-8 -*-
# A* algorithm
"""
Created on Thu Apr 13 15:22:23 2023

@author: Zachary Ruttle
"""
import util
import snakeGameData as data


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()
        """
        return state.snake_pos
        """

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()
        """
        return state.snake_pos == state.food_pos
        """

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()
        
def aStarSearch(state):
    """Search the node that has the lowest combined cost and heuristic first."""
    list = util.PriorityQueue()
    foodPos = data.food_pos
    states = []
    cur = (data.snake_pos, [], 0)
    list.push(cur, 0)
    while not list.isEmpty():
        cur, actions, previousCost = list.pop()
        if cur not in states:
            states.append(cur)
            #if cur == foodPos:
            if cur[0] == foodPos[0] and cur[1] == foodPos[1]:
                return actions
            for successor, action, cost in getSuccessors(cur):
                list.push((successor, actions + [action], previousCost + cost), previousCost + cost + util.manhattanDistance(foodPos, successor))
                #list.push((successor, actions + [action], previousCost + cost), previousCost + cost + eu(successor, foodPos))

    

    return actions

def getSuccessors(state):
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
        x,y = state
        if action == 'DOWN':
            dx, dy = 0, 10
        if action == 'UP':
            dx, dy = 0, -10
        if action == 'LEFT':
            dx, dy = -10, -0
        if action == 'RIGHT':
            dx, dy = 10, 0
        nextx, nexty = int(x + dx), int(y + dy)

        if(nextx >= data.frame_size_x or nexty >= data.frame_size_y
           or nexty <= 0 or nextx <= 0):
            continue

        if([nextx, nexty] in data.snake_body):
            continue

        #if not self.illegal_move(action):
        #    nextState = (nextx, nexty)
        #    cost = 1
        #    successors.append((nextState, action, cost))
        if not illegal_move(action):
            nextState = (nextx, nexty)
            cost = 1
            successors.append((nextState, action, cost))
    return successors


def illegal_move(nextMove):
    if nextMove == 'UP' and data.direction == 'DOWN':
        return True
    if nextMove == 'DOWN' and data.direction == 'UP':
        return True
    if nextMove == 'LEFT' and data.direction == 'RIGHT':
        return True
    if nextMove == 'RIGHT' and data.direction == 'LEFT':
        return True
        
    return False

def eu(curPos, foodPos):
    xy1 = curPos;
    xy2 = foodPos;
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5