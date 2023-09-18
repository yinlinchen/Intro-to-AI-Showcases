"""
Q learning agent
@author ElyseHaas
"""

import time
import os
import traceback
import sys
import random, util, math
import snakeGameData as sg_data

# from game import *
# from learningAgents import ReinforcementAgent
# from featureExtractors import *

# from pygame import Directions
from searchAgent import Agent
# from SnakeGame import Agent as GameAgent

frame_size_x = 720
frame_size_y = 480


class ValueEstimationAgent(Agent):
    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining=10):
        """
        Sets options, which can be passed in via the snake command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    ####################################
    #   These Functions are Overridden #
    ####################################

    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()


class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getLegalActions(state) to know which actions
                      are available in a state
    """

    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        util.raiseNotDefined()

    ####################################
    #    Read These Functions          #
    ####################################

    # def getLegalActions(self, state):
    #     """
    #       Get the actions available for a given
    #       state. This is what you should use to
    #       obtain legal actions for a state
    #     """
    #     return self.actionFn(state)

    def observeTransition(self, state, algorithm, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state, algorithm, nextState, deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, actionFn=None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        # if actionFn == None:
            # actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qVals = util.Counter()
    
    def get_state(self, game):
        #position of the snake, the food and the snake's body and score too
        self.state = "NotImplemented"
        return (sg_data.snake_pos, sg_data.snake_body, sg_data.food_pos, sg_data.score)
    
    # def getLegalActions(self, state):
    #     """
    #       Get the actions available for a given
    #       state. This is what you should use to
    #       obtain legal actions for a state
    #     """
    #     return self.actionFn(state)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qVals[(state, action)]  # return q value at (state, action)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,algorithm)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)  # set actions to the legal actions at state
        if len(actions) == 0:  # if there are no legal actions
            return 0.0  # return value of 0.0
        opt = util.Counter()  # create counter object
        for action in actions:  # iterate through legal actions
            opt[action] = self.getQValue(state, action)  # opt at index action is set to the q value of (state, action)
        return opt[opt.argMax()]  # return dict value at the argMax value of dict which is the q value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.get_legal_moves()  # find legal actions at state
        optimal = None  # initial optimal value is none
        max = float("-inf")  # set max to -infinity
        for action in actions:  # iterate through legal actions
            if not self.gameOverState(sg_data.snake_pos, sg_data.snake_body):
                qVal = self.qVals[(state, action)]  # set q value to q value at state, action
                if max < qVal:  # if max is less than the q value
                    max = qVal  # set max to q value
                    optimal = action  # in this case, this action is the most optimal when q value is the max value
        return optimal  # return the best action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        coin = util.flipCoin(self.epsilon)  # take random action with probability of success epsilon
        if coin:  # if an action has chances of being taken
            if not self.gameOverState(sg_data.snake_pos, sg_data.snake_body):
                return self.get_random_legal_move(state)  # return the random choice of legal actions found at state
        else:
            return self.getPolicy(state)  # else return the computed action from getPolicy method
        
    def get_random_legal_move(self, state):
        print("getting random legal move")
        if sg_data.direction == 'DOWN':
            return random.choice(['UP','LEFT','RIGHT'])
        if sg_data.direction == 'UP':
            return random.choice(['DOWN','LEFT','RIGHT'])
        if sg_data.direction == 'RIGHT':
            return random.choice(['UP','DOWN','LEFT'])
        if sg_data.direction == 'LEFT':
            return random.choice(['UP','DOWN','RIGHT'])
        
        return random.choice(['UP','DOWN', 'Left','RIGHT'])
    
    def get_legal_moves(self):
        print("getting legal moves")
        if sg_data.direction == 'DOWN':
            return (['UP','LEFT','RIGHT'])
        if sg_data.direction == 'UP':
            return (['DOWN','LEFT','RIGHT'])
        if sg_data.direction == 'RIGHT':
            return (['UP','DOWN','LEFT'])
        if sg_data.direction == 'LEFT':
            return (['UP','DOWN','RIGHT'])
        
        return ['UP','DOWN', 'Left','RIGHT']
    
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

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        oldqVal = self.getQValue(state, action)  # get q value of current state and algorithm
        oldState = (1 - self.alpha) * oldqVal  # the state is updated by using equation (1 - alpha) * q
        rewards = self.alpha * reward  # rewards is updated with equation alpha * reward
        if not nextState:  # if there's no next state
            self.qVals[(state, action)] = oldState + rewards  # the q value at index (state, action) is equal to the
            # updated state + updated rewards
        else:
            nextStates = self.alpha * self.discount * self.getValue(
                nextState)  # else the next state is equal to this equation
            self.qVals[(state, action)] = oldState + rewards + nextStates  # update q value at (state, action)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
class SnakeGameQAgent(QLearningAgent):
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
