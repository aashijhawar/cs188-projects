# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # if win state return infinity
        if successorGameState.isWin():
            return 99999999999999

        #set initial score
        ghostDist = manhattanDistance(currentGameState.getGhostPosition(1), newPos)
        score = ghostDist + successorGameState.getScore()

        #set bestfood to chase to high number
        bestFood = 9999999

        #find minimum  food
        foodList = [manhattanDistance(food, newPos) for food in newFood]
        bestFood = min(foodList)


        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            score += 100

        if action == Directions.STOP:
            score -= 999

        score -= 7 * bestFood

        return score




        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        #computes search, minimum if agent > 0,  maximum if agent is  pacman
        def minimax(agent, depth, state):
            if agent == state.getNumAgents():
                if depth is self.depth:
                    return self.evaluationFunction(state)
                else:
                    return minimax(0, depth + 1, state)
            else:
                actionList = state.getLegalActions(agent)

                if len(actionList) is 0:
                    return self.evaluationFunction(state)

                #computes minimax for every possible action, increment agent by 1 to make ghost
                successors = [minimax(agent+1,depth,state.generateSuccessor(agent, action)) for action in actionList]

                if agent is 0:
                    return max(successors)
                else:
                    return min(successors)

        #minimax each legal action at next level of agent
        def moves(action):
            return minimax(1, 1, gameState.generateSuccessor(0, action))

        return max(gameState.getLegalActions(0), key = moves)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        ghosts = gameState.getNumAgents() - 1


        def min_value(state, depth, agent, ghosts, alpha, beta):

            if state.isWin() or state.isLose():
              return self.evaluationFunction(state)

            v = 999999999999
            for successor in state.getLegalActions(agent):
              if agent == ghosts:
                if depth < self.depth:
                  x = max_value(state.generateSuccessor(agent, successor), depth + 1, ghosts, alpha, beta)
                else:
                  x = self.evaluationFunction(state.generateSuccessor(agent, successor))
              else:
                x = min_value(state.generateSuccessor(agent, successor), depth, agent + 1, ghosts, alpha, beta)
              if x < v:
                v = x

              if v < alpha:
                return v

              beta = min(beta, v)

            return v

        def max_value(state, depth, ghosts, alpha, beta):

            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            v = -999999999999
            best = Directions.STOP

            for successor in state.getLegalActions(0):
              x = min_value(state.generateSuccessor(0, successor), depth, 1, ghosts, alpha, beta)
              if x > v:
                v = x
                best = successor

              if v > beta:
                return v

              alpha = max(alpha, v)

            if depth > 1:
              return v

            return best

        return max_value(gameState, 1, ghosts, -999999999999, 999999999999)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def mean(lst):
            return (sum(lst)) / len(lst)

        def expectimax(agent, depth, state):
            if agent == state.getNumAgents():
                if depth is self.depth:
                    return self.evaluationFunction(state)
                else:
                    return expectimax(0, depth + 1, state)
            else:
                actionList = state.getLegalActions(agent)

                if len(actionList) is 0:
                    return self.evaluationFunction(state)

                #computes minimax for every possible action, increment agent by 1 to make ghost
                successors = [expectimax(agent+1,depth,state.generateSuccessor(agent, action)) for action in actionList]

                if agent is 0:
                    return max(successors)
                else:
                    return mean(successors)

        #minimax each legal action at next level of agent
        def moves(action):
            return expectimax(1, 1, gameState.generateSuccessor(0, action))

        return max(gameState.getLegalActions(0), key = moves)



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    #return score

# Abbreviation
better = betterEvaluationFunction
