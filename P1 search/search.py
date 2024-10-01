# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """


    fringe = util.Stack() #LIFO structure holds places and moves to get there
    fringe.push((problem.getStartState(), [])) #initialize fringe with startState and no moves to get there
    #print(fringe)
    visited = [] #tracks visited places

    while (fringe.isEmpty() is False):
        current = fringe.pop()
        #print("Current: ", current)
        if problem.isGoalState(current[0]): #check if reached goal and if so return corresponding list of moves
            return current[1]
        else:
            visited.append(current[0]) # add current place to visited places
            for move in problem.getSuccessors(current[0]): #for each successor check if visited already/add to fringe
                if move[0] not in visited:
                    actions = current[1] + [move[1]]
                    fringe.push((move[0], actions))
    return None



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue() #FIFO structure holds places and moves to get there
    fringe.push((problem.getStartState(), [])) #initialize fringe with startState and no moves to get there
    visited = [] #tracks visited places

    while (fringe.isEmpty() is False):
        current = fringe.pop()
        if current[0] in visited:
            continue #SOURCE @https://infohost.nmt.edu/tcc/help/pubs/python/web/continue-statement.html
        visited.append(current[0]) # add current place to visited places
        if problem.isGoalState(current[0]): #check if reached goal and if so return corresponding list of moves
            return current[1]
        else:
            for move in problem.getSuccessors(current[0]): #for each successor check if visited already/add to fringe
                if move[0] not in visited:
                    actions = current[1] + [move[1]]
                    fringe.push((move[0], actions))
    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    #node = problem.getStartState()
    fringe = util.PriorityQueue() #queue ordered by priority values, holds places and moves to get there
    #print("Start: ", problem.getStartState())
    fringe.push((problem.getStartState(), []), 0) #initialize fringe with startState, no moves to get there, 0 priority
    visited = [] #tracks visited places

    while (fringe.isEmpty() is False):
        current = fringe.pop()
        if current[0] in visited:
            continue
        visited.append(current[0]) # add current place to visited places
        if problem.isGoalState(current[0]): #check if reached goal and if so return corresponding list of moves
            return current[1]
        else:
            for move in problem.getSuccessors(current[0]): #for each successor check if visited already/add to fringe
                if move[0] not in visited: #and move[0] not in fringe
                    actions = current[1] + [move[1]]
                    cost = problem.getCostOfActions(actions)
                    fringe.push((move[0], actions), cost)
    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #node = problem.getStartState()
    fringe = util.PriorityQueue() #queue ordered by priority values, holds places and moves to get there
    #print("Start: ", problem.getStartState())
    fringe.push((problem.getStartState(), []), 0) #initialize fringe with startState, no moves to get there, 0 priority
    visited = [] #tracks visited places

    while (fringe.isEmpty() is False):
        current = fringe.pop()
        if current[0] in visited:
            continue
        visited.append(current[0]) # add current place to visited places
        if problem.isGoalState(current[0]): #check if reached goal and if so return corresponding list of moves
            return current[1]
        else:
            for move in problem.getSuccessors(current[0]): #for each successor check if visited already/add to fringe
                if move[0] not in visited: #and move[0] not in fringe
                    actions = current[1] + [move[1]]
                    ascore = problem.getCostOfActions(actions) + heuristic(move[0], problem)
                    fringe.push((move[0], actions), ascore)
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
