# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for k in range(self.iterations):
          optimal = util.Counter()
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state): #check for terminal state
              optimal[state] = 0
            else:
              best = -9999999999999999999 #initialize to low number

              for action in self.mdp.getPossibleActions(state):
                score = 0
                for transition, prob in self.mdp.getTransitionStatesAndProbs(state,action):
                    score += prob * (self.mdp.getReward(state, action, transition) + (self.discount * self.values[transition]))
                best = max(score, best)
                optimal[state] = best

          self.values = optimal






    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for transition, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            q += prob * (self.mdp.getReward(state, action, transition) + self.discount * self.values[transition])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        val = -9999999999999999
        action =  None

        if self.mdp.isTerminal(state):
          return None

        for act in self.mdp.getPossibleActions(state):
          q = self.computeQValueFromValues(state, act)
          if q >= val:
            val = q
            action = act

        return action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for k in range(self.iterations):
            states = self.mdp.getStates()
            state = states[k % len(states)]

            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                q = self.computeQValueFromValues(state, action)
                self.values[state] = q

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        #COMPUTE PREDECESSORS OF ALL STATES
        #PREDECESSOR: states that have a nonzero probability of reaching s by taking some action a
        preds = {}
        for state in self.mdp.getStates():
            preds[state] = set()

        for s in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(s): #look at all possible actions
                for successor, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                    preds[successor].add(s)

        #INITIALIZE AN EMPTY PRIORITY QUEUE
        queue = util.PriorityQueue()

        #FOR EACH NON TERMINAL STATE s
        #MUST ITERATE OVER STATES IN ORDER RETURNED MY self.mdp.getStates()
        for s in self.mdp.getStates():
          if not self.mdp.isTerminal(s):
            #FIND ABS OF DIFF BETWEEN CURRENT VALUE OF STATE IN SELF.VALUES AND HIGHEST Q VALUE ACROSS ALL POSSIBLE ACTIONS
            qvals = [self.getQValue(s, a) for a in self.mdp.getPossibleActions(s)]
            diff = abs(self.values[s] - max(qvals))
            queue.push(s, -diff) #use push

        #FOR EACH ITERATION
        for i in range(self.iterations):
          #IF PRIORITY QUEUE IS EMPTY TERMINATE
          if queue.isEmpty():
            return
          #POP A STATE OFF THE PRIORITY QUEUE
          s = queue.pop()

          #UPDATE STATES VALUE (IF NOT TERMINAL) IN SELF.VALUES
          if not self.mdp.isTerminal(s):
              qvals = [self.getQValue(s, a) for a in self.mdp.getPossibleActions(s)]
              self.values[s] = max(qvals)

          #FOR EACH PREDESSOR p OF s
          for p in preds[s]:
            #FIND ABS OF DIFF BEWEEN CURRENT VALE OF P IN SELF.VALUES AND HIGHEST Q ACROSS ALL ACTIONS FROM P
            #DO NOT UPDATE SELF.VALUES[P]
            if not self.mdp.isTerminal(p):
              qvals = [self.getQValue(p, a) for a in self.mdp.getPossibleActions(p)]
              diff = abs(self.values[p] - max(qvals))

              #IF DIFF > THETA, PUSH P TO QUEUE WITH PRIORITY -DIFF
              if diff > self.theta:
                queue.update(p, -diff) #use update


        """
        #function to get predecessors - NOT WORKING
        def getPredecessors(state):
            predecessors = {}
            if not self.mdp.isTerminal(state):
              for action in self.mdp.getPossibleActions(state):
                  for successor, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                      predecessors[successor].add(state)
            return predecessors
            """
