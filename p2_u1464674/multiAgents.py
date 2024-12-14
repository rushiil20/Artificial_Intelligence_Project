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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        # Get the current position of the closest ghost
        minGhostDistance = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        # Calculate the reciprocal of the distance to the closest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDistance = min(foodDistances) if foodDistances else 1  # Avoid division by zero
        reciprocalFoodDistance = 1.0 / minFoodDistance

        # Penalize if the next state will result in a collision with a ghost
        if minGhostDistance < 2:
            return -float('inf')

        # Return a weighted combination of reciprocal food distance and the score
        return successorGameState.getScore() + reciprocalFoodDistance

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
        def minimax_decision(state, depth, agent_index):
         if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None

         if agent_index == 0:  # Pacman's turn (MAX)
          best_value = float("-inf")
          best_action = None
          for action in state.getLegalActions(agent_index):
            successor_state = state.generateSuccessor(agent_index, action)
            score, _ = minimax_decision(successor_state, depth, agent_index + 1)
            if score > best_value:
                best_value = score
                best_action = action
          return best_value, best_action
         else:  # Ghosts' turns (MIN)
          worst_value = float("inf")
          next_agent = agent_index + 1
          if next_agent == state.getNumAgents():  # Next turn is Pacman's
            next_agent = 0
            depth -= 1  # Decrease depth when all ghosts have moved
          for action in state.getLegalActions(agent_index):
            successor_state = state.generateSuccessor(agent_index, action)
            score, _ = minimax_decision(successor_state, depth, next_agent)
            worst_value = min(worst_value, score)
          return worst_value, None

        _,best_action = minimax_decision(gameState, self.depth, 0)
        return best_action
        
        
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        def alpha_beta_search(state, depth, alpha, beta, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Pacman's turn (MAX)
                value = float("-inf")
                bestAction = None
                for action in state.getLegalActions(agentIndex):
                    successor_state = state.generateSuccessor(agentIndex, action)
                    score, _ = alpha_beta_search(successor_state, depth, alpha, beta, agentIndex + 1)
                    if score > value:
                        value = score
                        bestAction = action
                    alpha = max(alpha, value)
                    if value > beta:
                        return value, bestAction
                return value, bestAction
            else:  # Ghosts' turns (MIN)
                value = float("inf")
                next_agent = agentIndex + 1
                if next_agent == state.getNumAgents():  # Next turn is Pacman's
                    next_agent = 0
                    depth -= 1  # Decrease depth when all ghosts have moved
                for action in state.getLegalActions(agentIndex):
                    successor_state = state.generateSuccessor(agentIndex, action)
                    score, _ = alpha_beta_search(successor_state, depth, alpha, beta, next_agent)
                    value = min(value, score)
                    beta = min(beta, value)
                    if value < alpha:
                        return value, None
                return value, None

        _, bestAction = alpha_beta_search(gameState, self.depth, float("-inf"), float("inf"), 0)
        return bestAction
    
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
        def max_value(state, depth):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            value = float('-inf')
            best_action = None
            legal_actions = state.getLegalActions(0)
            for action in legal_actions:
                successor_state = state.generateSuccessor(0, action)
                successor_value, _ = expect_value(successor_state, depth, 1)
                if successor_value > value:
                    value = successor_value
                    best_action = action
            return value, best_action

        def expect_value(state, depth, ghost_index):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            value = 0
            legal_actions = state.getLegalActions(ghost_index)
            probability = 1.0 / len(legal_actions)
            for action in legal_actions:
                successor_state = state.generateSuccessor(ghost_index, action)
                if ghost_index == state.getNumAgents() - 1:
                    successor_value, _ = max_value(successor_state, depth - 1)
                else:
                    successor_value, _ = expect_value(successor_state, depth, ghost_index + 1)
                value += probability * successor_value
            return value, None

        _, best_action = max_value(gameState, self.depth)
        return best_action
        
            
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacman_pos = currentGameState.getPacmanPosition()
    food_grid = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()

    # Initialize evaluation score
    evaluation_score = currentGameState.getScore()

    # Factor 1: Distance to the nearest food pellets
    food_distances = [manhattanDistance(pacman_pos, food) for food in food_grid.asList()]
    if food_distances:
        closestFoodDistance = min(food_distances)
        evaluation_score += 1.0 / closestFoodDistance

    # Factor 2: Proximity to ghosts
    for ghost_state in ghost_states:
        ghost_position = ghost_state.getPosition()
        distance_to_ghost = manhattanDistance(pacman_pos, ghost_position)
        if distance_to_ghost == 0:
            # If Pacman is touching a ghost, heavily penalize the score
            evaluation_score -= 1000
        else:
            # Otherwise, inversely weight the distance to the ghost
            evaluation_score += 1.0 / distance_to_ghost

    # Factor 3: Number of remaining food pellets
    num_food_pellets = food_grid.count()
    if num_food_pellets == 0:
        # If all food pellets are eaten, heavily reward the score
        evaluation_score += 1000
    else:
        # Otherwise, inversely weight the number of remaining food pellets
        evaluation_score += 10.0 / num_food_pellets

    return evaluation_score


# Abbreviation
better = betterEvaluationFunction
