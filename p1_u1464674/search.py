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
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    visited_set = set()

    # initialize stack with start position
    start_state = problem.getStartState()
    stack.push((start_state, [], 0))

    while not stack.isEmpty():
        current_state, actions, cost = stack.pop()

        if problem.isGoalState(current_state):
            return actions

        if current_state not in visited_set:
            visited_set.add(current_state)

            successors = problem.getSuccessors(current_state)
            for successor, action, step_cost in successors:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                stack.push((successor, new_actions, new_cost))

    return []
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    visited_set = set()

    # initialize queue with start position
    start_state = problem.getStartState()
    queue.push((start_state, [], 0))

    while not queue.isEmpty():
        current_state, actions, cost = queue.pop()

        if problem.isGoalState(current_state):
            return actions

        if current_state not in visited_set:
            visited_set.add(current_state)

            successors = problem.getSuccessors(current_state)
            for successor, action, step_cost in successors:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                queue.push((successor, new_actions, new_cost))

    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueue()
    visited_set = set()

    # initialize priority queue with start position
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), 0)

    while not priority_queue.isEmpty():
        current_state, actions, cost = priority_queue.pop()         

        if problem.isGoalState(current_state):
            return actions

        if current_state not in visited_set:
            visited_set.add(current_state)

            successors = problem.getSuccessors(current_state)
            for successor, action, step_cost in successors:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                priority_queue.push((successor, new_actions, new_cost), new_cost)

    return []   

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueue()
    visited_set = set()

    # initialize priority queue with start position
    start_state = problem.getStartState()
    start_heuristic = heuristic(start_state, problem)
    priority_queue.push((start_state, [], 0), start_heuristic)

    while not priority_queue.isEmpty():
        current_state, actions, cost = priority_queue.pop()

        if problem.isGoalState(current_state):
            return actions

        if current_state not in visited_set:
            visited_set.add(current_state)

            successors = problem.getSuccessors(current_state)
            for successor, action, step_cost in successors:
                new_actions = actions + [action]
                new_cost = cost + step_cost
                new_heuristic = heuristic(successor, problem)
                priority_queue.push((successor, new_actions, new_cost), new_cost + new_heuristic)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
