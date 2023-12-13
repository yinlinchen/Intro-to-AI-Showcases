
import math
from piece import PlayerID
from game import Game
from game import GameState

# Node type alias for Beam Search
Node = tuple[float, any, GameState, 'Node']

class MinimaxAgent():
    """
    A minimax agent that uses a provided evaluation function
    to return the best action.
    """

    def __init__(self, evalFunc, depth = 2, max_states=20_000):
        """
        Initializes agent with depth and evaluation function.
        """
        self.depth = depth
        self.evalFunc = evalFunc
        self.states_explored = 0  # really, states expanded
        self.max_states = max_states


    def evaluationFunction(self, id: PlayerID, currentGameState: GameState):
        """
        Returns the score based on the state and evaluation function.
        """
        return self.evalFunc(id, currentGameState)


    def getAction(self, gameState: GameState, id: PlayerID):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        # we want the move for the specified Player:
        # note that depending on what situation the gameState is in, the minimax will
        # return a different kind of action.
        # If a normal action is needed: (piece, x, y)
        # If a decision/selection is needed: Triple
        # If a promotion is needed: (x, y)

        # tracking metrics
        self.states_explored = 0
        action, val = self.miniMax(gameState, id, 0)
        
        print(f"MinimaxAgent picked {action} with value {val}, states explored: {self.states_explored} ")
            
        return (action, val)


    def miniMax(self, gameState: GameState, id: PlayerID, depth):
        """
        helper function that makes sure we have not reached the terminal state.
        Minimax will determine which kind of agent we are dealing with and call the correct max or min
        function.
        """
        # order of turns based on my agent: player1, player 1 makes a decision, 
        # player 2 makes a decision, advance turn

        # weirdest scenario: player 2 places 8th piece leading to no triple and empty hand
        # but boops player1 into a decision (two triples), player1 makes decision, 
        # THEN player2 chooses their promotion.

        # moral of the story, player 1 always goes first on any kind of decision/promotion
        if depth == self.depth * 2 or gameState.is_terminal() or self.states_explored >= self.max_states:
            return (None, self.evaluationFunction(id, gameState))
        
        self.states_explored += 1
        
        # Uncomment for extended debugging
        #print(f"miniMax: {depth}, states: {self.states_explored}")
        
        # this is to deal with the edge cases where the non active player 
        # has a decision that needs to be made, by default we allow player
        # 1 to make their decision first
        if gameState.player1.pending_decision():
            return self.maxValue(gameState, depth)
        if gameState.player2.pending_decision():
            return self.minValue(gameState, depth)
        if id is PlayerID.ONE:
            return self.maxValue(gameState, depth)
        else:
            return self.minValue(gameState, depth)


    def maxValue(self, gameState: GameState, depth):
        """
        helper function that handles the case when we have a max player.
        it will call back to the minimax function when we must evaluate 
        the next depth
        """

        bestValue = -float('inf')
        bestAction = ()
        #handle when player1 has a pending decision
        if gameState.player1.pending_decision():
            for s in gameState.get_legal_selections(PlayerID.ONE):
                newDepth = depth
                new_gs = gameState.generate_successor_from_selection(PlayerID.ONE, s)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth)[1]
                if actionValue > bestValue:
                    bestValue = actionValue
                    bestAction = s

            # NOTE: if current player, could also check must_promote here for official rules edge case,
            # though we'd need to adjust GameState as well to flag both at once

        #handle when player 1 must promote, we do not have to worry about player 2
        #because this part will only ever be called when player 1 must promote and is
        #the curr player
        elif gameState.must_promote:
            for p in gameState.get_legal_promotions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_promotion(p[0], p[1])
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth)[1]
                if actionValue > bestValue:
                    bestValue = actionValue
                    bestAction = p

        #handle normal actions (placing a piece)
        else:
            for a in gameState.get_legal_actions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_action(a)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth)[1]
                if actionValue > bestValue:
                    bestValue = actionValue
                    bestAction = a

        return (bestAction, bestValue)
        


    def minValue(self, gameState: GameState, depth):
        """
        helper function that handles the case when we have a min player.
        it will call back to the minimax function when we must evaluate 
        the next depth
        """

        bestValue = float('inf')
        bestAction = "minValue"
        #handle when player2 has a pending decision
        if gameState.player2.pending_decision():
            for s in gameState.get_legal_selections(PlayerID.TWO):
                newDepth = depth
                new_gs = gameState.generate_successor_from_selection(PlayerID.TWO, s)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth)[1]
                if actionValue < bestValue:
                    bestValue = actionValue
                    bestAction = s

        #handle when player 2 must promote, we do not have to worry about player 1
        #because this part will only ever be called when player 2 must promote and is
        #the curr player
        elif gameState.must_promote:
            for p in gameState.get_legal_promotions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_promotion(p[0], p[1])
                if not new_gs.has_pending_decision(): newDepth = depth + 1 
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth)[1]
                if actionValue < bestValue:
                    bestValue = actionValue
                    bestAction = p

        else:
            for a in gameState.get_legal_actions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_action(a)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth)[1]
                if actionValue < bestValue:
                    bestValue = actionValue
                    bestAction = a

        return (bestAction, bestValue) 


class AlphaBetaAgent(MinimaxAgent):
    """
    A minimax agent that includes alpha-beta pruning for efficiency.
    """
    def __init__(self, evalFunc, depth = 2, max_states=20_000):
        """
        Initializes agent with base params.
        """
        super().__init__(evalFunc, depth=depth, max_states=max_states)


    def getAction(self, gameState: GameState, id: PlayerID):
        """
        Helper to kick off the search.
        """
        alpha = -math.inf
        beta = math.inf
        self.states_explored = 0
        action, val = self.miniMax(gameState, id, 0, alpha, beta)

        print(f"AlphaBetaAgent picked {action} with value {val}, states explored: {self.states_explored} ")
        
        return (action, val)


    def miniMax(self, gameState: GameState, id: PlayerID, depth: int, alpha, beta):
        """
        Helper function that makes sure we have not reached the terminal state.
        """
        if depth == self.depth * 2 or gameState.is_terminal() or self.states_explored >= self.max_states:
            return (None, self.evaluationFunction(id, gameState))
        
        self.states_explored += 1
        
        # Uncomment for extended debugging
        #print(f"miniMax: {depth}, states: {self.states_explored}")
        
        # this is to deal with the edge cases where the non active player 
        # has a decision that needs to be made, by default we allow player
        # 1 to make their decision first
        if gameState.player1.pending_decision():
            return self.maxValue(gameState, depth, alpha, beta)
        if gameState.player2.pending_decision():
            return self.minValue(gameState, depth, alpha, beta)
        if id is PlayerID.ONE:
            return self.maxValue(gameState, depth, alpha, beta)
        else:
            return self.minValue(gameState, depth, alpha, beta)


    def maxValue(self, gameState: GameState, depth, alpha, beta):
        """
        For the alpha-beta version, we only care if value found is greater
        than the current beta.
        """

        bestValue = -float('inf')
        bestAction = None
        #handle when player1 has a pending decision
        if gameState.player1.pending_decision():
            for s in gameState.get_legal_selections(PlayerID.ONE):
                newDepth = depth
                new_gs = gameState.generate_successor_from_selection(PlayerID.ONE, s)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth, alpha, beta)[1]
                if actionValue > bestValue:
                    bestValue = actionValue
                    bestAction = s
                    alpha = max(alpha, actionValue)
                if bestValue > beta:
                    return (bestAction, actionValue)

        #handle when player 1 must promote, we do not have to worry about player 2
        #because this part will only ever be called when player 1 must promote and is
        #the curr player
        elif gameState.must_promote:
            for p in gameState.get_legal_promotions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_promotion(p[0], p[1])
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth, alpha, beta)[1]
                if actionValue > bestValue:
                    bestValue = actionValue
                    bestAction = p
                    alpha = max(alpha, actionValue)
                if bestValue > beta:
                    return (bestAction, actionValue)

        #handle normal actions (placing a piece)
        else:
            for a in gameState.get_legal_actions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_action(a)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth, alpha, beta)[1]
                if actionValue > bestValue:
                    bestValue = actionValue
                    bestAction = a
                    alpha = max(alpha, actionValue)
                if bestValue > beta:
                    return (bestAction, actionValue)

        return (bestAction, bestValue)
        


    def minValue(self, gameState: GameState, depth, alpha, beta):
        """
        For the alpha-beta version, we only care if value found is less
        than the current alpha.
        """
        bestValue = float('inf')
        bestAction = None
        #handle when player2 has a pending decision
        if gameState.player2.pending_decision():
            for s in gameState.get_legal_selections(PlayerID.TWO):
                newDepth = depth
                new_gs = gameState.generate_successor_from_selection(PlayerID.TWO, s)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth, alpha, beta)[1]
                if actionValue < bestValue:
                    bestValue = actionValue
                    bestAction = s
                    beta = min(beta, bestValue)
                if bestValue < alpha:
                    return (bestAction, bestValue)

        #handle when player 2 must promote, we do not have to worry about player 1
        #because this part will only ever be called when player 2 must promote and is
        #the curr player
        elif gameState.must_promote:
            for p in gameState.get_legal_promotions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_promotion(p[0], p[1])
                if not new_gs.has_pending_decision(): newDepth = depth + 1 
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth, alpha, beta)[1]
                if actionValue < bestValue:
                    bestValue = actionValue
                    bestAction = p
                    beta = min(beta, bestValue)
                if bestValue < alpha:
                    return (bestAction, bestValue)

        else:
            for a in gameState.get_legal_actions():
                newDepth = depth
                new_gs = gameState.generate_successor_from_action(a)
                if not new_gs.has_pending_decision(): newDepth = depth + 1
                actionValue = self.miniMax(new_gs, new_gs.turn, newDepth, alpha, beta)[1]
                if actionValue < bestValue:
                    bestValue = actionValue
                    bestAction = a
                    beta = min(beta, bestValue)
                if bestValue < alpha:
                    return (bestAction, bestValue)

        return (bestAction, bestValue) 

    

class BeamAgent(MinimaxAgent):
    """
    An agent that uses Beam Search for Shannon's Type B strategy:
    this explores strong states to further depths.
    """
    def __init__(self, evalFunc, width=5, depth = 2, max_states=20_000):
        """
        Initializes agent with width and base params.
        """
        super().__init__(evalFunc, depth=depth, max_states=max_states)
        self.width = width


    def getAction(self, gameState: GameState, id: PlayerID):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        # tracking metrics
        self.states_explored = 0

        # This is actually more of a BFS style search, where we maintain the top width states / actions
        # at each level.
        # A node is a tuple of (score, action, state, parent_node)
        start = (0, None, gameState, None)
        depth = 0
        allTerm = gameState.is_terminal()
        orig_player = id
        nodes = [start]
        
        # if the current move led to pending decisions, that is our actual starting point,
        # as opposed to a singular state
        if (not gameState.is_terminal()) and gameState.has_pending_decision():
            nodes = self.getPossibleSubnodes(start)
            allTerm = all(node[2].is_terminal() for node in nodes)


        # perform beam search until max depth / max_states / terminal
        while depth < self.depth * 2 and (not allTerm) and self.states_explored < self.max_states:
            id = nodes[0][2].turn  # turn will change, and player will select best for them
            nodes, allTerm = self.beamSearchLevel(id, nodes)
            # Uncomment for advanced debugging
            #print(f"Depth: {depth}, allTerm: {allTerm}, nodes: {len(nodes)}, states: {self.states_explored}")

            depth += 1


        # we want the best score for the original player,
        # which may not be id if we bailed early
        top = orig_player == PlayerID.ONE
        nodes.sort(key=lambda node: node[0], reverse=top)
        bestNode = nodes[0]
        action, val = self.getOriginalAction(bestNode)

        print(f"BeamAgent picked {action} with value {val}, states explored: {self.states_explored} ")
        return (action, val)
    
    
    def beamSearchLevel(self, id: PlayerID, nodes: list[Node]):
        """
        Performs one level of Beam Search.
        This returns the successor nodes to expand.
        """
        successors: list[Node] = []
        allTerm = True
        for node in nodes:
            # get all successors for this state, with their scores
            state = node[2]

            # if terminal, don't go any further, this is the score, just maintain it
            if state.is_terminal():
                successors.append(node)
                continue

            # we're at least adding some new ones
            allTerm = False

            # get all successors
            for a in state.get_legal_actions():
                new_gs = state.generate_successor_from_action(a)

                # Flatten any pending decisions
                if (not new_gs.is_terminal()) and new_gs.has_pending_decision():
                    cur_node = (0, a, new_gs, node)
                    final_subnodes = self.getPossibleSubnodes(cur_node)
                    for subnode in final_subnodes:
                        successors.append(subnode)

                else:
                    # regular: just take the score for this one new game state
                    score = self.evaluationFunction(new_gs.turn, new_gs)
                    child: Node = (score, a, new_gs, node)
                    successors.append(child)
 
            self.states_explored += 1

        # filter the highest/lowest n successors, where n is the beam width
        top = id == PlayerID.ONE  # max wants top scores, min wants lowest
        successors.sort(key=lambda node: node[0], reverse=top)
        return successors[:self.width], allTerm
    

    def getPossibleSubnodes(self, cur_node: Node):
        """
        Helper to deal with the pending pseudostates while maintaing the
        current level.
        If we have pending decisions, resolve them each way
        so we can determine all valid next possible states, and still
        score and choose the best one.

        This is similar to the miniMax check, though queueing instead of
        recursive to avoid depth issues.
        """
        subnodes = [cur_node]
        final_subnodes = []
        while len(subnodes) > 0:
            node = subnodes.pop()
            gs = node[2]
            if gs.is_terminal() or not gs.has_pending_decision():
                # we're done with this one
                score = self.evaluationFunction(gs.turn, gs)
                final_node = (score, node[1], gs, node[3])
                final_subnodes.append(final_node)

            elif gs.player1.pending_decision():
                # get all possibilites for p1 selections
                for sel in gs.get_legal_selections(PlayerID.ONE):
                    sub_gs = gs.generate_successor_from_selection(PlayerID.ONE, sel)
                    sub_node = (0, sel, sub_gs, node)
                    subnodes.append(sub_node)

            elif gs.player2.pending_decision():
                # get all possibilites for p2 selections, for all p1 selections
                for sel in gs.get_legal_selections(PlayerID.TWO):
                    sub_gs = gs.generate_successor_from_selection(PlayerID.TWO, sel)
                    sub_node = (0, sel, sub_gs, node)
                    subnodes.append(sub_node)

            elif gs.must_promote:
                # get all possibilites for promotions
                for (x, y) in gs.get_legal_promotions():
                    sub_gs = gs.generate_successor_from_promotion(x, y)
                    sub_node = (0, (x, y), sub_gs, node)
                    subnodes.append(sub_node)
        

        return final_subnodes                    


    def getPlayerWhoMoved(self, new_gs: GameState) -> PlayerID:
        """
        Helper to retroactively determine whose move it was,
        given we had to resolve a pending decision on a miniMax scored
        game state.

        This follows our miniMax logic: we resolve player1 pending decisions
        first.
        """
        moved = new_gs.turn
        if new_gs.player1.pending_decision():
            moved = PlayerID.ONE
        elif new_gs.player2.pending_decision():
            moved = PlayerID.TWO
        return moved
    

    def getOriginalAction(self, node: Node):
        """
        Traces back the node to get the original action / sel / promotion
        move to take.
        """
        path = []
        score, action, _, parent = node
        while parent is not None:
           path.insert(0, action)
           _, action, _, parent = parent

        return path[0], score
