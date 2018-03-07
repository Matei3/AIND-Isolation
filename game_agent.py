"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import numpy as np
import random
import isolation
import timeit
class Timeout(Exception):
	pass



def custom_score(game,player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player): 
        #("loss")
        return float("-inf")

    if game.is_winner(player):
        #print("win")
        return float("inf")
    
    #Implementation of the open_space() heuristic
    number_empty_spaces=[0, 0, 0, 0]
    own_position=game.get_player_location(player);
    empty_spaces=game.get_blank_spaces()
    for space in empty_spaces:
        if (space[0]>own_position[0]) and (space[1]>own_position[1]):
             number_empty_spaces[3]+=1
        if (space[0]<=own_position[0]) and (space[1]>own_position[1]):
             number_empty_spaces[2]+=1
        if (space[0]>own_position[0]) and (space[1]<=own_position[1]):
             number_empty_spaces[1]+=1
        if (space[0]<=own_position[0]) and (space[1]<=own_position[1]):
             number_empty_spaces[0]+=1
    central=np.std( number_empty_spaces)
    #Implementation of improved_score heuristic
    number_own_moves=float(len(game.get_legal_moves(player)))
    number_adv_moves=float(len(game.get_legal_moves(game.get_opponent(player))))     
    #Combination of the results of the 2 heuristics
    return number_own_moves-8*number_adv_moves-central

    


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='alphabeta', timeout=15.,reflection=1):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.start_time=None
        self.TIMER_THRESHOLD = timeout
        self.reflection=reflection
    
    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        # If there are no legal moves an invalid position is returned
        if not legal_moves:
           return (-1,-1)
        # The first available move is saved as a backup if an exception occurs before any other moves can be evaluated
        current_move=legal_moves[0]
        # In case there is just one legal move, there is no reason for evaluation so we return it
        if (len(legal_moves)==1):
            return current_move
        
        #The reflection strategy described by Malcolm is implemented
        
        #Generalization of reflection method for every table size  
        width=int(game.width)
        height=int(game.height)
        central=(int(height/2),int(width/2))
        table_size=width*height
            
        #Implementation of the reflection strategy
        if self.reflection:
            free_positions=game.get_blank_spaces()
            if len(free_positions)==table_size:
                return central
            if len(free_positions)==table_size-1:
                self.reflection=0
                if central not in legal_moves:
                    return(central[0]-1,central[1]-1)
                else:
                    return central
            adv_pos=game.get_player_location(game.get_opponent(self))
            own_moves=game.get_legal_moves(self)
            refl_pos=(abs(adv_pos[0]-(height-1)),abs(adv_pos[1]-(width-1)))
            if refl_pos in own_moves:
                return refl_pos
            else:
                self.reflection=0
        
        moves=legal_moves
        max_score=-float("Inf")
        try:
          
            if self.iterative:
                if self.method=="alphabeta":
                #Implementation of iterative alphabeta
                    for depth in range(1,49):      
                        if depth==1:
                            max_score=self.score(game.forecast_move(legal_moves[0]),self) 
                            #For the first level each move is evaluated
                            for move in legal_moves:             
                                minimax_score=self.minimax(game.forecast_move(move),0)
                                score=minimax_score[0]
                                if score>=max_score:
                                    max_score=score  
                                    current_move=move
                        else:    
                            scores=[]
                            pos_moves=[]    
                            #For the deeper levels alphabeta is called for evaluation
                            for move in legal_moves:
                                alpha_result=self.alphabeta(game=game.forecast_move(move),depth=depth-1,maximizing_player=False)
                                scores.append(alpha_result[0])
                                if max(scores)==float("Inf"):
                                    return move
                                    break  
                            if scores:              
                                current_move=legal_moves[scores.index(max(scores))]
                    return current_move            
                else:
                #Implementation of iterative minimax
                    for depth in range(1,49):      
                        if depth==1:
                            max_score=self.score(game.forecast_move(legal_moves[0]),self) 
                            #For the first level each move is evaluated
                            for move in legal_moves:             
                                minimax_score=self.minimax(game.forecast_move(move),0)
                                score=minimax_score[0]
                                if score>=max_score:
                                    max_score=score  
                                    current_move=move
                        else:    
                            scores=[]
                            pos_moves=[]    
                            #For the deeper levels minimax is called for evaluation
                            for move in legal_moves:
                                minimax_result=self.minimax(game=game.forecast_move(move),depth=depth-1,maximizing_player=False)
                                scores.append(minimax_result[0])
                                if max(scores)==float("Inf"):
                                    return move
                                    break  
                            if scores:              
                                current_move=legal_moves[scores.index(max(scores))]        
                    return current_move
            else:
                if self.method=='alphabeta':
                 #Implementation of non-iterative alphabeta
                    try:
                        return  self.alphabeta(game,self.search_depth)[1]          
                    except Timeout:
                        return current_move
                else:
                 #Implementation of non-iterative minimax  
                    try:
                        return   self.minimax(game,self.search_depth)[1]          
                    except Timeout:
                        return current_move
        
        except Timeout:
            return current_move
        # Return the best move from the last completed search iteration
        #raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
         # TODO: finish this function!
        #Evaluation of the max level
        if maximizing_player:
            #print("player")
            scores=[]
            legal_moves=game.get_legal_moves(self)
            if len(legal_moves)==0:
                return (self.score(game,self),(-1,-1))
            if depth==0:
                return (self.score(game,self),(-1,-1))
            else:
                for move in legal_moves:
                    minmax_result=self.minimax(game.forecast_move(move),depth-1,not maximizing_player)
                    scores.append(minmax_result[0])
            return(scores[scores.index(max(scores))],legal_moves[scores.index(max(scores))])      
        else:
        #Evaluation of the min level
            scores=[]
            legal_moves=game.get_legal_moves(game.get_opponent(self))
            if len(legal_moves)==0:
                return (self.score(game,self),(-1,-1))
            if depth==0:
                return (self.score(game,self),(-1,-1)) 
            else:
                for move in legal_moves:
                    minmax_result=self.minimax(game.forecast_move(move),depth-1,not maximizing_player)
                    scores.append(minmax_result[0])
            return(scores[scores.index(min(scores))],legal_moves[scores.index(min(scores))])
        print (" ")
        
        #raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if maximizing_player:
        #Evaluation of the max level    
            scores=[]
            legal_moves=game.get_legal_moves(self)
            if len(legal_moves)==0:
                return (self.score(game,self),(-1,-1))
            if depth==0:
                return(self.score(game,self),(-1,-1))   
            else:
                for move in legal_moves:
                    alphabeta_result=self.alphabeta(game.forecast_move(move),depth-1,alpha,beta,not maximizing_player)
                    if not scores:
                        scores.append(alphabeta_result[0])
                        if alphabeta_result[0]>alpha:
                            alpha=alphabeta_result[0]
                    else:
                        if alphabeta_result[0]>max(scores) and alphabeta_result[0]>alpha:
                            alpha=alphabeta_result[0]
                        scores.append(alphabeta_result[0])    
                    if alpha>=beta:
                        break    
            return(scores[scores.index(max(scores))],legal_moves[scores.index(max(scores))])      
        else:
	    #Evaluation of the min level
            scores=[]
            legal_moves=game.get_legal_moves(game.get_opponent(self))
            if len(legal_moves)==0:
                return (self.score(game,self),(-1,-1))
            if depth==0:
                return(self.score(game,self),(-1,-1)) 
            else:
                for move in legal_moves:
                    alphabeta_result=self.alphabeta(game.forecast_move(move),depth-1,alpha,beta,not maximizing_player)
                    if not scores:
                        scores.append(alphabeta_result[0])
                        if alphabeta_result[0]<beta:
                            beta=alphabeta_result[0]
                    else:
                        if alphabeta_result[0]<min(scores) and alphabeta_result[0]<beta:
                            beta=alphabeta_result[0]
                        scores.append(alphabeta_result[0])  
                    if alpha>=beta:
                        break
            return(scores[scores.index(min(scores))],legal_moves[scores.index(min(scores))])
            
       
        #raise NotImplementedError
    

