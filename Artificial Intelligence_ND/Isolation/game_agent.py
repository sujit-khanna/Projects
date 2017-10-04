"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # percentage of my legal moves versus twice the opponents legal moves left
    my_moves_left = len(game.get_legal_moves(player)) / len(game.get_blank_spaces())
    player_moves_left = len(game.get_legal_moves(game.get_opponent(player))) / len(game.get_blank_spaces())
    return float(my_moves_left - (player_moves_left * 2))

    #raise NotImplementedError


def custom_score_2(game, player):
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
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Player's i.e. opponents moves are scored more than ours
    my_weight = 0.1
    player_weight = 1 - my_weight
    
    my_moves = len(game.get_legal_moves(player))
    player_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float((my_moves * my_weight) - (player_moves * player_weight))



def custom_score_3(game, player):
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
    if game.is_winner(player):
        return float('inf')
    if game.is_loser(player):
        return float('-inf')

    my_moves = len(game.get_legal_moves(player))
    player_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Award additional score for player's centraility position, central positions have a better probability of winning the game
    
    centrality_bonus = 0
    central_param1, central_param2 = 0.25, 0.75
    
    loc_1, loc_2 = game.get_player_location(player)
    loc_1_pos = float(loc_1) / game.width
    loc_2_pos = float(loc_2) / game.height
    if central_param1 <= loc_1_pos or loc_1_pos <= central_param2:
      centrality_bonus += .5
    if central_param1 <= loc_2_pos or loc_2_pos <= central_param2:
      centrality_bonus += .5

    # reward  when opponent player is not in central positions
    player_loc_1, player_loc_2 = game.get_player_location(game.get_opponent(player))
    player_loc_1_pos = float(player_loc_1) / game.width
    player_loc_2_pos = float(player_loc_2) / game.height
    if central_param1 > player_loc_1_pos or player_loc_1_pos > central_param2:
      centrality_bonus += .5
    if central_param1 > player_loc_2_pos or player_loc_2_pos > central_param2:
      centrality_bonus += .5

    return my_moves - player_moves + centrality_bonus



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        optimal_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return optimal_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        
        max_value = float("-inf")
        # keep track of optimal moves in the game , if no legal moves, then (-1, -1)
        optimal_move = (-1, -1) 
        if len(game.get_legal_moves(self)) >= 1:
            legal_moves = game.get_legal_moves(self)
            optimal_move = legal_moves[0]
        else:
            return optimal_move
        # looping over all legal moves for the player
        for player_move in game.get_legal_moves(self):
            player_game = game.forecast_move(player_move)
            player_value = self.min_value(player_game, depth-1)
            if player_value > max_value:       # if max value is available update optimal_move and max_value
                optimal_move, max_value = player_move, player_value
        return optimal_move

    def max_value(self, game, depth):
        """creating additional function to help in solving the minimax algorithm,
        this function returns max value for all possibilites in the min function
        using the depth
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            # score at leaf node
            return self.score(game, self)
        maximum_val = float("-inf")
        for move in game.get_legal_moves(game.active_player):
            # updating the function value as max of min outcomes
            maximum_val = max(maximum_val, self.min_value(game.forecast_move(move), depth-1))
        return maximum_val
		
		

    def min_value(self, game, depth):
        """creating additional function to help in solving the minimax algorithm,
        this function returns min value for all possibilites in the max function
        using the depth
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            # score at leaf node
            return self.score(game, self)
        minimum_val = float("inf")
        for move in game.get_legal_moves(game.active_player):
            # updating the function value as min of max outcomes
            minimum_val = min(minimum_val, self.max_value(game.forecast_move(move), depth-1))
        return minimum_val


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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
        #raise NotImplementedError


        # set the basecase optimal move
        optimal_move = (-1, -1)
        if len(game.get_legal_moves()) >= 1:
            legal_moves = game.get_legal_moves()
            optimal_move = legal_moves[0]
        else:
            return optimal_move

        try:
            for depth in range(1, len(game.get_blank_spaces())):
                updated_move = self.alphabeta(game, depth)
                if updated_move == ():
                    return optimal_move
                else:
                    optimal_move = updated_move

        except SearchTimeout:
            return optimal_move  # the initial basecase optimal move defined above

        # optimal move found by looping through the last search
        return optimal_move
        
        
        

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # TODO: finish this function!
        #raise NotImplementedError
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        optimal_value = float("-inf")
        optimal_move = ()
        
        #looping over all legal moves for the player
        for player_move in game.get_legal_moves():
            # generate new move
            player_game = game.forecast_move(player_move)
            player_value = self.min_value(player_game, depth-1, alpha, beta)
            # Updating optimal_move and max_value if player_value is available
            if player_value > optimal_value:
                optimal_move, optimal_value = player_move, player_value
            if optimal_value >= beta:
                break
            # finding the base level for pruning
            alpha = max(alpha, optimal_value)
        return optimal_move

    def max_value(self, game, depth, alpha, beta):
        """creating additional function to help in alpha beta pruning,
        this function returns max value for all possibilites in the min function
        using the depth
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        maximum_val = float("-inf")

        for move in game.get_legal_moves():
            maximum_val = max(maximum_val, self.min_value(game.forecast_move(move), depth-1, alpha, beta))
            alpha = max(alpha, maximum_val)
            if maximum_val >= beta:
                return maximum_val
        return maximum_val

    def min_value(self, game, depth, alpha, beta):
        """creating additional function to help in alpha beta pruning,
        this function returns min value for all possibilites in the max function
        using the depth
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)

        minimum_val = float("inf")

        for move in game.get_legal_moves():
            minimum_val = min(minimum_val, self.max_value(game.forecast_move(move), depth-1, alpha, beta))
            if minimum_val <= alpha:
                return minimum_val
            beta = min(beta, minimum_val)
        return minimum_val