"""
## Building a Tic-Tac-Toe agent.

### PRIMARY REQUIREMENTS
The user can be identified via it's user_id.
- A visual depiction of the ongoing game.
    - Shows the score of both the players.
    - Shows the `hint` button, for the challenger to our agent.
    - Shows the current state of the board.
    - Option to choose 'x' or 'o', before he starts his own game.

- An `update_board` function that receives moves as parameters and updates
    the `board` dictionary.

- A `board` dictionary containing configuration and state variables.

- An `agent` function.


### SECONDARY REQUIREMENTS
- User authentication.
    - User must be signed in.
    - User profile.
        - His profile pic.
        - His name.
        - His previous plays' summary.
        - His current standing in leaderboard.
        - His win/loss/draw counts (for both 'x' or 'o').
"""
import random

# Load the previous users and their models using pickle
# - key: user_id (int b/w 0 and 100, both inclusive)
# - value: model of the user
#model_file = r'C:\Users\113121\Programming\MachineLearningProjects\NLP\Pro25022022\models.pkl'
#user_models = pickle.load(model_file) 
# NOTE: remember to user pickle.dump() for updated version of model
# for previously 'seen' opponent, or new model of a new user.

# NOTE: Current undertakings:
# - layout the board in the terminal, as per chosen size ✅
# - Just give the application user the ability to draw 'x's and 'o's on the board ❌
# - provide an agent that plays randomly ❌
# - Revolt against illegal moves, and conclude win/loss/draw, based on tic-tac-toe rules ❌

board = {
    "size": None,
    "user_choice": '',
    "agent_choice": '',
    "curr_player": "agent",
    "curr_move": None,
    "filled_pos": {
        "x": [],
        "o": []
    }     
}

def _get_available_positions():
    all_pos = [tuple(i, j) for i in range(board["size"]) for j in range(board["size"])]
    all_filled_pos = []
    for val in board["filled_pos"].values():
        all_filled_pos.extend(val)
    return set(all_pos) - set(all_filled_pos)

def agent():
    """Our RL-powered AI, tic-tac-toe playing agent.
    
    It can :
    - perceive the state of the board {read the global `dict`}
    - See the opponent's past moves.
        (given the rules of the game, half of total squares are at max filled by one player)
    - Takes action {updates the `curr_move` in global `dict`}
    - Calls the `update_board()`, with draw=True.
    - When planning using lookahead simulations -> calls `update_board()`, with draw=False.

    """

    global board
    move = None
    try:
        assert board["curr_player"] == "agent"
    except AssertionError:
        raise Exception("[ERROR]: some logic error is there in implementation !!")
    
    size = board["size"]
    filled_by_agent = board["filled_pos"][board["agent_choice"]]
    filled_by_user = board["filled_pos"][board["user_choice"]]

    # PLANNING ALGORITHMS :
    # Lookahead for replies and counter-replies
    # The order in which algorithms are written is quite important
    # The first one to work just fine logically, will terminate the function
    # and the later algorithms will not be evaluated, even though they can
    # be better suited to solve the particular instance of the problem.
    #
    # 1. To check if agent can win in its current chance !!
    #
    # Can check if board has:
    # - either `size-1` filled rows in the same column,
    #   or `size-1` filled columns in the same row. ✅
    # - we can also have 2 entries filled in diagonally.  ❌
    # and then check the vacancy of the third in the same
    # row/column/diagonal.
    #
    # rows
    counts = []
    for i in range(size):
        counts.append((i, len([val for val in filled_by_agent if val[0] == i])))
    counts = sorted(counts, key=lambda tup: tup[1], reverse=True)
    
    for i in range(len(counts)):
        row = counts[i][0]
        t = [val for val in list(_get_available_positions()) if val[0] == row]
        if len(t) != 0:
            move = random.choice(t)
            break
    
    if move is not None:
        update_board(move=move, draw=True)
        return
    
    # cols
    counts = []
    for i in range(size):
        counts.append((i, len([val for val in filled_by_agent if val[1] == i])))
    counts = sorted(counts, key=lambda tup: tup[1], reverse=True)
    
    for i in range(len(counts)):
        col = counts[i][0]
        t = [val for val in list(_get_available_positions()) if val[1] == col]
        if len(t) != 0:
            move = random.choice(t)
            break
    
    if move is not None:
        update_board(move=move, draw=True)
        return
    
    # 2. To check if opponent is winning in the next playing chance it gets !!
    # Kind of lazy way to be aware of the situation as and when it arrives.
    counts_rows = []
    for i in range(size):
        counts_rows.append((i, len([val for val in filled_by_user if val[0] == i])))
    counts_cols = []
    for i in range(size):
        counts_cols.append((i, len([val for val in filled_by_user if val[1] == i])))
    
    if len(counts_rows) != 0:
        index_row = max(counts_rows, key=lambda tup: tup[1])
    if len(counts_cols) != 0:        
        index_col = max(counts_cols, key=lambda tup: tup[1])
    if index_row[1] > index_col[1]:
        if index_row[1] in set([size-3, size-2, size-1]):
            # Uh-oh, if agent is not vigilant now, the agent looses in next, or next few chances !!
            t = [val for val in list(_get_available_positions()) if val[0] == row]
            if len(t) != 0:
                move = random.choice(t)
    else:
        if index_col[1] in set([size-3, size-2, size-1]):
            # Uh-oh, if agent is not vigilant now, the agent looses in next, or next few chances !!
            t = [val for val in list(_get_available_positions()) if val[1] == col]
            if len(t) != 0:
                move = random.choice(t)
    


def update_board(move=None, draw=True):
    """Update the state of the board.
    
    Parameters
    ----------
    draw : bool
        this helps in deciding whether this function is called during
        actual gameplay, or during simulation of future game-plays by
        our AI agent, in deciding its own move (PLANNING).
    """
    global board

    if move is None:
        if draw:
            draw_board()
        return
    
    # check for the validity of the move
    if move not in _get_available_positions():
        print("\n[INVALID MOVE]: the position you want to fill is is not available. TRY AGAIN!\n")

    if board["curr_player"] == "user":
        board["filled_pos"][board["user_choice"]].append(move)
        board["curr_player"] = "agent"
        draw_board()
    else:
        board["filled_pos"][board["agent_choice"]].append(move)
        board["curr_player"] = "user"
        if draw: # agent can call upon `update_board()` when running simulations during planning strategy
            draw_board()

def _horizontal_lines(size):
    return " ---" * size

def _vertical_lines(size):
    return "|   " * (size+1)

def draw_board(init=False):
    """Draw the board in the terminal."""
    
    global board
    print("*" * 20, " WELCOME TO THE REALM OF Xs and Os ", "*" * 20)
    print("-" * (20 + 1 + 33 + 1 + 20), "\n")

    # Initialize the board
    if init:
        choices = set(["x", "o"])
        choice = str(input(
            "Do you choose 'x' or 'o' \
(HELP: enter x for cross and o for knot): "))

        size = int(input(
            "Enter the size of the board you desire \
(HELP: enter 3 for 3x3, 4 for 4x4 etc.): "))
        for _ in range(size):
            print(_horizontal_lines(size))
            print(_vertical_lines(size))
        print(_horizontal_lines(size))
        
        board.setdefault("size", size) # how diff from board["size"] = size
        board.setdefault("user_choice", choice)
        board.setdefault("agent_choice", list(choices - set(choice))[0])

if __name__ == '__main__':
    # Game play goes here !!
    draw_board(init=True)
    
    val = random.choice([0, 1])
    
    if val == 1:
        board["curr_player"] = "agent"
        agent()
        print("\nAGENT MOVED FIRST !!\n")
    else:
        board["curr_player"] = "user"
        print("YOU HAVE TO MOVE FIRST")

    while True:
        print(f"Which position you want to put {board.get('user_choice', '')} in ?\n")

        update_board(move=None, draw=True)