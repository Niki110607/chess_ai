import berserk
import chess
from tree_search.mcts import MCTS


engine = MCTS()


def get_bot_move(board, game_state, my_color):
    """
    Extracts the best move in the given position from my Monte Carlo tree search.
    """
    last_move = board.peek() if board.move_stack else None

    think_limit = calculate_thinking_time(board, game_state, my_color)

    if last_move is not None:
        engine.play_move(last_move)

    best_move, _ = engine.evaluate_position(
        max_time=think_limit,
        delta=0
    )

    engine.play_move(best_move)

    return best_move


def calculate_thinking_time(board, game_state, my_color):
    if my_color == "white":
        time_left_ms = game_state["wtime"]
        raw_inc = state["winc"]

    else:
        time_left_ms = game_state["btime"]
        raw_inc = state["winc"]

    time_left_s = normalize_time(time_left_ms)
    inc_s = normalize_time(raw_inc)

    move_num = board.fullmove_number

    moves_to_go = max(10, 50-move_num)

    think_time = ((time_left_s-inc_s) / moves_to_go) + (inc_s * 0.8)
    hard_limit = time_left_s - 0.5
    think_limit = min(hard_limit, think_time)

    return think_limit



def normalize_time(t):
    if isinstance(t, int):
        return t / 1000
    return t.total_seconds()


session = berserk.TokenSession()
client = berserk.Client(session=session)

print("Bot is listening for games...")

for event in client.bots.stream_incoming_events():

    if event["type"] == "challenge":
        if event["challenge"].get("direction") == "in":
            challenge_id = event['challenge']['id']
            client.bots.accept_challenge(challenge_id)
            print(f"Accepted incoming challenge {challenge_id}")
        else:
            print(f"Sent outgoing challenge to {event['challenge']['destUser']['id']}. Waiting for them to accept...")

    elif event["type"] == 'challengeDeclined':
        declined_info = event['challenge']
        reason = declined_info.get('declineReason', 'No reason provided')
        opponent = declined_info.get('destUser', {}).get('id', 'Unknown')
        print(f"❌ Challenge DECLINED by {opponent}. Reason: {reason}")

    elif event["type"] == "gameStart":
        game_id = event["game"]["id"]
        my_color = "white" if event["game"]["color"] == "white" else "black"
        board = chess.Board()
        engine.play_game()

        for game_event in client.bots.stream_game_state(game_id):

            if game_event["type"] == "gameFull":
                state = game_event["state"]

            elif game_event["type"] == "gameState":
                state = game_event

            else:
                continue

            moves = state["moves"].split()
            board = chess.Board()
            for m in moves:
                board.push_uci(m)

            if board.turn == (chess.WHITE if my_color == "white" else chess.BLACK):

                best_move = get_bot_move(board, state, my_color)

                client.bots.make_move(game_id, best_move.uci())
































