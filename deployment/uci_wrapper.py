import sys
import chess
from tree_search.mcts import MCTS
import math

def main():
    board = chess.Board()
    engine = MCTS()
    first_position = True

    for line in sys.stdin:
        command = line.strip()

        if command == "uci":
            print("id name MyPythonEngine 1.0")
            print("id author Niklas")
            print("uciok")
            sys.stdout.flush()

        elif command == "ucinewgame":
            engine.play_game()
            first_position = True
            print("New game has started")

        elif command == "isready":
            print("readyok")
            sys.stdout.flush()

        elif command.startswith("position"):
            engine.reset_board()
            if "startpos" in command:
                board = chess.Board()

            elif "fen" in command:
                fen_str = command.split("fen ")[1].split(" moves")[0]
                board = chess.Board(fen_str)
                if first_position:
                    engine.current_board = board

            if "moves " in command:
                moves_list = command.split("moves ")[1].split()
                for m in moves_list:
                    move_obj = chess.Move.from_uci(m)
                    board.push(move_obj)
                    engine.play_move(move_obj)
            first_position = False

        elif command.startswith("go"):
            params = command.split()
            time_data = {}
            for i in range(1, len(params), 2):
                time_data[params[i]] = int(params[i+1])

            if board.turn == chess.WHITE:
                my_time = time_data.get('wtime', 1000)
                my_inc = time_data.get('winc', 0)
            else:
                my_time = time_data.get('btime', 1000)
                my_inc = time_data.get('binc', 0)

            think_time = calculate_thinking_time(board, my_time, my_inc)

            last_move = board.peek() if board.move_stack else None

            #if last_move is not None:
                #engine.play_move(last_move)

            best_move, evaluation = engine.evaluate_position(
                max_time=max(think_time, 0),
                delta=0
            )

            if board.turn == chess.BLACK:
                evaluation = -evaluation

            evaluation = (evaluation + 1) / 2



            if evaluation == 1.0:
                evaluation -= 0.001
            elif evaluation == 0.0:
                evaluation += 0.001

            print(evaluation)

            score_cp = -400 * (math.log10((1 / evaluation) - 1))
            print(f"info nodes {engine.current_node.N} score cp {int(score_cp)}")

            engine.play_move(best_move)
            print(f"bestmove {best_move}")

            sys.stdout.flush()

        elif command == "quit":
            break


def calculate_thinking_time(board, my_time, my_inc):
    my_time /= 1000
    my_inc /= 1000

    move_num = board.fullmove_number

    moves_to_go = max(10, 60-move_num)

    think_time = ((my_time-(my_inc+0.1)) / moves_to_go) + (my_inc * 0.8)
    hard_limit = 0.5 * my_time
    think_limit = min(hard_limit, think_time)
    #if move_num <= 5:
        #think_limit = 1.0
    print(f"info | current time {my_time}, increment {my_inc}, thinking time {think_time}")

    return think_limit



def normalize_time(t):
    if isinstance(t, int):
        return t / 1000
    return t.total_seconds()


if __name__ == "__main__":
    main()