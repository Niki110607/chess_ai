import torch
import numpy as np
import chess
import time
from collections import OrderedDict

from data.Data_processing import process_test_data, get_all_moves
from model.architecture import ChessTransformer_conv
import os
from pathlib import Path


class Node:
    def __init__(self, prior, parent=None):

        self.N : int = 0
        self.W : float = 0.0
        self.prior : float = prior
        self.value : float|None = None
        self.color : float= 0
        self.board = None

        self.input = None
        self.legal_moves = None
        self.move_dist = None

        self.parent : Node = parent
        self.children : list[Node] = []
        self.children_moves : list[int] = []

        self.leaf_initialised : bool = False
        self.children_created : bool = False
        self.is_terminal : bool = False
        self.is_processing : bool = False


    def _init_leaf_node(self, board, move_to_idx):
        """
        This function first initializes a certain leaf with a dummy value and move distribution, so other iterations can move past it before the evaluation.
        Additionally it calls the data processing function and gets the input as a bitboard.
        """
        self.leaf_initialised = True
        self.is_processing = True
        self.board = board
        #self.color = 1 if it is whites turn and -1 otherwise
        self.color = (int(board.turn) * 2) - 1

        input, legal_moves = process_test_data(self.board, move_to_idx)
        self.input = input.float()
        self.legal_moves = legal_moves.float()

        self.value = 0.0
        self.move_dist = torch.zeros_like(self.legal_moves) + self.legal_moves


    def _update_with_real_value(self, real_value, real_move_dist, move_to_idx):
        """
        The dummy values and move distributions get replaced by the calculated results from the transformer model.
        """
        self.value = real_value
        self.move_dist = real_move_dist

        self._improve_move_dist(False, move_to_idx)

        valid_children = []
        valid_children_moves = []

        for child, move_idx in zip(self.children, self.children_moves):
            prior = self.move_dist[move_idx].item()
            if prior > 0.01:
                child.prior = prior
                valid_children.append(child)
                valid_children_moves.append(move_idx)

        self.children = valid_children
        self.children_moves = valid_children_moves
        self.is_processing = False

        self.input = None
        self.legal_moves = None


    def _improve_move_dist(self, is_dummy, move_to_idx):
        """
        The move distribution gets improved by adding temperature (dividing the move distribution by a number > 1), so that moves with small probabilities also get a chance.
        Also checks and captures are getting a min value, so they are definitely in contention.
        """
        if not is_dummy:
            self.move_dist /= 1.5

            if self.parent is None:
                for move in self.board.legal_moves:
                    idx = move_to_idx[str(move)[:4]]

                    if self.move_dist[idx] < -10:
                        continue

                    if self.board.gives_check(move):
                        self.move_dist[idx] = max(self.move_dist[idx], 0.0)

                    if self.board.is_capture(move):
                        self.move_dist[idx] = max(self.move_dist[idx], -0.5)

            self.value = float((torch.sigmoid(torch.tensor(self.value)) * 2) - 1)

        self.move_dist = torch.nn.functional.softmax(self.move_dist, dim=0)


    def _create_dummy_children(self):
        """
        For every legal move a dummy child gets created and later updated in self._update_with_real_value().
        """
        self.children_created = True

        #removes all illegal moves and creates a list of the indices of all legal moves
        valid_indices = torch.nonzero(self.legal_moves + 1e9).squeeze(-1).tolist()
        moves_count = len(valid_indices)

        if moves_count > 0:
            prob = 1 / moves_count

            for ind in valid_indices:
                self.children.append(Node(prob, self))
                self.children_moves.append(ind)


    def _choose_best_child(self):
        """
        For every child the Q and U values are calculated and added together and the child with the highest value gets chosen as the continuation of the current path.
        """
        move_values = []

        for child in self.children:
            #Q rates how good a move is: Total value divided by the count
            child.Q = child.W/(child.N + 1)
            #U is the exploration part. It encourages the model to try new paths
            child.U = 3.0  * child.prior * (np.sqrt(self.N) / (child.N + 1))

            move_value = child.Q + child.U.item()
            move_values.append(move_value)

        best_child_idx = np.argmax(move_values)
        best_move = self.children_moves[best_child_idx]

        return best_move, best_child_idx





class MCTS:
    def __init__(self):
        self.current_node = Node(1)
        self.current_board = chess.Board()
        self.batch_size = 128
        self.game_started = False

        self.all_moves, self.move_to_idx = get_all_moves()
        self.model, self.device = self._init_model()

        self.move_objects = [chess.Move.from_uci(move_str) if move_str[:2] != move_str[2:4] else "0000" for move_str in self.all_moves]


    def _init_model(self):
        """
        The model is created, transferred to gpu and feed with the trained parameters.
        """
        device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

        model = ChessTransformer_conv(
            d_model=512,
            nhead=16,
            num_layers=8,
            d_ff=1024,
            dropout=0.1
        ).to(device)

        current_dir = Path(__file__).resolve().parent
        weights_path = current_dir.parent / "model" / "chessformer_model_weights.pth"
        state_dict = torch.load(weights_path, map_location=torch.device(device))

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

        return model, device


    def play_game(self, start_pos=None):
        """
        This function starts a game. You can begin with a custom position or just the standard board.
        With .play_move() you can play a certain move and with .evaluate_position() you can evaluate the current position.
        By analyzing positions this way, the evaluation is more efficient because you may have already calculated some nodes on the previous moves.
        """
        self.game_started = True
        self.current_node = Node(1)
        self.current_board = chess.Board()

        if start_pos is not None:
            self.current_board = chess.Board(start_pos)

        return self.current_board


    def reset_board(self):
        while self.current_node.parent is not None:
            self.current_node = self.current_node.parent
        self.current_board = chess.Board()


    def play_move(self, move_str):
        """
        Intakes a uci-move string like "e2e4" and updates the board. It also checks if the game is over.
        """
        repetition = self.current_board.is_repetition()
        fifty_moves = self.current_board.is_fifty_moves()

        if self.current_board.result() != "*" or repetition or fifty_moves:
            self.game_started = False
            result = self.current_board.result()

            if self.current_board.result() == "*":
                result = "1/2-1/2"

            print(f"The game is already over. The final score is: {result}")
            return self.current_board

        move = chess.Move.from_uci(str(move_str))
        mirrored_move = chess.Move(
            from_square=chess.square_mirror(move.from_square),
            to_square=chess.square_mirror(move.to_square),
            promotion=move.promotion
        )

        if move in self.current_board.legal_moves:
            self.current_board.push(move)

        else:
            print(f"{move_str} is not a valid move in this position!")
            return self.current_board

        if self.current_board.turn == chess.WHITE:
            move_idx = self.move_to_idx[str(mirrored_move)[:4]]

        else:
            move_idx = self.move_to_idx[str(move)[:4]]

        if move_idx in self.current_node.children_moves:
            child_idx = self.current_node.children_moves.index(move_idx)
            self.current_node = self.current_node.children[child_idx]

        else:
            self.current_node = Node(1)

        return self.current_board


    def take_move_back(self):
        """
        Takes back the last played move.
        """
        if self.current_node.parent is None:
            try:
                self.current_board = self.current_board.pop()
                self.current_node = Node(1)
            except:
                print("There has been no move played yet!")

        else:
            self.current_board = self.current_board.pop()
            self.current_node = self.current_node.parent


    def evaluate_position(self, fen=None, max_time=5, delta=1):
        """
        This function evaluates a given position and returns both the best move and the evaluation.
        :param fen: Can be used to evaluate a custom position else the current position from an ongoing game will be evaluated.
        :param max_time: If the model found a clear best move after this time it stops the search.
        :param delta: If the model is uncertain about the best move after max_time it can expand the search for delta seconds, but has to return the best move after this time.
        """
        if fen is not None:
            self.current_board = chess.Board(fen)

        best_move = self._run(max_time, delta)
        evaluation = self.current_node.W / (self.current_node.N + 1)
        evaluation *= int(not self.current_board.turn) * 2 - 1

        if evaluation > 1.0:
            evaluation = 1.0
        elif evaluation < -1.0:
            evaluation = -1.0

        return best_move, evaluation


    def _run(self, max_time, delta):
        """
        Controls the whole algorithm. It calls forward, leaf evaluation and the backward function and manages the batch process and the time.
        """
        time_start = time.time()

        for i in range(1000):

            time_stop = time.time()
            time_delta = time_stop - time_start

            if time_delta > max_time + delta:
                break

            elif time_delta > max_time:
                N_values = sorted([child.N for child in self.current_node.children])
                if len(N_values) > 1:
                    N_margin = N_values[-1] / N_values[-2]
                else:
                    best_move = self._evaluate_best_move()
                    return best_move

                if N_margin > 1.2:
                    break

            paths = []
            leaves_to_evaluate = []

            for _ in range(self.batch_size):

                node = self._forward()
                paths.append(node)

                if node.is_processing and not node.is_terminal and node not in leaves_to_evaluate:
                    leaves_to_evaluate.append(node)

            if leaves_to_evaluate:
                self._calc_leaf_nodes(leaves_to_evaluate)

            for node in paths:
                self._backward(node)

        best_move = self._evaluate_best_move()

        self._print_infos()

        return best_move


    def _forward(self):
        """
        This function moves through the game tree until it reaches either an unexplored leaf or a terminated state.
        """
        board = self.current_board.copy()
        node = self.current_node

        for i in range(1000):
            node.N += 1
            node.W -= 0.3

            if node.is_terminal:
                break

            if not node.leaf_initialised:
                node._init_leaf_node(board, self.move_to_idx)
                node._improve_move_dist(True, self.move_to_idx)
                node._create_dummy_children()
                break

            if len(node.children) == 0:
                break

            move, best_child_idx = node._choose_best_child()
            node = node.children[best_child_idx]
            move_uci = self.move_objects[move]

            if board.turn == chess.BLACK:
                move_uci = chess.Move(
                    from_square=chess.square_mirror(move_uci.from_square),
                    to_square=chess.square_mirror(move_uci.to_square),
                    promotion=move_uci.promotion
                )

            if move_uci not in board.legal_moves:
                move_uci = chess.Move(move_uci.from_square, move_uci.to_square, promotion=chess.QUEEN)

            board.push(move_uci)

            repetition = board.is_repetition()
            fifty_moves = board.is_fifty_moves()

            if board.result() != "*" or repetition or fifty_moves:
                node.N += 1
                node.W -= 0.3
                node.color = (int(board.turn) * 2) - 1
                node.board = board

                node.leaf_initialised = True
                node.is_terminal = True

                discount_factor = 0.99 ** i

                if board.result() == "1-0":
                    node.value = 1.2 * discount_factor

                elif board.result() == "0-1":
                    node.value = -1.2 * discount_factor

                elif board.result() == "1/2-1/2" or repetition or fifty_moves:
                    node.value = 0.0

                break

        return node


    def _calc_leaf_nodes(self, nodes_batch):
        """
        The values and move distributions of the nodes are calculated.
        """
        nodes_input = [node.input for node in nodes_batch]
        nodes_legal_moves = [node.legal_moves for node in nodes_batch]

        nodes_input_tensor = torch.stack(nodes_input).to(self.device)
        nodes_lm_tensor = torch.stack(nodes_legal_moves).to(self.device)

        self.model.eval()
        with torch.no_grad():
            value, move_dist = self.model(nodes_input_tensor, nodes_lm_tensor)
            move_dist = move_dist.detach().cpu()

        for idx, node in enumerate(nodes_batch):
            move_dist_i = move_dist[idx]
            value_i = value[idx].item()
            node._update_with_real_value(value_i, move_dist_i, self.move_to_idx)


    def _backward(self, node):
        """
        The nodes are moving backwards through the tree and are correcting the previous penalty and adding newly discovered values.
        """
        value = node.value
        while node is not None:
            node.W += 0.3
            node.W += -node.color * value
            node = node.parent


    def _evaluate_best_move(self):
        """
        The final best move gets evaluated. This works by just picking the most visited child move from the root node.
        """
        N_values_children = [child.N for child in self.current_node.children]
        most_visited_child_idx = np.argmax(N_values_children)
        best_move_idx = self.current_node.children_moves[most_visited_child_idx]
        best_move_uci = self.all_moves[best_move_idx]
        best_move = chess.Move.from_uci(best_move_uci)

        if self.current_node.color == -1:
            best_move = chess.Move(
                    from_square=chess.square_mirror(best_move.from_square),
                    to_square=chess.square_mirror(best_move.to_square),
                    promotion=best_move.promotion
                )

        if best_move not in self.current_board.legal_moves:
            print("promotion")
            best_move = chess.Move(best_move.from_square, best_move.to_square, promotion=chess.QUEEN)

        return best_move




















    def _print_infos(self):
        N_values_children = [child.N for child in self.current_node.children]
        prior_children = [child.prior for child in self.current_node.children]
        children_best_moves = [self.all_moves[idx] for idx in self.current_node.children_moves]
        print(f"best moves: {children_best_moves}")
        print(f"probabilities: {prior_children}")
        print(f"visit count: {N_values_children}")
        print(f"total nodes visited: {self.current_node.N}")
        print(f"evaluation: {self.current_node.W / (self.current_node.N + 1)}")


























