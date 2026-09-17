# Chessformer

A transformer-based chess engine that combines a neural network policy and position evaluation with Monte Carlo Tree Search (MCTS).

The engine is inspired by the transformer-based chess research presented in [*Mastering Chess with a Transformer Model*](https://arxiv.org/abs/2409.12272).

It was trained on approximately **10 million Lichess chess positions**, using strong **Stockfish evaluations** as training labels. The resulting model predicts both a move distribution and a scalar evaluation of the current position. MCTS then uses these predictions to selectively explore the game tree.

## Architecture

The engine uses a hybrid **convolutional + transformer architecture** implemented in **PyTorch**.

The board is represented as a **19 × 64 tensor**, with the 19 input features describing the chess position. The model first processes these features using convolutional layers before passing the resulting 64 board-square representations through a transformer encoder.

The current configuration uses:

* **Input:** 19 × 64 board tensor
* **Convolutional layers:** 19 → 64 → 128 → 512 channels
* **Transformer dimension:** 512
* **Attention heads:** 16
* **Transformer layers:** 8
* **Feed-forward dimension:** 1024
* **Activation:** GELU
* **Dropout:** 0.1
* **Framework:** PyTorch

The engine uses `python-chess` for board representation, legal move generation, and game-state handling.

### Move Policy Head

The policy head produces logits for a large fixed set of possible chess moves. A legal-move mask removes illegal moves before the distribution is used by the search.

The resulting policy provides **prior probabilities** for MCTS, allowing the search to concentrate on moves that the neural network considers promising.

During search, these priors are combined with visit statistics and an exploration term to select the next branch to investigate.

### Value Evaluation Head

The value head takes the transformer's `[CLS]` representation and produces a **single scalar evaluation** for the position.

This value is used by MCTS when evaluating unexplored positions and is propagated backwards through the search tree. Terminal positions are handled separately by the search.

## Move Representation

Moves are represented using a fixed collection of possible move vectors.

`python-chess` is used to generate the legal moves in a position. A legal-move mask is then applied to the model's move logits so that only legal moves can participate in the policy distribution.

## Monte Carlo Tree Search

The neural network is combined with MCTS to improve move selection.

Each search iteration:

1. Traverses the existing tree using the MCTS selection formula.
2. Creates or reaches an unexplored leaf.
3. Evaluates leaf positions in batches using the transformer.
4. Expands the node using the predicted move distribution.
5. Backpropagates the predicted value through the selected path.

The implementation evaluates leaves in **batches of 16**, allowing multiple positions to be processed by the neural network simultaneously.
Child selection combines:

* **Q:** the accumulated value of a child
* **U:** an exploration term based on the neural network's prior probability

The final move is selected from the root by choosing the child with the **highest visit count**.

## Training

The model was trained using approximately **10 million positions from Lichess**.

Training targets are generated from **strong Stockfish evaluations**, allowing the network to learn both:

* which moves are likely to be strong
* how favorable a position is

## Playing Strength

The engine has an estimated playing strength of approximately **2200–2400 Elo**, depending on thinking time and search configuration.

This is an approximate estimate rather than an official FIDE rating. Engine strength can vary substantially with hardware, search time, and testing conditions.

## Evaluation

The engine converts its normalized position evaluation into an approximate **centipawn-style score** using a logarithmic transformation. This allows the neural evaluation to be presented in a familiar chess-engine format.

## Reference

This project is based in part on ideas from:

**Mastering Chess with a Transformer Model**
arXiv:2409.12272

https://arxiv.org/abs/2409.12272
