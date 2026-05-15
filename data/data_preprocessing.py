import torch
import numpy as np
from datasets import load_dataset
from Data_processing import process_training_data, pad_legal_moves

def preprocess_and_save(chunk_size=100000, total_samples=10000000):
    raw_data = load_dataset("bingbangboom/stockfish-evaluation-SAN", streaming=True, split="train")

    inputs_chunk = []
    labels_chunk = []
    best_move_chunk = []
    legal_moves_chunk = []

    count = 0
    chunk_num = 0

    for row in raw_data:
        if count >= total_samples:
            break

        try:
            processed_data = process_training_data(row)
            if processed_data is None:
                continue

            input, label, best, legal = processed_data

            inputs_chunk.append(input.to(torch.uint8))
            labels_chunk.append(label)
            best_move_chunk.append(best)
            legal_moves_chunk.append(legal)

            count += 1

            if count % chunk_size == 0:
                save_path = f"preprocessed_data/{chunk_num}.pt"

                legal_moves_chunk = pad_legal_moves(legal_moves_chunk)

                payload = {
                    "x": torch.tensor(np.array(inputs_chunk)),
                    "y": torch.tensor(np.array(labels_chunk)),
                    "best": torch.tensor(np.array(best_move_chunk)),
                    "legal": legal_moves_chunk
                }
                print(torch.tensor(np.array(inputs_chunk)).shape, torch.tensor(np.array(labels_chunk)).shape, torch.tensor(np.array(best_move_chunk)).shape, torch.tensor(np.array(legal_moves_chunk)).shape)
                torch.save(payload, save_path)
                print(f"Saved {save_path}")

                inputs_chunk = []
                labels_chunk = []
                best_move_chunk = []
                legal_moves_chunk = []
                chunk_num += 1

        except Exception as e:
            continue

preprocess_and_save()