# Agent Instructions for Wikitext-2 MoE Experiment

This document outlines the structure and implementation plan for the Wikitext-2 language modeling experiment.

## Project Structure

- `scripts/`: Contains all Python source code.
  - `wikitext2_train.py`: The main script for training models. Handles command-line arguments for model type, hyperparameters, etc.
  - `wikitext2_eval.py`: The main script for evaluating trained models. Measures perplexity, latency, memory, and handles model exporting.
  - `data.py`: Contains functions for loading, preprocessing, and batching the Wikitext-2 dataset.
  - `model.py`: Defines all `torch.nn.Module` classes, including the base Transformer, the `FlashMoeLayer`, and the complete language models (`FlashMoEModel`, `DenseModel`).
  - `utils.py`: A collection of helper functions for things like:
    - Reproducibility (setting seeds).
    - FLOPs calculation.
    - Logging configuration.
    - Router statistics analysis.

- `artifacts/`: Stores all outputs from training and evaluation.
  - `wikitext2/{model_name}/`: Each model variant will have its own directory containing:
    - Model checkpoints (`.pt` files).
    - Training and evaluation logs (`.csv` files).
    - Exported models (`.pt` for TorchScript, `.onnx` for ONNX).
    - Evaluation reports.

- `requirements.txt`: Lists all Python dependencies.

## Implementation Plan

1.  **`scripts/utils.py`**:
    - Implement `set_seed` to ensure reproducibility.
    - Implement FLOPs calculation helpers: `calculate_flashmoe_flops` and `calculate_dense_flops`.
    - Implement `calculate_dense_hidden_dim` to find the equivalent hidden dimension for the dense baseline.

2.  **`scripts/data.py`**:
    - Use `datasets.load_dataset` to get Wikitext-2.
    - Use a pre-trained tokenizer from `transformers` (e.g., `GPT2Tokenizer`) for consistency and a reasonable vocabulary size.
    - Create a `DataHandler` class or functions to manage tokenization, sequence creation, and batching. The data should be flattened and then reshaped into `(batch_size, seq_len)` batches.

3.  **`scripts/model.py`**:
    - **`FlashMoeLayer`**: This is the most complex module.
      - It will handle routing (`top_k`), expert execution, and combining results.
      - It will compute and return auxiliary information for the load balancing loss.
      - The `forward` pass will be carefully implemented to handle token dispatching and capacity limits. Tokens routed to an expert beyond its capacity will be dropped (simplest approach first).
    - **Base Transformer**: A standard decoder-only Transformer architecture will be implemented, but the FFN block of each layer will be configurable.
    - **Model Factories**: Functions will create the three required models:
      - `FlashMoEModel`: Transformer with `FlashMoeLayer`.
      - `DenseModel`: Transformer with a standard `nn.Sequential` FFN, sized using the utility functions.
      - `SwitchModel`: A configuration of `FlashMoEModel` with `top_k=1`.

4.  **`scripts/wikitext2_train.py`**:
    - Use `argparse` to handle all command-line arguments as specified in the prompt.
    - The main function will orchestrate:
      - Setting the seed.
      - Loading the data.
      - Initializing the correct model based on CLI args.
      - Setting up the AdamW optimizer and a learning rate scheduler.
      - The training loop, which calls `train_one_epoch` and `evaluate` functions.
      - Logging metrics (train/val loss, perplexity, router stats) to both the console and a CSV file.
      - Saving model checkpoints every epoch.

5.  **`scripts/wikitext2_eval.py`**:
    - Use `argparse` to load a model from a checkpoint.
    - Perform the following evaluations:
      - **Perplexity**: On the validation set.
      - **Latency**: Measure wall-clock time for a forward pass (with warm-up).
      - **Memory**: Use `psutil` to measure peak RSS during a forward pass.
      - **FLOPs**: Report the calculated theoretical FLOPs.
      - **Router Analysis**: Compute and display expert utilization histograms.
      - **Export**: Implement `torch.jit.trace_module` and `torch.onnx.export` for the specified `forward_export` method.

## Agent Directives

- **CPU Only**: Ensure no code attempts to move models or tensors to a CUDA device.
- **Reproducibility**: The `set_seed` function must be called at the start of `wikitext2_train.py`.
- **Clarity**: Code should be well-commented, especially the `FlashMoeLayer`. Console output should be informative.
- **Verification**: After implementing each major component, I will write simple checks to ensure it behaves as expected before moving to the next. For example, after implementing the `FlashMoeLayer`, I'll test its forward pass with dummy data.
