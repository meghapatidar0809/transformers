# Transformer Implementations: Multi-Head Attention & KV Caching

### Overview
This project explores various aspects of Transformer models using **PyTorch**. It encompasses two main implementations:
1.  A **full Encoder-Decoder Transformer** designed for sequence-to-sequence learning, emphasizing **multi-head attention** and standard Transformer components.
2.  A **simple 2-layer Decoder-Only Transformer** focused on demonstrating and benchmarking the performance benefits of **Key-Value (KV) caching** during autoregressive token generation.

---

### General Features
* **Multi-Head Attention**: Utilizes multiple attention heads for improved context understanding across various implementations.
* **Positional Encoding**: Incorporates positional information into token embeddings, crucial for sequence-aware processing.
* **Layer Normalization**: Stabilizes training by maintaining consistent feature distributions.
* **Fully Connected Layers**: Enhances feature extraction through linear transformations.
* **Encoder-Decoder Architecture**: Implements a complete Transformer model for sequence-to-sequence tasks (specific to Problem 1).
* **Autoregressive Generation**: The decoder-only model can generate `n` tokens sequentially (specific to Problem 2).
* **KV Caching**: Optimizes autoregressive generation by caching Key and Value states (specific to Problem 2).
* **Logit Validation**: Tools to verify model output correctness.
* **Performance Benchmarking**: Functionality to compare generation latency with and without KV caching.

---

### Tools & Libraries
-   **Python**
-   **PyTorch**: For building neural networks.
-   **NumPy**: For numerical operations.
-   **Matplotlib**: For plotting benchmark results.
-   **Transformers (Hugging Face)**: (Potentially, if used for the full Transformer, otherwise remove this line if only PyTorch is used for the base implementation).

---

### Problem 1: Encoder-Decoder Transformer with Multi-Head Attention

#### Overview
This section implements a **Transformer model** with one encoder and one decoder, utilizing **multi-head attention** for sequence-to-sequence learning. The model is designed to process text data efficiently with **self-attention mechanisms** and **layer normalization**.

#### Features
* **Multi-Head Attention**: Utilizes 8 attention heads for improved context understanding.
* **Positional Encoding**: Incorporates positional information into token embeddings.
* **Layer Normalization**: Stabilizes training by maintaining zero mean and unit variance.
* **Fully Connected Layers**: Enhances feature extraction through linear transformations.
* **Encoder-Decoder Architecture**: Implements a Transformer model for text processing tasks.

---

### Problem 2: Decoder-Only Transformer with KV Caching

#### Overview
This section focuses on implementing a simple 2-layer decoder-only Transformer model and optimizing its autoregressive generation process using KV caching.

#### Implementation Details

This part of the repository addresses the following challenges:

##### 2a. Transformer Model Implementation
* **Task**: Implement all `TODO` sections within the provided skeleton code to create a fully functional 2-layer decoder-only Transformer model.
* **Functionality**: The model should be capable of autoregressively generating `n` tokens given an initial prompt.

##### 2b. Model Evaluation
* **Task**: Ensure the correct implementation of the model by verifying that the logits generated after the initial prompt processing match the expected values for a given model weight file.
* **How to Run**:
    ```bash
    python your_decoder_script.py --evaluate
    ```
* **Validation**: Check the console output for matching logits against the provided reference values.

##### 2c. KV Caching Implementation and Validation
* **Task**: Implement KV caching to store the generated Key and Value states from previous iterations. In subsequent generation steps, these cached values should be used instead of naively processing the entire sequence.
* **Validation**: Verify that the logits produced with KV caching exactly match the logits produced without KV caching. This confirms the correctness of the caching mechanism.
* **How to Run**:
    ```bash
    python your_decoder_script.py --kv_evaluate
    ```
* **Validation**: The script should report whether logits match between cached and non-cached runs.

##### 2d. Performance Benchmarking
* **Task**: Compare the generation latency (time to generate `n` tokens given a prompt) with and without KV caching. This comparison should be performed for varying sequence lengths: `[10, 50, 100, 500, 1000]`.
* **Output**: Generate a plot showing latency with and without KV as a function of sequence length.
* **How to Run**:
    ```bash
    python your_decoder_script.py --benchmark
    ```
* **Validation**: Observe the generated plot, which should clearly illustrate the performance benefits of KV caching, especially for longer sequences.
