# "Attention is all you need" from scratch

This project implements the foundational transformer model introduced in the paper "Attention Is All You Need" by Vaswani et al. The transformer architecture revolutionized natural language processing and sequence modeling tasks by eliminating recurrence and focusing entirely on self-attention mechanisms.

# Features

- Multi-Head Self-Attention: Implements the scaled dot-product attention mechanism with multi-head capabilities.

- Positional Encoding: Adds positional information to input embeddings for sequence modeling.

- Encoder-Decoder Architecture: Includes both encoder and decoder stacks for tasks like machine translation.

- Feedforward Neural Networks: Utilizes position-wise feedforward networks in each transformer block.

- Layer Normalization and Dropout: Enhances stability and generalization during training.

- Masking Mechanisms: Supports padding and look-ahead masks for handling variable-length sequences and autoregressive tasks.

- Scalability: Modular design to extend and experiment with variations of the transformer model.

# Goals

- Recreate the original transformer architecture from scratch for educational and experimental purposes.

- Explore how the model handles sequence-to-sequence tasks like machine translation.

- Provide a foundation for extending the transformer to advanced architectures (e.g., BERT, GPT).
