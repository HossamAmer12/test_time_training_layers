import torch
import random
import numpy as np

def check_similarity(A, A_, Comment):
    difference = torch.abs(A - A_)
    if torch.allclose(A, A_, atol=1e-6):
        print("Tensors are the same.")
    else:
        print("Tensors are different.")
    print("max diff: ", difference.max())

# ======== Reproducibility Setup ========
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# =======================================

# Define dimensions
B, nh, K, f = 2, 4, 5, 8  # Batch size, num heads, sequence length, features

print(f"B={B} nh={nh} K={K} f={f}")

# Generate fixed random input tensors
XQ_mini_batch = torch.randn(B, nh, K, f)
X1 = torch.randn(B, nh, K, f)
eta_mini_batch = torch.rand(B, nh, K, 1)  # Learning rate per token
grad_l_wrt_Z1 = torch.randn(B, nh, K, f)
W1_init = torch.randn(B, nh, f, f)
b1_init = torch.randn(B, nh, 1, f)

# Compute attention (lower-triangular masked QK^T)
Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))  # [B, nh, K, K]

# Expand eta and apply mask
eta_masked = torch.tril(eta_mini_batch.expand(-1, -1, -1, K))  # [B, nh, K, K]

# Update bias: b1_bar = b1_init - tril(eta) @ grad
b1_bar = b1_init - eta_masked @ grad_l_wrt_Z1  # [B, nh, K, f]

# Compute intermediate terms
t1 = XQ_mini_batch @ W1_init  # [B, nh, K, f]
t2 = (eta_mini_batch * Attn1) @ grad_l_wrt_Z1  # [B, nh, K, f]
t3 = b1_bar  # [B, nh, K, f]

# Final output
Z1_bar = t1 - t2 + t3

# Print output shapes to verify
print("Attn1 shape:", Attn1.shape)
print("b1_bar shape:", b1_bar.shape)
print("Z1_bar shape:", Z1_bar.shape)
# print(Z1_bar)

########## Sharding

tp_size = 2

# Shard W0
W1_shards = torch.chunk(W1_init, tp_size, dim=-1)  # [B, nh, f, f_chunk]
t1_parts = [XQ_mini_batch @ W1_i for W1_i in W1_shards]  # each [B, nh, K, f_chunk]

# Shard the gradients
grad_shards = torch.chunk(grad_l_wrt_Z1, tp_size, dim=-1)  # [B, nh, K, f_chunk]
t2_parts = [(eta_mini_batch * Attn1) @ g_i for g_i in grad_shards]  # [B, nh, K, f_chunk]

# Split grad and b1_init across the feature dim (dim=-1)
b1_init_shards = torch.chunk(b1_init.expand(-1, -1, K, -1), tp_size, dim=-1)  # match K

# Optional: ensure eta_masked has shape [B, nh, K, K]
eta_masked = torch.tril(eta_mini_batch.expand(-1, -1, -1, K))  # causal LR mask

# Compute b1_bar shards
b1_bar_shards = [
    b1_init_shards[i] - eta_masked @ grad_shards[i]
    for i in range(tp_size)
]

# Communication (All Gather)
# All Gather because TTT-Linear (single layer) is used.
t1 = torch.cat(t1_parts, dim=-1)  # [B, nh, K, f]
t2 = torch.cat(t2_parts, dim=-1)  # [B, nh, K, f]
b1_bar = torch.cat(b1_bar_shards, dim=-1)
t3 = b1_bar

Z1_bar_sharded = t1 - t2 + t3

check_similarity(Z1_bar, Z1_bar_sharded, "Test")



#
#  X1_shards = torch.chunk(X1, num_chunks, dim=-1)  # Split keys (f-dim)
# XQ_shards = torch.chunk(XQ_mini_batch, num_chunks, dim=-1)

# partial_logits = [
#     XQ_shards[i] @ X1_shards[i].transpose(-2, -1)  # [B, nh, K, K]
#     for i in range(num_chunks)
# ]


# print(partial_logits[0].shape)
# print(partial_logits)

# Attn1 = torch.tril(sum(partial_logits))  # Combine and apply causal mask

