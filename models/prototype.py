# Learnable multi-prototype memory bank storing 1 Real prototype and K=6 diverse Fake prototypes in 128-d embedding space.
# Inputs: L2-normalized embeddings z of shape (B, 128); Outputs: Cosine distances d_real(z), min fake distance d_fake(z), and per-prototype similarities.
