# Computes Multi-Prototype distance, margin separation, and fake-prototype diversity loss (L_proto) and combines L_total = L_cls + 0.5*L_con + 0.2*L_proto.
# Inputs: Cosine distances d_real(z), d_fake(z), prototype vectors, and labels y; Outputs: Scalar prototype loss L_proto and total loss L_total.
