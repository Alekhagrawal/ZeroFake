# Computes calibrated open-set uncertainty U(x) and applies the 3-way decision rule (REAL, FAKE, or UNKNOWN) calibrated strictly on val.
# Inputs: Embeddings z, fake probabilities p, and prototype distances; Outputs: Calibrated uncertainty U(x), threshold tau_reject, and 3-way verdict.
