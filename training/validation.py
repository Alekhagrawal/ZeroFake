# Runs validation-split evaluation after each epoch to compute ROC-AUC and calibrate open-set z-score parameters and tau_reject (95% coverage).
# Inputs: Trained model and validation split DataLoader; Outputs: Validation ROC-AUC, EER, F1, and calibrated open-set uncertainty statistics.
