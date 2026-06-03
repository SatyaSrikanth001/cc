# cc
Item Detail
Feature pool You have a code that can generate up to ~441 features. From those, you have currently selected a subset of ~90 features (the ones you listed), after dropping many via a drop_classes array.
Model OCSVM, RBF kernel, per‑user (17 independent models). Training on ~40 genuine sessions per user. Test on ~35‑40 genuine + 20‑25 impostor sessions.
Current metrics Accuracy ~69 %, FRR = 32.25 % (genuine users rejected), FAR = 0.15 % (impostors barely accepted).
Hyperparameters nu=0.05, gamma=0.00055, threshold = 0 (score > 0 → genuine).
Constraints OCSVM cannot be replaced, but you can adjust the decision threshold, tune nu and gamma, and use data from other users for feature selection.