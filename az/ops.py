"""Op costs for estimators, in multiply-adds. A forward pass costs SAMPLE_OPS (az/model.py)."""
FN = 10    # one exp, log, sqrt, erf, Phi, or phi
BVN = 100  # one bivariate normal CDF
