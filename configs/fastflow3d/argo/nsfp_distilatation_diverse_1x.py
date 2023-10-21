_base_ = "./nsfp_distilatation_diverse.py"

dataset = dict(args=dict(subset_fraction=0.5, subset_mode='sequential'))