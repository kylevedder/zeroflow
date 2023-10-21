_base_ = "./supervised_xl_backbone.py"

epochs = 9
validate_every = 158  # Number of training batches
dataset = dict(args=dict(subset_fraction=0.1, subset_mode='sequential'))