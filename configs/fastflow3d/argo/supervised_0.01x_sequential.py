_base_ = "./supervised.py"

epochs = 500
check_val_every_n_epoch = 500
validate_every = None
dataset = dict(args=dict(subset_fraction=0.01, subset_mode='sequential'))
