_base_ = "./nsfp_distilatation.py"

test_dataloader = dict(
    args=dict(batch_size=16, num_workers=16, shuffle=False, pin_memory=True))