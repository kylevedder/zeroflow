_base_ = "./unsupervised_lidar_all.py"

test_sequence_dir = "/efs/argoverse_lidar_subsampled_all_complimentary/"
flow_save_folder = "/efs/argoverse_lidar_subsampled_all_complimentary/train_nsfp_flow/"


model = dict(args=dict(flow_save_folder=flow_save_folder))

test_loader = dict(args=dict(sequence_dir=test_sequence_dir))

