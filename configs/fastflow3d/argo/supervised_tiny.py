_base_ = "supervised.py"

test_sequence_dir = "/efs/argoverse2_tiny/val/"
test_flow_dir = "/efs/argoverse2_tiny/val_sceneflow/"

save_output_folder = "/efs/argoverse2_tiny/val_supervised_out/"

max_test_sequence_length = 2

test_loader = dict(name="ArgoverseSupervisedFlowSequenceLoader",
                   args=dict(raw_data_path=test_sequence_dir,
                             flow_data_path=test_flow_dir))

test_dataloader = dict(
    args=dict(batch_size=8, num_workers=8, shuffle=False, pin_memory=True))

test_dataset = dict(name="SubsequenceSupervisedFlowDataset",
                    args=dict(max_sequence_length=max_test_sequence_length))
