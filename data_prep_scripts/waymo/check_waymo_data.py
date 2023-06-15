import argparse
from pathlib import Path

# Get path to the Waymo dataset

parser = argparse.ArgumentParser()
parser.add_argument('waymo_dir', type=Path, help='path to the Waymo dataset')
args = parser.parse_args()

training_dir = args.waymo_dir / 'training'
validation_dir = args.waymo_dir / 'validation'

flow_label_dir = args.waymo_dir / 'train_nsfp_flow'

assert training_dir.exists(), 'Training directory does not exist'
assert validation_dir.exists(), 'Validation directory does not exist'

training_subfolders = sorted(training_dir.glob('segment*'))
validation_subfolders = sorted(validation_dir.glob('segment*'))

flow_subfolders = sorted(flow_label_dir.glob('segment*'))

assert len(
    training_subfolders
) == 798, f'Expected 798 training subfolders, got {len(training_subfolders)}'
assert len(
    validation_subfolders
) == 202, f'Expected 202 validation subfolders, got {len(validation_subfolders)}'

# Check that all subfolders have the expected number of flows
for subfolder in training_subfolders:
    num_flows = sorted(subfolder.glob('*.pkl'))
    if len(num_flows) < 196:
        print(str(subfolder), len(num_flows))

for subfolder in validation_subfolders:
    num_flows = sorted(subfolder.glob('*.pkl'))
    if len(num_flows) < 196:
        print(str(subfolder), len(num_flows))

train_subfolder_names = [subfolder.name for subfolder in training_subfolders]
for subfolder in flow_subfolders:
    num_flows = sorted(subfolder.glob('*.npz'))
    if len(num_flows) < 196:
        idx = train_subfolder_names.index(subfolder.name)
        print(subfolder.name, len(num_flows), idx)

# Get all the folders in the training subset that aren't in flow subfolders
missing_training_subfolders = sorted(
    set(training_subfolders) - set(flow_subfolders))
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Missing training subfolders:")
for subfolder in missing_training_subfolders:
    print(subfolder.name)
