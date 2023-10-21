from pathlib import Path
import shutil
import multiprocessing
import tqdm
import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict, Optional, Any
import enum
# Needed to properly load the feather files
import pyarrow

ssd_data_lidar_split1 = Path('/mnt/kostas-graid/datasets/argoverse_lidar_subsampled_all_hdd')
hdd_data_lidar_split1 = Path('/home/kvedder/datasets/argoverse_lidar_subsampled_all_hdd')
ssd_data_lidar_split2 = Path('/mnt/kostas-graid/datasets/argoverse_lidar_subsampled_all_complimentary')
ssd_data_av2 = Path('/mnt/kostas-graid/datasets/argoverse2')
hdd_data_av2 = Path('/home/kvedder/datasets/argoverse2')

max_cpu_count = multiprocessing.cpu_count() * 4


def build_hdd_cache(split_folder : Path, sequence_folder_subpath : Optional[str], file_extension : str, size_int : Optional[int] = None):
    assert split_folder.exists(), f'{split_folder} does not exist'
    cache_file = split_folder / 'info_cache.json'
    if cache_file.exists():
        # remove the cache file
        cache_file.unlink()

    print('Building cache file for', split_folder.name, '...')
    # build the cache file, mapping sequence name to number of .feather files in the sensors/lidar/ subfolder
    cache = {}
    split_folders = sorted(split_folder.glob('*'))
    for sequence_folder in tqdm.tqdm(split_folders):
        if sequence_folder_subpath is not None:
            sequence_folder = sequence_folder / sequence_folder_subpath
        cache[sequence_folder.name] = len(list(sequence_folder.glob(f'*{file_extension}')))

    with open(cache_file, 'w') as f:
        json.dump(cache, f)


def load_hdd_cache(split_folder : Path, sequence_folder_subpath : Optional[str], file_extension : str) -> Dict[str, int]:
    assert split_folder.exists(), f'{split_folder} does not exist'
    cache_file = split_folder / 'info_cache.json'
    if not cache_file.exists():
        build_hdd_cache(split_folder, sequence_folder_subpath, file_extension)
    with open(cache_file, 'r') as f:
        cache = json.load(f)
    return cache

def load_hdd_lidar_cache(split_folder : Path) -> Dict[str, int]:
    return load_hdd_cache(split_folder, "sensors/lidar/", '.feather')

def load_hdd_flow_cache(split_folder : Path) -> Dict[str, int]:
    return load_hdd_cache(split_folder, None, '.npz')


# validate the train_nsfp_flow subfolders have one npz file in them

def is_valid_npz(npz : Path):
    assert npz.exists()
    try:
        data = dict(np.load(npz))
    except Exception as e:
        print("Error reading npz file:", npz, e)
        return False
    return True

def is_valid_npy(npy : Path):
    assert npy.exists()
    try:
        data = np.load(npy)
    except Exception as e:
        print("Error reading npy file:", npy, e)
        return False
    return True

def is_valid_feather(feather : Path):
    assert feather.exists()
    assert feather.is_file(), f'{feather} is not a file'
    try:
        data = pd.read_feather(feather)
    except Exception as e:
        print("Error reading feather file:", feather, e)
        return False
    return True

def is_valid_json(json_file : Path):
    assert json_file.exists()
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print("Error reading json file:", json_file, e)
        return False
    return True

def is_valid_by_extension(file : Path):
    assert file.exists(), f'{file} does not exist'
    if file.suffix == '.feather':
        return is_valid_feather(file)
    elif file.suffix == '.json':
        return is_valid_json(file)
    elif file.suffix == '.npy':
        return is_valid_npy(file)
    elif file.suffix == '.npz':
        return is_valid_npz(file)
    else:
        print("Unknown file extension:", file)
        return False


# File vs Folder Enum
class PathType(enum.Enum):
    FILE = 1
    FOLDER = 2

def get_lidar_folder_paths_that_need_repair(folder : Path, num_lidar_frames : int, verbose : bool = False) -> List[Tuple[Path, PathType]]:
    broken_components = []

    if not folder.exists():
        print("Folder does not exist:", folder)
        # No need to log all other failures
        verbose = False

    def exists_and_valid(path : Path, validate_fn):
        if not path.exists():
            if verbose:
                print("Path does not exist:", path)
            return False
        if path.is_symlink():
            if verbose:
                print("Path is a symlink:", path)
            return False
        return validate_fn(path)
    
    def multi_exist_and_valid(paths : List[Path], validate_fn, expected_number : int):
        if len(paths) != expected_number:
            return False
        

        return all([exists_and_valid(e, validate_fn) for e in paths])
    
    # Ensure city_SE3_egovehicle.feather exists
    if not exists_and_valid(folder / 'city_SE3_egovehicle.feather', is_valid_feather):
        if verbose:
            print("city_SE3_egovehicle.feather broken:", folder)
        broken_components.append((folder / 'city_SE3_egovehicle.feather', PathType.FILE))

    if (folder / 'map').is_symlink():
        if verbose:
            print("map/ folder is a symlink:", folder)
        broken_components.append((folder / 'map', PathType.FOLDER))
    elif not multi_exist_and_valid(list((folder / 'map').glob('*.json')), is_valid_json, 2):
        if verbose:
            print("map/ folder does not have 2 json files:", folder)
        broken_components.append((folder / 'map', PathType.FOLDER))
    elif not multi_exist_and_valid(list((folder / 'map').glob('*.npy')), is_valid_npy, 1):
        if verbose:
            print("map/ folder does not have 1 npy file:", folder)
        broken_components.append((folder / 'map', PathType.FOLDER))

    elif not multi_exist_and_valid(list((folder / 'sensors' / 'lidar').glob('*.feather')), is_valid_feather, num_lidar_frames):
        if verbose:
            print("sensors/lidar/ folder broken:", folder)
        broken_components.append((folder / 'sensors' / 'lidar', PathType.FOLDER))

    return broken_components
    


def is_nsfp_flow_folder_bad(folder : Path, num_flow_frames : int) -> bool:
    if not folder.exists():
        return True
    
    npzs = list(folder.glob('*.npz'))
    if len(npzs) != num_flow_frames:
        return True
    
    valid_npzs = [is_valid_npz(e) for e in npzs]
    return not all(valid_npzs)

def is_gt_flow_bad(npz_file : Path, verbose : bool = False) -> bool:
    if not npz_file.exists():
        print("NPZ file", npz_file , "missing")
        return True
    try:
        data = dict(np.load(npz_file))
    except Exception as e:
        print("Loading NPZ", npz_file, "failed with", e)
        return True
    return False


def repair_folder(ssd_folder : Path, hdd_folder : Path):
    print("Repairing", ssd_folder)
    assert hdd_folder.exists(), f'{hdd_folder} does not exist'
    # Delete SSD folder if it exists
    if ssd_folder.exists():
        shutil.rmtree(ssd_folder)
    # Copy from HDD to SSD
    shutil.copytree(hdd_folder, ssd_folder, symlinks=False)

def repair_file(ssd_file : Path, hdd_file : Path):
    print("Repairing", ssd_file)
    assert hdd_file.exists(), f'{hdd_file} does not exist'
    # Delete SSD file if it exists
    if ssd_file.exists():
        ssd_file.unlink()
    # Copy from HDD to SSD
    ssd_file.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(hdd_file, ssd_file)


############################################

def process_lidar_problem(ssd_problem_setup : dict):
    ssd_root_folder = ssd_problem_setup['ssd_root_folder']
    hdd_root_folder = ssd_problem_setup['hdd_root_folder']
    ssd_subfolder = ssd_problem_setup['ssd_subfolder']
    num_lidar_frames = ssd_problem_setup['num_lidar_frames']
    repairs_needed = get_lidar_folder_paths_that_need_repair(ssd_subfolder, num_lidar_frames, True)
    for repair_path, repair_type in repairs_needed:
        relative_path = repair_path.relative_to(ssd_root_folder)
        hdd_folder = hdd_root_folder / relative_path
        if repair_type == PathType.FILE:
            repair_file(repair_path, hdd_folder)
        else:
            repair_folder(repair_path, hdd_folder)

def process_gt_problem(ssd_problem_setup : dict):
    ssd_file = ssd_problem_setup['ssd_file']
    hdd_file = ssd_problem_setup['hdd_file']
    if not is_gt_flow_bad(ssd_file, verbose=True):
        return
    repair_file(ssd_file, hdd_file)

def process_nsfp_problem(ssd_problem_setup : dict):
    ssd_root_folder = ssd_problem_setup['ssd_root_folder']
    hdd_root_folder = ssd_problem_setup['hdd_root_folder']
    folder = ssd_problem_setup['ssd_subfolder']
    num_flow_frames = ssd_problem_setup['num_flow_frames']
    if not is_nsfp_flow_folder_bad(folder, num_flow_frames):
        return
    
    # Repair the entire folder
    repair_folder(folder, hdd_root_folder / folder.relative_to(ssd_root_folder))

def touch_folder_problem(ssd_problem_setup : dict):
    folder = ssd_problem_setup['folder']

    process_queue = [folder]

    def process_element(element : Path):
        if element.is_dir():
            # Add all subfolders to the queue
            for subfolder in element.iterdir():
                process_queue.append(subfolder)
            return
        
        # Touch the file
        element.touch()
        if not is_valid_by_extension(element):
            print("Error touching", element)


    while len(process_queue) > 0:
        process_element(process_queue.pop())

############################################


def check_lidar(ssd_data : Path, hdd_data : Path, split : str):
    print("Check lidar for", ssd_data, split)
    hdd_data_cache = load_hdd_lidar_cache(hdd_data / split)

    def hdd_cache_item_to_ssd_problem(subfolder_name : str, num_lidar_frames : int):
        return {
            'hdd_root_folder' : hdd_data,
            'ssd_root_folder' : ssd_data,
            'ssd_subfolder' : ssd_data / split / subfolder_name,
            'num_lidar_frames' : num_lidar_frames,
        }
    
    # Build the SSD validation problem setups using the cache information
    problem_setups = [
        hdd_cache_item_to_ssd_problem(subfolder_name, num_lidar_frames)
        for subfolder_name, num_lidar_frames in hdd_data_cache.items()
    ]
    run_problems(process_lidar_problem, problem_setups)

def check_gt_flow(ssd_data : Path, hdd_data : Path, split : str):
    print("Check gt flow for", split)
    # All the gt flows are just .npz files in the root of the split folder.
    # Just check that they're all valid
    problem_setups = [
        {
            'hdd_root_folder' : hdd_data,
            'ssd_root_folder' : ssd_data,
            'ssd_file' : ssd_data / split / e.name,
            'hdd_file' : e,
        }
        for e in sorted((hdd_data / split).glob('*.npz'), key=lambda x: x.name)
    ]
    run_problems(process_gt_problem, problem_setups)


def check_nsfp_flow(ssd_data : Path, hdd_data : Path, split : str):
    print("Check nsfp flow for", split)
    hdd_data_cache = load_hdd_flow_cache(hdd_data / split)
    # Each *_nsfp_flow folder has a bunch of subfolders, each of which is a sequence of NPZ files.

    def hdd_cache_item_to_ssd_problem(subfolder_name : str, num_flow_frames : int):
        return {
            'hdd_root_folder' : hdd_data,
            'ssd_root_folder' : ssd_data,
            'ssd_subfolder' : ssd_data / split / subfolder_name,
            'num_flow_frames' : num_flow_frames,
        }
    
    # Build the SSD validation problem setups using the cache information
    problem_setups = [
        hdd_cache_item_to_ssd_problem(subfolder_name, num_flow_frames)
        for subfolder_name, num_flow_frames in hdd_data_cache.items()
    ]
    run_problems(process_nsfp_problem, problem_setups)

def touch_folder_recursive(folder : Path, ignore_names : List[str] = []):
    print("Touching subfolders of", folder)
    problem_setups = [
        {
            'folder' : e,
        }
        for e in folder.glob('*') if e.name not in ignore_names
    ]
    run_problems(touch_folder_problem, problem_setups)
    


def run_problems(process_fn, problem_setups):
    num_processes = min(len(problem_setups), max_cpu_count)
    print("Num processes:", num_processes)
    if num_processes > 1:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # use imap to get a progress bar
            for _ in tqdm.tqdm(
                    pool.imap_unordered(process_fn, problem_setups),
                    total=len(problem_setups)):
                pass
    else:
        for problem_setup in tqdm.tqdm(problem_setups):
            process_fn(problem_setup)

if __name__ == '__main__':
    touch_folder_recursive(ssd_data_lidar_split2 / 'train')
    touch_folder_recursive(ssd_data_lidar_split2 / 'train_nsfp_flow')
    check_nsfp_flow(ssd_data_lidar_split1, hdd_data_lidar_split1, 'train_nsfp_flow')
    check_nsfp_flow(ssd_data_av2, hdd_data_av2, 'train_nsfp_flow')
    check_lidar(ssd_data_lidar_split1, hdd_data_lidar_split1, 'train')
    check_gt_flow(ssd_data_av2, hdd_data_av2, 'train_sceneflow')
    check_gt_flow(ssd_data_av2, hdd_data_av2, 'val_sceneflow')
    check_lidar(ssd_data_av2, hdd_data_av2, 'val')
    check_lidar(ssd_data_av2, hdd_data_av2, 'train')
    
    
    

