# ZeroFlow: Scalable Scene Flow via Distillation

[Kyle Vedder](http://vedder.io), [Neehar Peri](http://www.neeharperi.com/), [Nathaniel Chodosh](https://scholar.google.com/citations?user=b4qKr7gAAAAJ&hl=en), [Ishan Khatri](https://ishan.khatri.io/), [Eric Eaton](https://www.seas.upenn.edu/~eeaton/), [Dinesh Jayaraman](https://www.seas.upenn.edu/~dineshj/), [Yang Liu](https://youngleox.github.io/), [Deva Ramanan](https://www.cs.cmu.edu/~deva/), and [James Hays](https://faculty.cc.gatech.edu/~hays/)

Project webpage: [vedder.io/zeroflow](http://vedder.io/zeroflow)

arXiv link: [arxiv.org/abs/2305.10424](http://arxiv.org/abs/2305.10424)

**News:**

- July 31st, 2023: The ZeroFlow XL student model is now **state-of-the-art** on the [Scene Flow Challenge](https://eval.ai/web/challenges/challenge-page/2010/overview)! See the [Getting Started](./GETTING_STARTED.md) document for details on setting up training on additional data.
 - June 18th, 2023: ZeroFlow was selected as a highlighted method in the CVPR 2023 _Workshop on Autonomous Driving_ [Scene Flow Challenge](https://eval.ai/web/challenges/challenge-page/2010/overview)!
 

**Citation:**

```
@article{Vedder2023zeroflow,
    author    = {Kyle Vedder and Neehar Peri and Nathaniel Chodosh and Ishan Khatri and Eric Eaton and Dinesh Jayaraman and Yang Liu Deva Ramanan and James Hays},
    title     = {{ZeroFlow: Scalable Scene Flow via Distillation}},
    journal   = {arXiv},
    year      = {2023},
}
```

## Pre-requisites / Getting Started

Read the [Getting Started](./GETTING_STARTED.md) doc for detailed instructions to setup the AV2 and Waymo Open datasets and use the prepared docker environments.

## Pretrained weights

All trained weights from the paper are available for download from [this repo](https://github.com/kylevedder/zeroflow_weights).

## Training a model

 Inside the main container (`./launch.sh`), run the `train_pl.py` with a path to a config (inside `configs/`) and optionally specify any number of GPUs (defaults to all GPUs on the system).

```
python train_pl.py <my config path> --gpus <num gpus>
```

The script will start by verifying the val dataloader works, and then launch the train job.

## Testing a model

Inside the main  (`./launch.sh`), run the `train_pl.py` with a path to a config (inside `configs/`), a path to a checkpoint, and the number of GPUs (defaults to a single GPU).

```
python test_pl.py <my config path> <my checkpoint path> --gpus <num gpus>
```

## Generating paper plots

After all relevant checkpoints have been tested, thus generating result files in `validation_results/configs/...`, run `plot_performance.py` to generate the figures and tables used in the paper.

## Submitting to the AV2 Scene Flow competition

1. Dump the outputs of the model
    - `configs/fastflow3d/argo/nsfp_distilatation_dump_output.py` to dump the `val` set result 
    - `configs/fastflow3d/argo/nsfp_distilatation_dump_output_test.py` to dump the `test` set result
3. Convert to the competition submission format (`av2_scene_flow_competition_submit.py`) using [the official evaluation point subset](https://argoverse.github.io/user-guide/tasks/3d_scene_flow.html#evaluation-point-subset)
4. Use official zip `make_submission_archive.py` file (`python /av2-api/src/av2/evaluation/scene_flow/make_submission_archive.py <path to step 2 results> /efs/argoverse2/test_official_masks.zip`)
