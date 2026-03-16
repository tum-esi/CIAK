# CIAK - Camera-feed Injection Attack Framework

This repository contains the implementation of CIAK (Camera-feed Injection Attack), a framework for evaluating the robustness of cooperative perception systems against camera feed tampering attacks. CIAK is designed to inject adversarial objects into camera feeds of connected autonomous vehicles (CAVs) to assess the impact on collaborative perception performance.



## Installation

### Clone the repository:
```bash
git clone https://...
cd CIAK
```

### Build docker container
The repository comes with a docker container under `scripts/dockerfile`. To build it simply run the command 

```console
docker build . -t "nameOfTheContainer"
```

### Run the docker container

```console
docker run --gpus all --shm-size=64g -it -v ../yourFolder:/FolderInTheDocker/ nameOfTheContainer
```

### Add the python path
```console
export PYTHONPATH=$PYTHONPATH:/workspace/CoBEVT/opv2v
```

## Generate the snippets using rembg(u2net)
### Background removal tools:
- [rembg](https://github.com/danielgatis/rembg?tab=readme-ov-file)
- [dis-bg-remover](https://test.com)

### Download and install Dataset Modifications
Download the dataset modifications from [here](https://mediatum.ub.tum.de/1846877).

Copy the contents of `<dataset_modification_src_dir>/dataset_modifications` to `<CIAK_src_dir>/dataset_modifications`.

Copy the contents of `<dataset_modification_src_dir>/assets` to `<CIAK_src_dir>/assets`.

### Invoke the data injection for provided scenes
Directly run `attacker.py`. It performs data injection on the original input data from `./dataset_modifications/` and exports the results to `./assets/tampered_feeds/`.
```console
python attacker.py
```

### (OPTIONAL) Invoke the data injection for custom scenes

The data injection step is performed using the class `Attacker` from `attacker.py`.
First, instantiate an object of the class:

```python
attacker = Attacker()
```

Then invoke the data injection. This has to be performed for each image:

```python
attacker.run_pipeline(
    model_image_path,      # Path to the image of the to-be-injected vehicle
    input_path_attacker,   # Path to the attacker's PoV image, in which the injection shall be performed
    output_dir,            # Output directory (the output file's name will be identical to the name of the attacker's input image)
    preprocess,            # To automatically remove the background of the to-be-injected vehicle, set to true
    injection_coords       # BBox object with the x,y, width and height values where the vehicle should be injected in the attacker's image
)
```

If preprocess is set to true, the background of the image containing the to-be-injected vehicle is automatically removed using rembg.


### CoBEVT OPV2V Track
follow the [Installation Instructions here](https://github.com/DerrickXuNu/CoBEVT/tree/main/opv2v)

### Command Overview
```console

    ▄▄▄▄    ▄▄▄▄▄▄      ▄▄     ▄▄   ▄▄▄
  ██▀▀▀▀█   ▀▀██▀▀     ████    ██  ██▀
 ██▀          ██       ████    ██▄██
 ██           ██      ██  ██   █████
 ██▄          ██      ██████   ██  ██▄
  ██▄▄▄▄█   ▄▄██▄▄   ▄██  ██▄  ██   ██▄
    ▀▀▀▀    ▀▀▀▀▀▀   ▀▀    ▀▀  ▀▀    ▀▀
usage: main.py [-h] [--model_dir MODEL_DIR] [--model_type MODEL_TYPE] --config CONFIG [--explore] [--limit LIMIT]
               [--no-cuda] [--save-dir SAVE_DIR] [--dump-dynamic-masks] [--debug] [--save-collab-mask] [--evaluate]

CoBEVT inference runner with optional exploration mode

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Optional checkpoint directory (overrides default if set).
  --model_type MODEL_TYPE
                        Model type: "dynamic" or "static" prediction
  --config CONFIG       Path to model/data config yaml
  --explore             Enable detailed exploratory mode, and saves the outputs of the different layers (Default Encoder 1-4, fusion output, and decoder)
  --limit LIMIT         Limit number of batches to process (0 = all).
  --no-cuda             Force CPU even if CUDA is available.
  --save-dir SAVE_DIR   Directory to save visualizations & outputs.
  --debug               Print per-batch tensor stats for model outputs.
  --evaluate            Enable evaluation between benign and attack runs
```

### Dynamic Evaluation example command
```python
python main.py  --config model_snapshots/cobevt/config.yaml --model_type dynamic --model_dir model_snapshots/cobevt --evaluate
```
### Static Evaluation example command
```python
python main.py  --config model_snapshots/cobevt_static/config.yaml --model_type static --model_dir model_snapshots/cobevt_static --evaluate
```
  
