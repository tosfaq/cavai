# CAVAI
CAVAI (CT Annotation, Viewing, and Analyzing Instrument) is a tool designed for efficient annotation, viewing, and analysis of CT scans. Aimed at researchers and practitioners in the medical imaging field, CAVAI offers a suite of features to enhance the workflow of CT image analysis.

## Features

- Per-slice series labeling for classification (multiclass/multilabel)
- Per-slice series labeling for detection
- Linear interpolation for bounding boxes
- Saving and loading checkpoints
- Windowing
- Maximum Intensity Projection (MIP) mode
- Readable JSON support for checkpoints

## Installation

```
cd cavai
pip install -r ./requirements.txt
```

## Usage

The repository contains 2 files: `cavai.py` for classification and `cavai_det.py` for detection.

The example below uses `cavai_det.py`.

```
python cavai_det.py --series <path to series to open> --labels <path to json file with per-series labeling> --checkpoint <path to json file with CAVAI checkpoint> <path to JSON output checkpoint>
```

You can also type `python cavai2.py -h` for help.

## To-do

- [ ] Add keybinds to README
- [ ] Ability to choose keybinds

## Contact

scenic-00airs \<at\> icloud \<dot\> com
