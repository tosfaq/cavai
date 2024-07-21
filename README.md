# CAVAI
CAVAI (CT Annotation, Viewing, and Analyzing Instrument) is a tool designed for efficient annotation, viewing, and analysis of CT scans. Aimed at researchers and practitioners in the medical imaging field, CAVAI offers a suite of features to enhance the workflow of CT image analysis.

## Features

- Per-slice series labeling for classification (multiclass/multilabel) *(not maintained anymore)*
- Per-slice series labeling for detection
- Linear interpolation for bounding boxes
- Saving and loading checkpoints
- Windowing
- Maximum Intensity Projection (MIP) mode
- Readable JSON support for checkpoints

## Installation

> Tip: use a python venv

```
cd cavai
pip install -r ./requirements.txt
```

## Usage

The repository contains 2 files: `cavai.py` for detection and `cavai_clf.py` for classification *(not maintained anymore)*.

The example below uses `cavai.py`.

```
python cavai.py --series <path to series to open> --labels <path to json file with per-series labeling> --checkpoint <path to json file with CAVAI checkpoint> <path to JSON output checkpoint>
```

You can also type `python cavai.py -h` for help.

## Keybindings

- Use your mouse wheel to scroll through the slices. You can also use `LeftArrow`/`a` and `RightArrow`/`d`
- Put your cursor exactly on the box border and click `Control`+`RightMouseButton` to select first interpolation box. Scroll a few slices and select second interpolation box by hovering on the box border and clicking `Option`+`RightMouseButton`.
- To remove a box, put your cursor on the box border and click `RightMouseButton`. 
- Use `Space` to interpolate with 2 boxes selected.

## Internal representation of labels

The program uses JSON format for saving and loading labels:

```
{
    "img_size": 512,
    "labels": {
        "ct_lungs_artefacts_71_1/031/10000000/10000001/10000007": {
            "13": { 
                "filename": "10000001",
                "bboxes": [[0, start_x, start_y, end_x, end_y],
                           [0, start_x, start_y, end_x, end_y]]
            }
        }
    }
}
```

13 is an index of a slice.

> Currently viewer relies on the folder structure as above (a series folder has 3 more subfolders inside). You can adjust `get_folder_key` function for your needs.

## To-do

- [ ] Functionality to choose keybinds

## Contact

scenic-00airs \<at\> icloud \<dot\> com
