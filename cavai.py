import argparse
import tkinter as tk
from ct_viewer import CTViewer


def parse_arguments():
    parser = argparse.ArgumentParser(description='CAVAI - CT Annotation, Viewing and Analysing Instrument')
    parser.add_argument(
        'destination', 
        help='path to the json file where labelling will be saved')
    parser.add_argument(
        '--series',
        metavar='S',
        help='path to a folder with a CT series')
    parser.add_argument(
        '--labels',
        metavar='L',
        help='path to the json file with per-series labelling')
    parser.add_argument(
        '--checkpoint',
        metavar='C',
        help='path to the json file with labelling checkpoint to append to')
    parser.add_argument(
        '--genimages',
        action="store_true",
        help='if specified generates images with bboxes and exits')
    parser.add_argument(
        '--genvideos',
        action="store_true",
        help='if specified generates videos with bboxes and exits')
    parser.add_argument(
        '--gencompress',
        action="store_true",
        help='if specified generates images/videos with compression')
    parser.add_argument(
        '--genparent',
        type=str,
        help='parent folder for every folder in checkpoint')
    parser.add_argument(
        '--genlabels',
        type=str,
        help='parent folder for detections (txt)')
    parser.add_argument(
        '--gen-show-hu',
        action="store_true",
        help='show max HU for bboxes in exported files')
    parser.add_argument(
        '--genfile',
        type=str,
        default=None,
        help='a txt file with folder keys (can be just like "ct_lungs_artefacts_71_1/031") on each line for generation; folder keys will be simply joined with --genparent argument')
    parser.add_argument(
        '--gen-out-folder',
        type=str,
        default=None,
        help='name of the folder where generated videos/images will be stored')
    parser.add_argument(
        '--sigma',
        type=float,
        default=2.0,
        help='sigma parameter for Laplacian of Gaussian filter')
    parser.add_argument(
        '--windowed-log',
        action="store_true",
        help='apply windowing before Laplacian of Gaussian filter')
    parser.add_argument(
        '--replace-bg',
        action="store_true",
        help='replace background (-2048) with -1000 before windowing in Laplacian of Gaussian filtering')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = tk.Tk()
    viewer = CTViewer(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()