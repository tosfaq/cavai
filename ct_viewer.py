import tqdm
import json
import cv2
import sys
import os
import glob

import tkinter as tk
import numpy as np
import pydicom
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from tkinter.messagebox import showerror

from utils import LoG_filter, to_interval, get_folder_key, parse_detected_bboxes, createMIP

from constants import IMAGE_SIZE

class CTViewer:
    def __init__(self, root, args):
        self.root = root
        self.root.title("CAVAI")

        self.bring_window_to_forefront()

        self.is_numpy = False
        self.to_show = tk.StringVar(value="original")
        self.artifact_id = 0
        self.ct_series = None  # a 3d numpy array
        self.ct_series_windowed = None  # a 3d numpy array
        self.mip_series = None  # a 3d numpy array
        self.mip_series_windowed = None  # a 3d numpy array

        self.sigma = args.sigma
        self.windowed_LoG = args.windowed_log
        self.replace_bg = args.replace_bg
        self.current_log_image = None
        self.log_window_min = 200
        self.log_window_max = 1200  # could be 700

        # format: (slice_idx, [start_x, start_y, end_x, end_y])
        self.inter_box_1 = None
        self.inter_box_2 = None
        self.last_interpolation = []

        self.bbox_square_threshold = 20
        self.default_level = 0
        self.default_width = 3500

        self.series_path = args.series
        self.output_path = args.destination
        
        self.series_labels = self.load_json_file(args.labels)
        #self.artifact_options = self.load_json_file(artifacts_path, required=True)
        #self.final_labelling = self.load_json_file(args.checkpoint, default={})

        self.ct_series_data = self.load_json_file(args.checkpoint, default={"img_size": IMAGE_SIZE, "labels": {}})
        self.old_ckpt_labels = None
        self.old_series_labels = None

        # A set of tkinter ids of bboxes on the current slice
        self.current_bbox_ids = set()

        self.img_size = self.ct_series_data["img_size"]

        self.generate_compress = args.gencompress

        self.generate_output_folder = args.gen_out_folder

        # dictionary with predicted bboxes (--genlabels argument)
        self.detected_bboxes = parse_detected_bboxes(args.genlabels, self.img_size) if args.genlabels else None
        if self.detected_bboxes:
            print("self.detected_bboxes length:", len(self.detected_bboxes["labels"]))

        #print('Количество серий уникальных ', len(self.ct_series_data["labels"]))
        print("Series in checkpoint:\n", "\n".join(list(self.ct_series_data["labels"].keys())), sep="")

        #self.artifact_ranges = self.construct_artifact_ranges()

        self.viewer_ui()

        generate_mode = 'images' if args.genimages else ('videos' if args.genvideos else None)

        # generating bboxes with max values nearby
        if args.gen_show_hu:
            self.show_max_bbox_value.set(1)

        if generate_mode is None:
            self.load_ct_series(args.series)
        else:
            if not args.genparent:
                print("Cannot generate images/videos without parent folder (specify --genparent)")
                exit(1)
            self.export_as_images.set(1 if generate_mode == 'images' else 0)  # generate_mode is either "images" or "videos"
            if args.genfile is not None:
                with open(args.genfile, 'r') as f:
                    for line in f:
                        short_folder_key = line.rstrip()
                        self.load_ct_series(os.path.join(args.genparent, short_folder_key))
                        self.save_ct_scan_with_boxes()
            else:
                for folder_key in tqdm.tqdm(self.ct_series_data["labels"].keys()):
                    self.load_ct_series(os.path.join(args.genparent, folder_key))
                    self.save_ct_scan_with_boxes()
            print("Generation is complete. Exiting...")
            exit(0)

    def load_json_file(self, file_path, required=False, default=None):
        if file_path:
            with open(file_path, mode='r', encoding='UTF-8') as file:
                return json.load(file)
        elif required:
            raise ValueError(f"{file_path} is a required path.")
        return default

    def bring_window_to_forefront(self):
        if sys.platform == 'darwin':  # currently only macOS is supported
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')


    def viewer_ui(self):
        self.folder_key_label = ttk.Label(self.root, text="Folder Key")
        self.folder_key_label.grid(row=0, column=1, pady=20)

        self.load_button = ttk.Button(self.root, text="Load CT Series", command=self.load_ct_series)
        self.load_button.grid(row=0, column=2, pady=20)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=1, column=1, padx=10, sticky="ns")

        self.slider = tk.Scale(self.canvas_frame, from_=0, orient="horizontal", command=self.update_image)    
        self.slider.pack(fill="x")

        self.image_canvas = tk.Canvas(self.canvas_frame, height=self.img_size, width=self.img_size, cursor="cross")
        self.image_canvas.pack(pady=10)

        # Bind mouse events for drawing bounding boxes
        self.image_canvas.bind("<ButtonPress-1>", self.on_box_start)
        self.image_canvas.bind("<B1-Motion>", self.on_box_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_box_release)

        self.root.bind("<space>", self.interpolate)

        # Bind scrolling event to series scrolling
        self.image_canvas.bind("<MouseWheel>", self.on_scroll)
        # Bind motion event to the image label
        self.image_canvas.bind("<Motion>", self.show_pixel_value)

        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('a', self.prev_image)
        self.root.bind('d', self.next_image)

        #self.root.bind('<Escape>', self.lose_focus())

        # Controls on left side
        self.left_controls_frame = tk.Frame(self.root)
        self.left_controls_frame.grid(row=1, column=0, padx=10, sticky="n")

        # Boxes interpolation
        self.start_slice_label = ttk.Label(self.left_controls_frame, text="Start Slice:")
        self.start_slice_label.pack(pady=5)
        self.start_box_label = ttk.Label(self.left_controls_frame, text="Start Box:")
        self.start_box_label.pack(pady=5)
        self.end_slice_label = ttk.Label(self.left_controls_frame, text="End Slice:")
        self.end_slice_label.pack(pady=5)
        self.end_box_label = ttk.Label(self.left_controls_frame, text="End Box:")
        self.end_box_label.pack(pady=5)

        self.interpolate_button = ttk.Button(self.left_controls_frame, text="Interpolate Boxes", command=self.interpolate)
        self.interpolate_button.pack(pady=5)
        self.undo_interpolation_button = ttk.Button(self.left_controls_frame, text="Undo interpolation", command=self.undo_interpolation)
        self.undo_interpolation_button.pack(pady=5)

        self.remove_boxes_button = ttk.Button(self.left_controls_frame, text="Remove Boxes in Interval", command=self.remove_range)
        self.remove_boxes_button.pack(pady=5)
        self.undo_remove_boxes_button = ttk.Button(self.left_controls_frame, text="Undo Boxes Removal", command=self.undo_removal)
        self.undo_remove_boxes_button.pack(pady=5)

        self.apply_LoG = tk.IntVar()
        ttk.Checkbutton(self.left_controls_frame, text=f"Apply LoG", variable=self.apply_LoG, command=self.update_image).pack(pady=2)

        # LoG windowing
        #   Window Min
        vcmd_log_min = self.root.register(self.validate_window_level)
        self.log_window_min_frame = ttk.Frame(self.left_controls_frame)
        self.log_window_min_frame.pack(fill="x", pady=5)

        self.log_window_min_label = ttk.Label(self.log_window_min_frame, text="LoG Min Val:")
        self.log_window_min_label.pack(side="left")

        self.log_window_min_entry = ttk.Entry(self.log_window_min_frame, validate="key", validatecommand=(vcmd_log_min, '%P'), width=10)

        self.log_window_min_entry.insert(0, str(self.log_window_min))  # Insert default value
        self.log_window_min_entry.bind("<Return>", self.update_log_window)  # Bind the Enter key
        self.log_window_min_entry.pack(side="left", expand=False, fill="x")
        #   Window Max
        vcmd_log_max = self.root.register(self.validate_window_width)
        self.log_window_max_frame = ttk.Frame(self.left_controls_frame)
        self.log_window_max_frame.pack(fill="x", pady=5)
        
        self.log_window_max_label = ttk.Label(self.log_window_max_frame, text="LoG Max Val:")
        self.log_window_max_label.pack(side="left")

        self.log_window_max_entry = ttk.Entry(self.log_window_max_frame, validate="key", validatecommand=(vcmd_log_max, '%P'), width=10)
        self.log_window_max_entry.insert(0, str(self.log_window_max))  # Insert default value
        self.log_window_max_entry.bind("<Return>", self.update_log_window)  # Bind the Enter key
        self.log_window_max_entry.pack(side="left", expand=False, fill="x")


        # Controls on right side
        self.right_controls_frame = tk.Frame(self.root)
        self.right_controls_frame.grid(row=1, column=2, padx=10, sticky="n")

        ttk.Radiobutton(self.right_controls_frame, state=tk.NORMAL, text="Show MIP", value="mip", variable=self.to_show, command=self.toggle_view).pack(pady=5)
        ttk.Radiobutton(self.right_controls_frame, state=tk.ACTIVE, text="Show original", value="original", variable=self.to_show, command=self.toggle_view).pack(pady=5)

        self.add_window_info = tk.IntVar()
        ttk.Checkbutton(self.right_controls_frame, text=f"Include window info", variable=self.add_window_info).pack(pady=10)

        save_button = ttk.Button(self.right_controls_frame, text="Save current view", command=self.save_current_view)
        save_button.pack(pady=5)
        
        save_video_button = ttk.Button(self.right_controls_frame, text="Save video with boxes", command=self.save_ct_scan_with_boxes)
        save_video_button.pack(pady=5)

        self.export_as_images = tk.IntVar()
        ttk.Checkbutton(self.right_controls_frame, text=f"Export boxes as images", variable=self.export_as_images).pack(pady=2)

        # Label to display pixel value
        self.pixel_value_label = ttk.Label(self.right_controls_frame, text="Pixel Value: N/A")
        self.pixel_value_label.pack(pady=5)

        self.show_max_bbox_value = tk.IntVar()
        ttk.Checkbutton(self.right_controls_frame, text=f"Show max bbox value", variable=self.show_max_bbox_value).pack(pady=2)

        

        # New window adjustment logic
        #   Window Level
        vcmd_level = self.root.register(self.validate_window_level)
        self.window_level_frame = ttk.Frame(self.right_controls_frame)
        self.window_level_frame.pack(fill="x", pady=5)

        self.window_level_label = ttk.Label(self.window_level_frame, text="Window Level:")
        self.window_level_label.pack(side="left")

        self.window_level_entry = ttk.Entry(self.window_level_frame, validate="key", validatecommand=(vcmd_level, '%P'), width=10)

        self.window_level_entry.insert(0, str(self.default_level))  # Insert default value
        self.window_level_entry.bind("<Return>", self.update_windowing)  # Bind the Enter key
        self.window_level_entry.pack(side="left", expand=False, fill="x")
        #   Window Width
        vcmd_width = self.root.register(self.validate_window_width)
        self.window_width_frame = ttk.Frame(self.right_controls_frame)
        self.window_width_frame.pack(fill="x", pady=5)
        
        self.window_width_label = ttk.Label(self.window_width_frame, text="Window Width:")
        self.window_width_label.pack(side="left")

        self.window_width_entry = ttk.Entry(self.window_width_frame, validate="key", validatecommand=(vcmd_width, '%P'), width=10)
        self.window_width_entry.insert(0, str(self.default_width))  # Insert default value
        self.window_width_entry.bind("<Return>", self.update_windowing)  # Bind the Enter key
        self.window_width_entry.pack(side="left", expand=False, fill="x")

        # Window level adjustment
        #self.window_level_label = ttk.Label(self.right_controls_frame, text="Window Level: {self.default_level}")
        #self.window_level_label.pack(pady=5)
        #self.window_level = ttk.Scale(self.right_controls_frame, from_=-2000, to=8000, orient="horizontal", command=self.update_window_level)
        #self.window_level.set(self.default_level)
        #self.window_level.pack(fill="x")

        #self.window_width_label = ttk.Label(self.right_controls_frame, text="Window Width: {self.default_width}")
        #self.window_width_label.pack(pady=5)
        #self.window_width = ttk.Scale(self.right_controls_frame, from_=1, to=5000, orient="horizontal", command=self.update_window_width)
        #self.window_width.set(self.default_width)
        #self.window_width.pack(fill="x")


        tk.Button(self.right_controls_frame, text="Save Current Boxes", fg='#2AC600', command=self.save_boxes).pack(pady=5)

        ttk.Button(self.right_controls_frame, text="Clear boxes for series", command=self.clear_boxes_series).pack(pady=5)
        ttk.Button(self.right_controls_frame, text="Undo clear boxes for series", command=self.undo_clear_boxes_series).pack(pady=5)

        tk.Button(self.right_controls_frame, text="Clear boxes for ckpt", command=self.clear_boxes_checkpoint, fg='#ff542f').pack(pady=5)
        tk.Button(self.right_controls_frame, text="Undo clear boxes for ckpt", command=self.undo_clear_boxes_checkpoint, fg='#ff542f').pack(pady=5)

        self.n_labelled = ttk.Label(self.right_controls_frame, text=f'Labelled series: {len(self.ct_series_data["labels"])}')
        self.n_labelled.pack(pady=3)

        self.text_series_len = ttk.Label(self.right_controls_frame, text=f'Series length:')
        self.text_series_len.pack(pady=3)

        self.text_with_art = ttk.Label(self.right_controls_frame, text=f'Slices with artefacts:')
        self.text_with_art.pack(pady=3)

    def save_ct_scan_with_boxes(self):
        """
        Saves CT scan frames with bounding boxes and series name to a video file.
    
        Parameters:
        - ct_scan: A 3D NumPy array representing the CT scan, with dimensions [frames, height, width].
        - boxes: A list of tuples, each representing a bounding box in the format (frame_index, start_x, end_x, start_y, end_y).
        - series_name: The name of the CT scan series to display on each frame.
        - output_filename_video: The name of the output video file.
        """
        # Define the codec and create VideoWriter object
        def to_window(img):
            minval = int(self.window_level_entry.get()) - (int(self.window_width_entry.get()) / 2)
            maxval = int(self.window_level_entry.get()) + (int(self.window_width_entry.get()) / 2)
            img = ((img - minval) / (maxval - minval) * 255).clip(0, 255).astype('uint8')
            return img

        export_as_images = bool(self.export_as_images.get())
        folder_key = get_folder_key(self.series_path)

        parent_path = [os.path.dirname(os.path.realpath(__file__))]
        if self.generate_output_folder is not None:
            parent_path.append(self.generate_output_folder)

        if export_as_images:
            output_folder = os.path.join(*parent_path, "-".join(f"{get_folder_key(self.series_path, short=True)}_export".split(os.sep)))
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_filename_video = os.path.join(*parent_path, "-".join(f"{get_folder_key(self.series_path, short=True)}_vid.mp4".split(os.sep)))
            os.makedirs(os.path.join(*parent_path), exist_ok=True)

        frame_height, frame_width = self.ct_series[0].shape[0], self.ct_series[0].shape[1]
        
        if not export_as_images:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_filename_video, fourcc, 10.0, (frame_width, frame_height))

        tl = 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness

        # Iterate through each frame in the CT scan
        for current_idx, frame in enumerate(self.ct_series):

            frame = to_window(frame)

            if self.apply_LoG.get():
                frame = self.get_current_img()

            # Convert grayscale frame to BGR for colored drawing
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Redraw bounding boxes if they exist for the current slice    
            if folder_key in self.ct_series_data["labels"]:
                if str(current_idx) in self.ct_series_data["labels"][folder_key]:
                    for bbox in self.ct_series_data["labels"][folder_key][str(current_idx)]["bboxes"]:
                        _, start_x, start_y, end_x, end_y = bbox

                        # ground truth (green)
                        cv2.rectangle(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 1)


                        # showing max value for the box
                        if self.show_max_bbox_value.get():
                            max_value = int(self.ct_series[current_idx][to_interval(start_y-1):to_interval(end_y-1), to_interval(start_x-1):to_interval(end_x-1)].max())
                            label = f'{max_value:,}'.replace(',', ' ')
                            
                            font_color = [0, 0, 255]
                            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                            filler_end_x, filler_end_y = start_x + t_size[0], start_y - t_size[1] - 3
    
                            cv2.rectangle(frame, (int(start_x), int(start_y)), (int(filler_end_x), int(filler_end_y)), (0, 255, 255), -1, cv2.LINE_AA)  # filled
                            cv2.putText(frame, label, (int(start_x), int(start_y) - 2), 0, 0.3, font_color, thickness=1, lineType=cv2.LINE_AA)


            # drawing bboxes from predictions (generated by a trained model)
            if self.detected_bboxes:
                if folder_key in self.detected_bboxes["labels"]:
                    filename = self.idx_to_name[current_idx]
                    if filename in self.detected_bboxes["labels"][folder_key]:
                        for bbox in self.detected_bboxes["labels"][folder_key][filename]["bboxes"]:
                            _, start_x, start_y, end_x, end_y, confidence = bbox
                            # yellow R 255 G 255 B 0, in cv2 - BGR
                            cv2.rectangle(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 255), 1)

                            # confidence text
                            font_color = [0, 0, 255]
                            label = f'{confidence:.2f}'
                            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                            filler_end_x, filler_end_y = start_x + t_size[0], start_y - t_size[1] - 3

                            cv2.rectangle(frame, (int(start_x), int(start_y)), (int(filler_end_x), int(filler_end_y)), (0, 255, 255), -1, cv2.LINE_AA)  # filled
                            cv2.putText(frame, label, (int(start_x), int(start_y) - 2), 0, 0.3, font_color, thickness=1, lineType=cv2.LINE_AA)
    
            # Put the series name on each frame
            cv2.putText(frame, get_folder_key(self.series_path, short=True), (3, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
            # Write the frame into the file 'output_filename_video'
            if export_as_images:
                output_filename_image = os.path.join(output_folder, str(current_idx))
                if self.generate_compress:
                    cv2.imwrite(output_filename_image+".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                else:
                    cv2.imwrite(output_filename_image+".png", frame)
                
            else:
                out.write(frame)
    
        if not export_as_images:
            # Release everything when job is finished
            cv2.destroyAllWindows()
            out.release()

        print(f"Successfully saved the {'images' if export_as_images else 'video'} to", output_folder if export_as_images else output_filename_video)

    def update_labelled_series(self):
        self.n_labelled.config(text=f'Labelled series: {len(self.ct_series_data["labels"])}')

    def update_labelled_slices(self):
        folder_key = get_folder_key(self.series_path)
        n = len(self.ct_series_data["labels"][folder_key]) if folder_key in self.ct_series_data["labels"] else 0
        self.text_with_art.config(text=f'Slices with artefacts: {n}')

    def update_log_window(self, _=None):
        self.log_window_min = int(self.log_window_min_entry.get())
        self.log_window_max = int(self.log_window_max_entry.get())

        self.lose_focus()
        self.update_image()

    def update_windowing(self, _=None):
        try:
            window_level = int(self.window_level_entry.get())
            window_width = int(self.window_width_entry.get())
        except:
            window_level = self.default_level
            window_width = self.default_width

        minval = window_level - (window_width / 2)
        maxval = window_level + (window_width / 2)
        if self.ct_series is not None:
            self.ct_series_windowed = ((self.ct_series - minval) / (maxval - minval) * 255).clip(0, 255).astype('uint8')
        if self.mip_series is not None:
            self.mip_series_windowed = ((self.mip_series - minval) / (maxval - minval) * 255).clip(0, 255).astype('uint8')
        self.lose_focus()
        self.update_image()

    def save_boxes(self):
        with open(self.output_path, mode="w", encoding="UTF-8") as output_file:
            json.dump(self.ct_series_data, output_file, ensure_ascii=False, indent=4)
        self.update_labelled_series()
        print('Saved to', self.output_path)

    def clear_boxes_checkpoint(self):
        if not self.ct_series_data["labels"]:
            return
        self.old_ckpt_labels = self.ct_series_data["labels"]
        self.ct_series_data["labels"] = {}
        self.update_labelled_series()
        self.update_labelled_slices()
        self.update_image()

    def undo_clear_boxes_checkpoint(self):
        if not self.old_ckpt_labels:
            return
        self.ct_series_data["labels"] = self.old_ckpt_labels
        self.old_ckpt_labels = {}
        self.update_labelled_series()
        self.update_labelled_slices()
        self.update_image()

    def clear_boxes_series(self):
        folder_key = get_folder_key(self.series_path)
        if folder_key not in self.ct_series_data["labels"]:
            return
        self.old_series_labels = self.ct_series_data["labels"][folder_key]    
        #self.ct_series_data["labels"][folder_key] = {}
        self.ct_series_data["labels"].pop(folder_key)
        self.update_labelled_series()
        self.update_labelled_slices()
        self.update_image()

    def undo_clear_boxes_series(self):
        if not self.old_series_labels:
            return
        folder_key = get_folder_key(self.series_path)
        self.ct_series_data["labels"][folder_key] = self.old_series_labels
        self.old_series_labels = None
        self.update_labelled_series()
        self.update_labelled_slices()
        self.update_image()

    def lose_focus(self, event=None):
        self.root.focus_set()

    def validate_window_level(self, value):
        if not value:
            # this accounts for the field being empty which is a valid state (like deleting all content)
            return True
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            try:
                num = int(value)
            except ValueError:
                # Just in case, though if the checks above are true, this should not happen
                return False
            return True
        return False

    def validate_window_width(self, value):
        if not value:
            # this accounts for the field being empty which is a valid state (like deleting all content)
            return True
        if value.isdigit():
            try:
                num = int(value)
            except:
                return False
            if 0 < num:
                return True
        return False

    def undo_interpolation(self):
        if not self.last_interpolation:
            return
        for current_idx, (artifact_id, start_x, start_y, end_x, end_y) in self.last_interpolation:
            folder_key = get_folder_key(self.series_path)
            bbox = [artifact_id, start_x, start_y, end_x, end_y]
            self.delete_box(folder_key, current_idx, bbox)
        self.update_image()
        self.last_interpolation = []

    def undo_removal(self):
        if not self.last_removal:
            return
        for current_idx, (artifact_id, start_x, start_y, end_x, end_y) in self.last_removal:
            folder_key = get_folder_key(self.series_path)
            bbox = [artifact_id, start_x, start_y, end_x, end_y]
            self.add_box(folder_key, current_idx, bbox)
        self.update_image()
        self.last_removal = []

    def interpolate(self, event=None):
        if not self.inter_box_1 or not self.inter_box_2:
            return

        # Empty interpolation history
        self.last_interpolation = []

        # Unpack the bounding boxes and their slice indices
        idx1, (start_x1, start_y1, end_x1, end_y1) = self.inter_box_1
        idx2, (start_x2, start_y2, end_x2, end_y2) = self.inter_box_2
    
        # Determine the step (positive or negative) based on the order of indices
        step = 1 if idx1 < idx2 else -1
    
        # Calculate the number of slices to interpolate
        num_slices = abs(idx2 - idx1) - 1
    
        # Loop through each slice and interpolate the bounding box
        for i in range(1, num_slices + 1):
            t = i / (num_slices + 1)  # Normalized position between the two bounding boxes
    
            # Linearly interpolate the coordinates
            start_x = round(start_x1 + (start_x2 - start_x1) * t, 1)
            start_y = round(start_y1 + (start_y2 - start_y1) * t, 1)
            end_x = round(end_x1 + (end_x2 - end_x1) * t, 1)
            end_y = round(end_y1 + (end_y2 - end_y1) * t, 1)

            current_idx = idx1 + i * step
            folder_key = get_folder_key(self.series_path)
            bbox = [self.artifact_id, start_x, start_y, end_x, end_y]
            self.add_box(folder_key, current_idx, bbox)
            self.last_interpolation.append((current_idx, bbox))

        self.inter_box_1 = None
        self.inter_box_2 = None
        self.start_slice_label.config(text=f"Start Slice:")
        self.start_box_label.config(text=f"Start Box:")
        self.end_slice_label.config(text=f"End Slice:")
        self.end_box_label.config(text=f"End Box:")
        self.update_image()
        self.lose_focus()

    def remove_range(self):
        if not self.inter_box_1 or not self.inter_box_2:
            return

        # Empty interpolation history
        self.last_removal = []

        # Unpack the bounding boxes and their slice indices
        idx1, (start_x1, start_y1, end_x1, end_y1) = self.inter_box_1
        idx2, (start_x2, start_y2, end_x2, end_y2) = self.inter_box_2
    
        # Determine the step (positive or negative) based on the order of indices
        step = 1 if idx1 < idx2 else -1
    
        # Calculate the number of slices to interpolate
        num_slices = abs(idx2 - idx1) - 1

        folder_key = get_folder_key(self.series_path)
    
        # Loop through each slice and interpolate the bounding box
        for i in range(0, num_slices):
            t = i / (num_slices + 1)  # Normalized position between the two bounding boxes
    
            # Linearly interpolate the coordinates
            start_x = round(start_x1 + (start_x2 - start_x1) * t, 1)
            start_y = round(start_y1 + (start_y2 - start_y1) * t, 1)
            end_x = round(end_x1 + (end_x2 - end_x1) * t, 1)
            end_y = round(end_y1 + (end_y2 - end_y1) * t, 1)

            current_idx = idx1 + i * step
            
            if folder_key in self.ct_series_data["labels"] and str(current_idx) in self.ct_series_data["labels"][folder_key]:
                for bbox in self.ct_series_data["labels"][folder_key][str(current_idx)]["bboxes"]:
                    art_id, in_start_x, in_start_y, in_end_x, in_end_y = bbox
                    if start_x < in_start_x and start_y < in_start_y and end_x > in_end_x and end_y > in_end_y:
                        self.delete_box(folder_key, current_idx, bbox)
                        self.last_removal.append((current_idx, bbox))

        # delete inter boxes
        self.delete_box(folder_key, idx1, [self.artifact_id] + self.inter_box_1[1])
        self.delete_box(folder_key, idx2, [self.artifact_id] + self.inter_box_2[1])

        self.inter_box_1 = None
        self.inter_box_2 = None
        self.start_slice_label.config(text=f"Start Slice:")
        self.start_box_label.config(text=f"Start Box:")
        self.end_slice_label.config(text=f"End Slice:")
        self.end_box_label.config(text=f"End Box:")
        self.update_image()
        self.lose_focus()


    def on_box_start(self, event):
        # Start drawing a bounding box
        self.start_x = self.image_canvas.canvasx(event.x)
        self.start_y = self.image_canvas.canvasy(event.y)
        self.current_box = self.image_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_box_drag(self, event):
        # Update the size of the bounding box
        end_x = self.image_canvas.canvasx(event.x)
        end_y = self.image_canvas.canvasy(event.y)

        target_list = self.get_current_viewtype()

        if 0 <= end_x < target_list[0].shape[1] and 0 <= end_y < target_list[0].shape[0]:
            self.image_canvas.coords(self.current_box, self.start_x, self.start_y, end_x, end_y)
        
    def on_rectangle_click(self, event, rect_id):
        current_idx = self.slider.get()

        start_x, start_y, end_x, end_y = self.image_canvas.coords(rect_id)

        self.image_canvas.delete(rect_id)  # remove from the canvas

        if str(event.type) == '5':
        #if square <= self.bbox_square_threshold:

            # if event is "release"
            print("on_rectangle_click triggered with release")
            return

        folder_key = get_folder_key(self.series_path)
        bbox = [self.artifact_id, start_x, start_y, end_x, end_y]
        self.delete_box(folder_key, current_idx, bbox)
        self.current_bbox_ids.remove(rect_id)

    def delete_box(self, folder_key, slice_idx, bbox):
        """
        bbox = [self.artifact_id, start_x, start_y, end_x, end_y]
        """
        try:
            # Assuming self.ct_series_data is a nested dictionary and you're trying to access and modify a list within it
            folder_key = get_folder_key(self.series_path)
            if folder_key in self.ct_series_data["labels"] and str(slice_idx) in self.ct_series_data["labels"][folder_key]:
                
                if bbox in self.ct_series_data["labels"][folder_key][str(slice_idx)]["bboxes"]:
                    self.ct_series_data["labels"][folder_key][str(slice_idx)]["bboxes"].remove(bbox)
                    if len(self.ct_series_data["labels"][folder_key][str(slice_idx)]["bboxes"]) == 0:
                        del self.ct_series_data["labels"][folder_key][str(slice_idx)]
                    if len(self.ct_series_data["labels"][folder_key]) == 0:
                        del self.ct_series_data["labels"][folder_key]

                    self.update_labelled_slices()
                else:
                    print(f"Error: Specified bounding box not found in the list ({bbox})")
            else:
                print("Error: Folder key or current index not found in the data.")
        except KeyError as e:
            print(f"Key error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def add_box(self, folder_key, slice_idx, bbox):
        """
        bbox = [self.artifact_id, start_x, start_y, end_x, end_y]
        """
        # Store the bounding box coordinates for the current slice
        try:
            # Ensure that the folder key exists in the dictionary
            if folder_key not in self.ct_series_data["labels"]:
                self.ct_series_data["labels"][folder_key] = {}
        
            # Ensure that the current index exists in the nested dictionary
            if str(slice_idx) not in self.ct_series_data["labels"][folder_key]:
                self.ct_series_data["labels"][folder_key][str(slice_idx)] = {"bboxes": []}
        
            # Add the bbox to the list of bboxes for the current index
            self.ct_series_data["labels"][folder_key][str(slice_idx)]["bboxes"].append(bbox)
            self.ct_series_data["labels"][folder_key][str(slice_idx)]["filename"] = self.idx_to_name[slice_idx]

            self.update_labelled_slices()

        except KeyError as e:
            print(f"Key error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    def on_box_release(self, event):
        current_idx = self.slider.get()
        # Finalize the bounding box
        end_x = self.image_canvas.canvasx(event.x)
        end_y = self.image_canvas.canvasy(event.y)

        # ensuring the box is inside the image and start coordinate is less than the end coordinate
        final_start_x = max(0, min(min(self.start_x, end_x), self.img_size - 1))
        final_end_x   = max(0, min(max(self.start_x, end_x), self.img_size - 1))
        final_start_y = max(0, min(min(self.start_y, end_y), self.img_size - 1))
        final_end_y   = max(0, min(max(self.start_y, end_y), self.img_size - 1))

        if final_start_x != self.start_x or final_end_x != end_x or \
           final_start_y != self.start_y or final_end_y != end_y:
            self.image_canvas.delete(self.current_box)  # remove bad rectangle
            self.current_box = self.image_canvas.create_rectangle(final_start_x, final_start_y, final_end_x, final_end_y, outline="red")

        square = abs(final_end_x - final_start_x) * abs(final_end_y - final_start_y)
        if square <= self.bbox_square_threshold:
            #print(f'square of {(final_start_x, final_start_y, final_end_x, final_end_y)} is less than threshold ({self.bbox_square_threshold})')
            self.image_canvas.delete(self.current_box)  # remove from the canvas
            return

        #target_list = self.get_current_viewtype()

        self.image_canvas.coords(self.current_box, final_start_x, final_start_y, final_end_x, final_end_y)
        self.image_canvas.tag_bind(self.current_box, '<Button-2>', lambda event, rect_id=self.current_box: self.on_rectangle_click(event, rect_id))
        self.image_canvas.tag_bind(self.current_box, '<Control-Button-2>', lambda event, rect_id=self.current_box: self.set_interpolation_box(event, rect_id, 1))
        self.image_canvas.tag_bind(self.current_box, '<Option-Button-2>', lambda event, rect_id=self.current_box: self.set_interpolation_box(event, rect_id, 2))

        tl = 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness

        # showing max value for the box
        if self.show_max_bbox_value.get():
            max_value = int(self.ct_series[current_idx][to_interval(final_start_y-1):to_interval(final_end_y-1), to_interval(final_start_x-1):to_interval(final_end_x-1)].max())
            formatted_max_value = f'{max_value:,}'.replace(',', ' ')
            # h=12, w=7 for font=("Courier", 9)
            semiwidth = len(formatted_max_value) * 7 // 2
            offset_x = semiwidth - end_x if end_x < semiwidth else (semiwidth - (self.img_size - final_end_x) if self.img_size - final_end_x < semiwidth else 0)
            offset_y = 7
            text_id = self.image_canvas.create_text(final_end_x + offset_x, final_start_y - offset_y, text=formatted_max_value, fill="red", font=("Courier", 9))
            self.current_bbox_ids.add(text_id)

        
        folder_key = get_folder_key(self.series_path)
        bbox = [self.artifact_id, final_start_x, final_start_y, final_end_x, final_end_y]
        self.add_box(folder_key, current_idx, bbox)
        self.current_bbox_ids.add(self.current_box)

    def set_interpolation_box(self, event, rect_id, box_n):
        current_idx = self.slider.get()

        start_x, start_y, end_x, end_y = self.image_canvas.coords(rect_id)

        self.image_canvas.itemconfig(rect_id, outline="green2")

        if box_n == 1:
            self.inter_box_1 = (current_idx, [start_x, start_y, end_x, end_y])
            self.start_slice_label.config(text=f"Start Slice: {current_idx}")
            self.start_box_label.config(text=f"Start Box: \n{start_x}, {start_y}, {end_x}, {end_y}")
        elif box_n == 2:
            self.inter_box_2 = (current_idx, [start_x, start_y, end_x, end_y])
            self.end_slice_label.config(text=f"End Slice: {current_idx}")
            self.end_box_label.config(text=f"End Box: \n{start_x}, {start_y}, {end_x}, {end_y}")
        else:
            raise Exception(f"unknown box_n parameter (must be 1 or 2, got {box_n})")


    def get_current_img(self):
        slice_idx = self.slider.get()
        if self.apply_LoG.get():
            target_list = self.get_current_viewtype(windowed=self.windowed_LoG)
            slice_idx = min(len(target_list)-1, slice_idx)
            ct_slice = target_list[slice_idx]
            if self.replace_bg:
                ct_slice = ct_slice.copy()
                ct_slice[ct_slice == -2048.0] = -1000.
            self.current_log_image = LoG_filter(ct_slice, self.sigma)
            minval = self.log_window_min
            maxval = self.log_window_max  # 700 - 1200
            self.current_log_image = ((self.current_log_image - minval) / (maxval - minval) * 255).clip(0, 255).astype('uint8')
            ct_slice = self.current_log_image
        else:
            target_list = self.get_current_viewtype()
            slice_idx = min(len(target_list)-1, slice_idx)
            ct_slice = target_list[slice_idx]
        return ct_slice

    def save_current_view(self):
        if self.ct_series is None:
            return
        img = self.get_current_img()
        rgb_img = np.stack((img,)*3, axis=-1)

        slice_idx = self.slider.get()
        #img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "-".join(f"{get_folder_key(self.series_path)}_{slice_idx}.png".split(os.sep)))
        mip = "_mip" if self.to_show.get() == "mip" else ""
        window_info = f"_L={int(self.window_level_entry.get())}_W={int(self.window_width_entry.get())}" if self.add_window_info.get() else ""
        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "-".join(f"{get_folder_key(self.series_path, short=True)}_{slice_idx}{mip}{window_info}.png".split(os.sep)))

        folder_key = get_folder_key(self.series_path)
        current_idx = self.slider.get()
        tl = 1  # line/font thickness
        tf = max(tl - 1, 1)  # font thickness

        if folder_key in self.ct_series_data["labels"]:
            if str(current_idx) in self.ct_series_data["labels"][folder_key]:
                for bbox in self.ct_series_data["labels"][folder_key][str(current_idx)]["bboxes"]:
                    _, start_x, start_y, end_x, end_y = bbox

                    cv2.rectangle(rgb_img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 255), 1)
                    
                    # showing max value for the box
                    if self.show_max_bbox_value.get():
                        max_value = int(self.ct_series[current_idx][to_interval(start_y-1):to_interval(end_y-1), to_interval(start_x-1):to_interval(end_x-1)].max())
                        label = f'{max_value:,}'.replace(',', ' ')
                        
                        font_color = [0, 0, 255]
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        filler_end_x, filler_end_y = start_x + t_size[0], start_y - t_size[1] - 3

                        cv2.rectangle(rgb_img, (int(start_x), int(start_y)), (int(filler_end_x), int(filler_end_y)), (0, 255, 255), -1, cv2.LINE_AA)  # filled
                        cv2.putText(rgb_img, label, (int(start_x), int(start_y) - 2), 0, 0.3, font_color, thickness=1, lineType=cv2.LINE_AA)


        cv2.imwrite(img_path, rgb_img)
        print(f"Image saved at: {img_path}")

    def on_scroll(self, event):
        # Check the platform 
        if event.delta:
            # For Windows and MacOS
            delta = event.delta
            if sys.platform == 'darwin':  # MacOS has a reversed scroll
                delta = -event.delta
            increment = 1 if delta > 0 else -1
        else:
            # For Linux
            increment = 1 if event.num == 4 else -1
    
        new_value = self.slider.get() + increment
        if 0 <= new_value < len(self.ct_series):  # Ensure the new value is within bounds
            self.slider.set(new_value)
        self.update_image()

    def prev_image(self, event):
        target_list = self.get_current_viewtype()
        new_value = self.slider.get() - 1
        if 0 <= new_value < len(target_list):  # Ensure the new value is within bounds
            self.slider.set(new_value)
            self.update_image()

    def next_image(self, event):
        target_list = self.get_current_viewtype()
        new_value = self.slider.get() + 1
        if 0 <= new_value < len(target_list):  # Ensure the new value is within bounds
            self.slider.set(new_value)
            self.update_image()

    def get_current_viewtype(self, windowed=True):
        if self.to_show.get() == "original":
            return self.ct_series_windowed if windowed else self.ct_series
        if self.to_show.get() == "mip":
            return self.mip_series_windowed  if windowed else self.mip_series
        raise NotImplemented("unknown view: "+self.to_show.get())

    def show_pixel_value(self, event):
        # Extract x and y coordinates of the mouse cursor on the image
        x, y = event.x, event.y

        target_list = self.get_current_viewtype(windowed=False)

        if target_list is not None:
            # Check if the coordinates are within bounds
            if 0 <= x < target_list[0].shape[1] and 0 <= y < target_list[0].shape[0]:
                slice_idx = self.slider.get()
                if self.apply_LoG.get() and self.current_log_image is not None:
                    ct_slice = self.current_log_image 
                else:
                    ct_slice = target_list[slice_idx]
                pixel_value = ct_slice[y, x]
                self.pixel_value_label.config(text=f"Pixel Value: {pixel_value:.2f} {'(LoG)' if self.apply_LoG.get() else ''}")
            else:
                self.pixel_value_label.config(text="Pixel Value: N/A")

    def toggle_view(self, _=None):
        target_list = self.get_current_viewtype()
        if self.to_show.get() == "mip":
            self.interpolate_button.config(state="disable")
            self.undo_interpolation_button.config(state="disable")
            self.remove_boxes_button.config(state="disable")
            self.undo_remove_boxes_button.config(state="disable")
        elif self.to_show.get() == "original":
            self.interpolate_button.config(state="normal")
            self.undo_interpolation_button.config(state="normal")
            self.remove_boxes_button.config(state="normal")
            self.undo_remove_boxes_button.config(state="normal")

        self.slider.config(to=len(target_list)-1)
        self.update_image()

    def update_image(self, _=None):
        if self.ct_series is None:
            return

        # Get the current CT image
        img = self.get_current_img()
        img = Image.fromarray(img)
        width, height = img.size

        if width != self.img_size or height != self.img_size:
            img = img.resize((self.img_size, self.img_size))

        # Convert the image to a format suitable for Tkinter
        self.tk_img = ImageTk.PhotoImage(image=img)
        ####
        ##### If an image is already displayed on the canvas, remove it
        ####if hasattr(self, 'canvas_img_id'):
        ####    self.image_canvas.delete(self.canvas_img_id)
        ####
        ##### Display the new image on the canvas
        ####self.canvas_img_id = self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        # Delete old bboxes
        for rect_id in self.current_bbox_ids:
            self.image_canvas.delete(rect_id)
        self.current_bbox_ids.clear()
        
        # Check if the canvas image item has been created
        if not hasattr(self, 'canvas_img_id'):
            # If not, create the canvas image item and store its ID
            self.canvas_img_id = self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        else:
            # If the canvas image item exists, update its image
            self.image_canvas.itemconfig(self.canvas_img_id, image=self.tk_img)

        # Update the canvas scroll region to accommodate the new image
        #self.image_canvas.config(height=height, width=width)

        # Redraw bounding boxes if they exist for the current slice
        folder_key = get_folder_key(self.series_path)
        current_idx = self.slider.get()

        if folder_key in self.ct_series_data["labels"]:
            if str(current_idx) in self.ct_series_data["labels"][folder_key]:
                for bbox in self.ct_series_data["labels"][folder_key][str(current_idx)]["bboxes"]:
                    _, start_x, start_y, end_x, end_y = bbox
                    # properly processing boxes selected for interpolation
                    if self.inter_box_1 is not None and bbox[1:] == self.inter_box_1[1] and current_idx == self.inter_box_1[0] or \
                       self.inter_box_2 is not None and bbox[1:] == self.inter_box_2[1] and current_idx == self.inter_box_2[0]:
                        color = "green2"
                    else:
                        color = "red"
                    self.current_box = self.image_canvas.create_rectangle(start_x, start_y, end_x, end_y, outline=color)
                    self.image_canvas.tag_bind(self.current_box, '<Button-2>', lambda event, rect_id=self.current_box: self.on_rectangle_click(event, rect_id))
                    self.image_canvas.tag_bind(self.current_box, '<Control-Button-2>', lambda event, rect_id=self.current_box: self.set_interpolation_box(event, rect_id, 1))
                    self.image_canvas.tag_bind(self.current_box, '<Option-Button-2>', lambda event, rect_id=self.current_box: self.set_interpolation_box(event, rect_id, 2))
                    self.current_bbox_ids.add(self.current_box)
                    # showing max value for the box
                    if self.show_max_bbox_value.get():
                        max_value = int(self.ct_series[current_idx][to_interval(start_y-1):to_interval(end_y-1), to_interval(start_x-1):to_interval(end_x-1)].max())
                        formatted_max_value = f'{max_value:,}'.replace(',', ' ')
                        # h=12, w=7 for font=("Courier", 9)
                        semiwidth = len(formatted_max_value) * 7 // 2
                        offset_x = semiwidth - end_x if end_x < semiwidth else (semiwidth - (self.img_size - end_x) if self.img_size - end_x < semiwidth else 0)
                        offset_y = 7
                        text_id = self.image_canvas.create_text(end_x + offset_x, start_y - offset_y, text=formatted_max_value, fill="red", font=("Courier", 9))
                        self.current_bbox_ids.add(text_id)




    def load_ct_series(self, path=None):
        if not path:
            path = filedialog.askdirectory(title="Select Folder Containing CT Series", initialdir=os.path.realpath(__file__))
            self.root.focus_set()
            self.root.lift()
            if not path:
                return


        ct_series_old = self.ct_series
        series_path_old = self.series_path
        #artifact_ranges_old = self.artifact_ranges
        self.ct_series = []
        self.series_path = None
        self.is_numpy = False
        self.root.lift()

        series = {} # SeriesInstanceUID


        for root_dir, _, files in os.walk(path):
            files = list(filter(lambda x: x[0] != '.', files))
            if 'npy' in root_dir:
                images = []  # a list of tuples (idx, filename, img_array)
                self.is_numpy = True
                for file in files:
                    image = np.load(os.path.join(root_dir, file))
                    slice_idx = int(file.split('.')[0])
                    images.append((slice_idx, file, image))
                self.series_path = root_dir
            else:
                for file in files:
                    try:
                        dcm = pydicom.dcmread(os.path.join(root_dir, file))
                        if dcm[0x00080060].value != 'CT': # checking modality
                            continue
                        #if hasattr(dcm, "SliceLocation"):
                        if not hasattr(dcm, "ImagePositionPatient"):
                            continue
                        if dcm.SeriesInstanceUID in series:
                            series[dcm.SeriesInstanceUID].append((file, dcm))
                        else:
                            series[dcm.SeriesInstanceUID] = [(file, dcm)]
                        if not self.series_path:
                            self.series_path = root_dir
                    except Exception as e:
                        print(f'Could not read file {file}, error:', e)

        if self.is_numpy:
            # idk why but .npy series start from neck (0 slice) and end up in stomach (last slice)
            # that is why these sequences are sorted in reverse order
            images.sort(key=lambda x: x[0], reverse=True)
            self.ct_series = [img for idx, filename, img in images]
            self.idx_to_name = {i: filename for i, (idx, filename, img) in enumerate(images)}
            self.name_to_idx = {filename: i for i, (idx, filename, img) in enumerate(images)}
        else:
            if not series:
                showerror("Error", f"No series in {path} has been found")
                self.ct_series = ct_series_old
                self.series_path = series_path_old
                #self.artifact_ranges = artifact_ranges_old
                return
    
            chosen_series = ask_option(self.root, list(series.keys()), "Choose the series") if len(series) > 1 else list(series.keys())[0]
            if chosen_series is None:
                self.ct_series = ct_series_old
                self.series_path = series_path_old
                #self.artifact_ranges = artifact_ranges_old
                return
            series[chosen_series].sort(key=lambda x: float(x[1].ImagePositionPatient[2]))
            self.idx_to_name = {i: filename for i, (filename, dcm) in enumerate(series[chosen_series])}
            self.name_to_idx = {filename: i for i, (filename, dcm) in enumerate(series[chosen_series])}
            series[chosen_series] = [pydicom.pixel_data_handlers.apply_modality_lut(dcm.pixel_array, dcm).astype('float32') for filename, dcm in series[chosen_series]]
            for image in series[chosen_series]:
                # Making all paddings have the same value
                image[image.astype(int) == -3024] = -2048.
            self.ct_series = series[chosen_series]

        self.ct_series = np.stack(self.ct_series, axis=0)
        
        self.mip_series = [createMIP(np.stack(self.ct_series, axis=0))]
        self.mip_series = np.stack(self.mip_series, axis=0)

        self.update_windowing()

        target_list = self.get_current_viewtype()

        self.slider.config(to=len(target_list)-1)

        folder_key = get_folder_key(self.series_path)

        if self.series_labels:
            possible_artifacts = '\n'.join(self.series_labels[get_folder_key(self.series_path)]) \
                if get_folder_key(self.series_path) in self.series_labels else "not available"
        else:
            possible_artifacts = "not available"

        self.folder_key_label.config(text=get_folder_key(self.series_path)+'\n'+"Possible artifacts: \n"+possible_artifacts, font=("Courier", 12))

        self.text_series_len.config(text=f'Series length: {len(self.ct_series)}')
        self.update_labelled_slices()

        self.old_series_labels = None
        self.update_image()
        self.root.focus_set()
