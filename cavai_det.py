import os
import sys
import json
import pydicom
import argparse
import tkinter as tk
import numpy as np
from tkinter import filedialog, ttk
from tkinter.messagebox import showinfo, showerror, showwarning
from tkinter import simpledialog, Toplevel, Listbox
from PIL import Image, ImageTk



def get_folder_key(path, last_n=5, short=False, short_cut=3):
    '''
    Function returns key for a specific folder path (last_n folders in path, excludes npy)
    '''
    f_path_list = path.split(os.sep)
    if short:
        return os.sep.join(path.split(os.sep)[-last_n:-short_cut]) if 'npy' not in path else \
                      os.sep.join(path.split(os.sep)[-last_n-1:-1-short_cut])
    return os.sep.join(path.split(os.sep)[-last_n:]) if 'npy' not in path else \
                      os.sep.join(path.split(os.sep)[-last_n-1:-1])

def createMIP(np_img, slices_num=15):
    return np.max(np_img, axis=0)

# output format
# class x-center y-center width height


class CustomDialog(simpledialog.Dialog):

    def __init__(self, parent, title, options):
        self.options = options
        super().__init__(parent, title=title)

    def body(self, master):
        self.listbox = Listbox(master)
        for option in self.options:
            self.listbox.insert(tk.END, option)
        self.listbox.pack()
        return self.listbox

    def apply(self):
        self.result = self.listbox.get(tk.ACTIVE)

def ask_option(parent, options, title="Choose an option"):
    d = CustomDialog(parent, title=title, options=options)
    return d.result



"""
self.ct_series_data = {
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
"""


class CTViewer:
    def __init__(self, root, series_path, labels_path, checkpoint_path, output_path):
        self.root = root
        self.root.title("CAVAI")

        self.bring_window_to_forefront()

        self.is_numpy = False
        self.to_show = tk.StringVar(value="original")
        self.artifact_id = 0
        self.ct_series = []
        self.mip_series = []

        # format: (slice_idx, [start_x, start_y, end_x, end_y])
        self.inter_box_1 = None
        self.inter_box_2 = None
        self.last_interpolation = []

        self.bbox_square_threshold = 20

        self.series_path = series_path
        self.output_path = output_path
        
        self.series_labels = self.load_json_file(labels_path)
        #self.artifact_options = self.load_json_file(artifacts_path, required=True)
        #self.final_labelling = self.load_json_file(checkpoint_path, default={})

        self.ct_series_data = self.load_json_file(checkpoint_path, default={"img_size": 512, "labels": {}})

        self.img_size = self.ct_series_data["img_size"]

        #print('Количество серий уникальных ', len(self.ct_series_data["labels"]))
        print("Series in checkpoint:\n", "\n".join(list(self.ct_series_data["labels"].keys())))

        #self.artifact_ranges = self.construct_artifact_ranges()

        self.viewer_ui()
        self.load_ct_series(series_path)

    def load_json_file(self, file_path, required=False, default=None):
        if file_path:
            with open(file_path, mode='r', encoding='UTF-8') as file:
                return json.load(file)
        elif required:
            raise ValueError(f"{file_path} is a required path.")
        return default

    def bring_window_to_forefront(self):
        if sys.platform == 'darwin':  # macOS
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')


    def viewer_ui(self):
        self.folder_key_label = ttk.Label(self.root, text="Folder Key")
        self.folder_key_label.grid(row=0, column=0, pady=20)

        self.load_button = ttk.Button(self.root, text="Load CT Series", command=self.load_ct_series)
        self.load_button.grid(row=0, column=1, pady=20)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=1, column=0, padx=10, sticky="ns")

        self.slider = tk.Scale(self.canvas_frame, from_=0, orient="horizontal", command=self.update_image)    
        self.slider.pack(fill="x")

        self.image_canvas = tk.Canvas(self.canvas_frame, height=self.img_size, width=self.img_size, cursor="cross")
        self.image_canvas.pack(pady=20)

        # Bind mouse events for drawing bounding boxes
        self.image_canvas.bind("<ButtonPress-1>", self.on_box_start)
        self.image_canvas.bind("<B1-Motion>", self.on_box_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_box_release)

        # Bind scrolling event to series scrolling
        self.image_canvas.bind("<MouseWheel>", self.on_scroll)
        # Bind motion event to the image label
        self.image_canvas.bind("<Motion>", self.show_pixel_value)

        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)

        #self.root.bind('<Escape>', self.lose_focus())

        # Controls on the side
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.grid(row=1, column=1, padx=10, sticky="n")

        ttk.Radiobutton(self.controls_frame, state=tk.NORMAL, text="Show MIP", value="mip", variable=self.to_show, command=self.toggle_view).pack(pady=5)
        ttk.Radiobutton(self.controls_frame, state=tk.ACTIVE, text="Show original", value="original", variable=self.to_show, command=self.toggle_view).pack(pady=5)

        self.add_window_info = tk.IntVar()
        ttk.Checkbutton(self.controls_frame, text=f"Include window info", variable=self.add_window_info).pack(pady=10)

        save_button = ttk.Button(self.controls_frame, text="Save current view", command=self.save_current_view)
        save_button.pack(pady=20)

        # Label to display pixel value
        self.pixel_value_label = ttk.Label(self.controls_frame, text="Pixel Value: N/A")
        self.pixel_value_label.pack(pady=20)

        # Window level adjustment
        default_level = 0
        self.window_level_label = ttk.Label(self.controls_frame, text="Window Level: {default_level}")
        self.window_level_label.pack(pady=5)
        self.window_level = ttk.Scale(self.controls_frame, from_=-2000, to=8000, orient="horizontal", command=self.update_window_level)
        self.window_level.set(default_level)
        self.window_level.pack(fill="x")

        default_width = 1200
        self.window_width_label = ttk.Label(self.controls_frame, text="Window Width: {default_width}")
        self.window_width_label.pack(pady=5)
        self.window_width = ttk.Scale(self.controls_frame, from_=1, to=5000, orient="horizontal", command=self.update_window_width)
        self.window_width.set(default_width)
        self.window_width.pack(fill="x")

        save_button = ttk.Button(self.controls_frame, text="Save current view", command=self.save_current_view)
        save_button.pack(pady=20)

        vcmd = self.root.register(self.validate_input)
        
        # Boxes interpolation
        self.start_slice_label = ttk.Label(self.controls_frame, text="Start Slice:")
        self.start_slice_label.pack(pady=5)
        self.start_box_label = ttk.Label(self.controls_frame, text="Start Box:")
        self.start_box_label.pack(pady=5)
        self.end_slice_label = ttk.Label(self.controls_frame, text="End Slice:")
        self.end_slice_label.pack(pady=5)
        self.end_box_label = ttk.Label(self.controls_frame, text="End Box:")
        self.end_box_label.pack(pady=5)

        ttk.Button(self.controls_frame, text="Interpolate Boxes", command=self.interpolate).pack(pady=20)
        ttk.Button(self.controls_frame, text="Undo interpolation", command=self.undo_interpolation).pack(pady=20)


        ttk.Button(self.controls_frame, text="Save Current Boxes", command=self.save_boxes).pack(pady=20)

        ttk.Button(self.controls_frame, text="Clear boxes for series", command=self.clear_boxes).pack(pady=20)
        self.n_labelled = ttk.Label(self.controls_frame, text=f'Labelled series: {len(self.ct_series_data["labels"])}')
        self.n_labelled.pack(pady=20)


    def update_window_level(self, _=None):
        self.window_level_label.config(text=f"Window Level: {int(self.window_level.get())}")
        self.update_image()

    def update_window_width(self, _=None):
        self.window_width_label.config(text=f"Window Width: {int(self.window_width.get())}")
        self.update_image()

    def save_boxes(self):
        with open(self.output_path, mode="w", encoding="UTF-8") as output_file:
            json.dump(self.ct_series_data, output_file, ensure_ascii=False, indent=4)
        self.n_labelled.config(text=f'Labelled series: {len(self.ct_series_data["labels"])}')
        print('Saved to disk')

    def clear_boxes(self):
        self.ct_series_data = {"img_size": 512, "labels": {}}
        self.update_image()

    def lose_focus(self, event=None):
        self.root.focus_set()

    def validate_input(self, value):
        if not value or not self.ct_series:
            # this accounts for the field being empty which is a valid state (like deleting all content)
            return True
        if value.isdigit():
            num = int(value)
            if 0 <= num < len(self.ct_series):
                return True
        return False

    def undo_interpolation(self):
        if not self.last_interpolation:
            return
        for current_idx, (start_x, start_y, end_x, end_y) in self.last_interpolation:
            folder_key = get_folder_key(self.series_path)
            bbox = [self.artifact_id, start_x, start_y, end_x, end_y]
            self.delete_box(folder_key, current_idx, bbox)
        self.update_image()
        self.last_interpolation = []

    def interpolate(self):
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
            self.last_interpolation.append((current_idx, bbox[1:]))

        self.inter_box_1 = None
        self.inter_box_2 = None
        self.start_slice_label.config(text=f"Start Slice:")
        self.start_box_label.config(text=f"Start Box:")
        self.end_slice_label.config(text=f"End Slice:")
        self.end_box_label.config(text=f"End Box:")
        self.update_image()


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

        except KeyError as e:
            print(f"Key error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    def on_box_release(self, event):
        current_idx = self.slider.get()
        # Finalize the bounding box
        end_x = self.image_canvas.canvasx(event.x)
        end_y = self.image_canvas.canvasy(event.y)

        print(f"self.current_box before = {self.current_box}")
        # Ensuring that start_x <= end_x and start_y <= end_y
        if self.start_x > end_x or self.start_y > end_y:
            self.image_canvas.delete(self.current_box) # remove bad rectangle
            final_start_x = min(self.start_x, end_x)
            final_end_x = max(self.start_x, end_x)
            final_start_y = min(self.start_y, end_y)
            final_end_y = max(self.start_y, end_y)
            self.current_box = self.image_canvas.create_rectangle(final_start_x, final_start_y, final_end_x, final_end_y, outline="red")
            print(f"self.current_box after = {self.current_box}")
        else:
            final_start_x = self.start_x
            final_end_x = end_x
            final_start_y = self.start_y
            final_end_y = end_y

        print(f"final_start_x: {final_start_x}; final_end_x: {final_end_x}; final_start_y: {final_start_y}; final_end_y: {final_end_y}")


        square = abs(final_end_x - final_start_x) * abs(final_end_y - final_start_y)
        if square <= self.bbox_square_threshold:
            print(f'square of {(final_start_x, final_start_y, final_end_x, final_end_y)} is less than threshold ({self.bbox_square_threshold})')
            self.image_canvas.delete(self.current_box)  # remove from the canvas
            return

        target_list = self.get_current_viewtype()

        if 0 <= end_x < target_list[0].shape[1] and 0 <= end_y < target_list[0].shape[0]:
            self.image_canvas.coords(self.current_box, final_start_x, final_start_y, final_end_x, final_end_y)
            self.image_canvas.tag_bind(self.current_box, '<Button-2>', lambda event, rect_id=self.current_box: self.on_rectangle_click(event, rect_id))
            self.image_canvas.tag_bind(self.current_box, '<Control-Button-2>', lambda event, rect_id=self.current_box: self.set_interpolation_box(event, rect_id, 1))
            self.image_canvas.tag_bind(self.current_box, '<Option-Button-2>', lambda event, rect_id=self.current_box: self.set_interpolation_box(event, rect_id, 2))
        
        folder_key = get_folder_key(self.series_path)
        bbox = [self.artifact_id, final_start_x, final_start_y, final_end_x, final_end_y]
        self.add_box(folder_key, current_idx, bbox)

    def set_interpolation_box(self, event, rect_id, box_n):
        current_idx = self.slider.get()

        start_x, start_y, end_x, end_y = self.image_canvas.coords(rect_id)

        self.image_canvas.itemconfig(rect_id, outline="green2")

        if box_n == 1:
            self.inter_box_1 = (current_idx, [start_x, start_y, end_x, end_y])
            self.start_slice_label.config(text=f"Start Slice: {current_idx}")
            self.start_box_label.config(text=f"Start Box: {start_x}, {start_y}, {end_x}, {end_y}")
        elif box_n == 2:
            self.inter_box_2 = (current_idx, [start_x, start_y, end_x, end_y])
            self.end_slice_label.config(text=f"End Slice: {current_idx}")
            self.end_box_label.config(text=f"End Box: {start_x}, {start_y}, {end_x}, {end_y}")
        else:
            raise Exception(f"unknown box_n parameter (must be 1 or 2, got {box_n})")


    def get_current_img(self):
        slice_idx = self.slider.get()
        target_list = self.get_current_viewtype()
        slice_idx = min(len(target_list)-1, slice_idx)
        ct_slice = target_list[slice_idx]

        minval = self.window_level.get() - (self.window_width.get() / 2)
        maxval = self.window_level.get() + (self.window_width.get() / 2)
        ct_slice = ((ct_slice - minval) / (maxval - minval) * 255).clip(0, 255).astype('uint8')

        return Image.fromarray(ct_slice)

    def save_current_view(self):
        if not self.ct_series:
            return
        img = self.get_current_img()
        slice_idx = self.slider.get()
        #img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "-".join(f"{get_folder_key(self.series_path)}_{slice_idx}.png".split(os.sep)))
        mip = "_mip" if self.to_show.get() == "mip" else ""
        window_info = f"_L={int(self.window_level.get())}_W={int(self.window_width.get())}" if self.add_window_info.get() else ""
        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "-".join(f"{get_folder_key(self.series_path, short=True)}_{slice_idx}{mip}{window_info}.png".split(os.sep)))
        img.convert('RGB').save(img_path)
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

    def get_current_viewtype(self):
        if self.to_show.get() == "original":
            return self.ct_series
        if self.to_show.get() == "mip":
            return self.mip_series
        raise NotImplemented("unknown view: "+self.to_show.get())

    def show_pixel_value(self, event):
        # Extract x and y coordinates of the mouse cursor on the image
        x, y = event.x, event.y

        target_list = self.get_current_viewtype()

        if target_list:
            # Check if the coordinates are within bounds
            if 0 <= x < target_list[0].shape[1] and 0 <= y < target_list[0].shape[0]:
                slice_idx = self.slider.get()
                pixel_value = target_list[slice_idx][y, x]
                self.pixel_value_label.config(text=f"Pixel Value: {pixel_value}")
            else:
                self.pixel_value_label.config(text="Pixel Value: N/A")

    def toggle_view(self, _=None):
        target_list = self.get_current_viewtype()

        self.slider.config(to=len(target_list)-1)
        self.update_image()

    def update_image(self, _=None):
        if not self.ct_series:
            return

        # Get the current CT image
        img = self.get_current_img()
        width, height = img.size

        if width != self.img_size or height != self.img_size:
            img = img.resize((self.img_size, self.img_size))

        # Convert the image to a format suitable for Tkinter
        self.tk_img = ImageTk.PhotoImage(image=img)

        # If an image is already displayed on the canvas, remove it
        if hasattr(self, 'canvas_img_id'):
            self.image_canvas.delete(self.canvas_img_id)

        # Display the new image on the canvas
        self.canvas_img_id = self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

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


        #studies = set() # StudyInstanceUID

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
                        #image = dc.pixel_data_handlers.apply_modality_lut(dc_dataset.pixel_array, dc_dataset).astype('float32')
                        if dcm[0x00080060].value != 'CT': # checking modality
                            continue
                        #if hasattr(dcm, "SliceLocation"):
                        if not hasattr(dcm, "ImagePositionPatient"):
                            continue
                        #self.ct_series.append(dcm)
                        if dcm.SeriesInstanceUID in series:
                            series[dcm.SeriesInstanceUID].append((file, dcm))
                        else:
                            series[dcm.SeriesInstanceUID] = [(file, dcm)]
                        #studies.add(dcm.StudyInstanceUID)
                        #series.add(dcm.SeriesInstanceUID)
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
            series[chosen_series] = [pydicom.pixel_data_handlers.apply_modality_lut(dcm.pixel_array, dcm).astype('float32') for filename, dcm in series[chosen_series]]
            for image in series[chosen_series]:
                # Making all paddings have the same value
                image[image.astype(int) == -3024] = -2048.
            self.ct_series = series[chosen_series]
        
        self.mip_series = [createMIP(np.stack(self.ct_series, axis=0))]

        target_list = self.get_current_viewtype()

        self.slider.config(to=len(target_list)-1)

        #self.artifact_ranges = self.construct_artifact_ranges()

        #if self.ranges_listbox.size() > 0:
        #    self.ranges_listbox.delete(0, tk.END)

        #for text_repr in self.artifact_ranges.keys():
        #    self.ranges_listbox.insert(tk.END, text_repr)

        possible_artifacts = '\n'.join(self.series_labels[get_folder_key(self.series_path)]) if get_folder_key(self.series_path) in self.series_labels else "not available"
        self.folder_key_label.config(text=get_folder_key(self.series_path)+'\n'+"Possible artifacts: \n"+possible_artifacts, font=("Courier", 12))
        self.update_image()
        self.root.focus_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAVAI - CT Annotation, Viewing and Analysing Instrument')
    #parser.add_argument(
    #    'artifacts',
    #    help='path to the json file with artifact types')
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
    
    args = parser.parse_args()


    root = tk.Tk()
    viewer = CTViewer(root, series_path=args.series, labels_path=args.labels, checkpoint_path=args.checkpoint, output_path=args.destination)
    root.mainloop()
