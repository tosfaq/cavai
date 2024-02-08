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
from skimage.transform import radon



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

class ArtifactRange:
    def __init__(self, start_slice: int, end_slice: int, selected_artifacts: list):
        self.start_slice = start_slice
        self.end_slice = end_slice
        self.selected_artifacts = selected_artifacts

    def __str__(self):
        return f"{self.start_slice}-{self.end_slice} : {', '.join(self.selected_artifacts)}"

    def toDict(self):
        return {'start': self.start_slice, 'end': self.end_slice, 'artifacts': self.selected_artifacts}

    def __repr__(self):
        return self.__str__()


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


class CTViewer:
    def __init__(self, root, series_path, labels_path, artifacts_path, checkpoint_path, output_path):
        self.root = root
        self.root.title("CAVAI")

        self.bring_window_to_forefront()

        self.is_numpy = False
        self.to_show = tk.StringVar(value="original")
        self.ct_series = []
        self.mip_series = []

        self.series_path = series_path
        self.output_path = output_path
        
        self.series_labels = self.load_json_file(labels_path)
        self.artifact_options = self.load_json_file(artifacts_path, required=True)
        self.final_labelling = self.load_json_file(checkpoint_path, default={})
        print('Количество серий уникальных ', len(self.final_labelling))

        self.artifact_ranges = self.construct_artifact_ranges()

        self.viewer_ui()
        self.load_ct_series(series_path)

    def construct_artifact_ranges(self):
        if self.series_path and get_folder_key(self.series_path) in self.final_labelling:
            ranges_dict = dict()
            for range_dict in self.final_labelling[get_folder_key(self.series_path)]:
                range_obj = ArtifactRange(start_slice=range_dict["start"], end_slice=range_dict["end"], selected_artifacts=range_dict["artifacts"])
                ranges_dict[str(range_obj)] = range_obj
            return ranges_dict
        else:
            return dict()

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
        self.folder_key_label.grid(row=0, column=0, pady=20, columnspan=2)

        self.load_button = ttk.Button(self.root, text="Load CT Series", command=self.load_ct_series)
        self.load_button.grid(row=0, column=2, pady=20)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.grid(row=1, column=1, padx=10, sticky="ns")

        # Bar on the left
        self.left_frame = tk.Frame(self.root)
        self.left_frame.grid(row=1, column=0, padx=10, sticky="nsew")

        # Configure rows and columns to behave properly when resizing
        self.left_frame.grid_rowconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)  # This is for the Listbox
        self.left_frame.grid_columnconfigure(1, weight=0)  # This is for the Scrollbar
        self.left_frame.grid_rowconfigure(0, weight=0)
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_rowconfigure(2, weight=0)
        
        self.ranges_label = ttk.Label(self.left_frame, text="Artifact Ranges")
        self.ranges_label.grid(row=0, column=0, columnspan=2, sticky="ew")
        #
        self.ranges_listbox = Listbox(self.left_frame)
        #self.ranges_listbox.pack(pady=10, fill="x")
        self.ranges_listbox.grid(row=1, column=0, sticky='nsew', pady=10)

        # Setting up the Scrollbar alongside the Listbox
        scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.ranges_listbox.yview)
        scrollbar.grid(row=1, column=1, sticky='ns', pady=10)
        self.ranges_listbox.config(yscrollcommand=scrollbar.set)
        
        #scrollbar = ttk.Scrollbar(self.ranges_listbox, orient="vertical", command=self.ranges_listbox.yview)
        #scrollbar.pack(side="right", fill="x")
        #self.ranges_listbox.config(yscrollcommand=scrollbar.set)
        
        self.add_range_button = ttk.Button(self.left_frame, text="+", command=self.add_artifact_range)
        self.add_range_button.grid(row=2, column=0, pady=20, sticky='w', columnspan=2)
        
        self.remove_range_button = ttk.Button(self.left_frame, text="-", command=self.remove_artifact_range)
        self.remove_range_button.grid(row=2, column=0, pady=20, sticky='e', columnspan=2)
        #self.add_range_button = ttk.Button(self.left_frame, text="+", command=self.add_range_to_listbox)
        #self.add_range_button.pack(side="left", pady=20)
        
        #self.remove_range_button = ttk.Button(self.left_frame, text="-", command=self.remove_range_from_listbox)
        #self.remove_range_button.pack(side="left", pady=20)
        #

        self.slider = tk.Scale(self.canvas_frame, from_=0, orient="horizontal", command=self.update_image)    
        self.slider.pack(fill="x")

        self.img_label = tk.Label(self.canvas_frame)
        self.img_label.pack(pady=20)

        # Bind scrolling event to series scrolling
        self.img_label.bind("<MouseWheel>", self.on_scroll)
        # Bind motion event to the image label
        self.img_label.bind("<Motion>", self.show_pixel_value)

        # Bind [ and ] to setting range boundaries
        self.root.bind("<KeyPress-bracketleft>", self.set_start_slice)
        self.root.bind("<KeyPress-bracketright>", self.set_end_slice)
        # Bind Return to adding artifact range
        self.root.bind("<Return>", self.add_artifact_range)
        self.root.bind("<BackSpace>", self.bksp_remove_artifact_range)

        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)

        #self.root.bind('<Escape>', self.lose_focus())

        # Controls on the side
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.grid(row=1, column=2, padx=10, sticky="n")

        ttk.Radiobutton(self.controls_frame, state=tk.NORMAL, text="Show MIP", value="mip", variable=self.to_show, command=self.toggle_view).pack(pady=5)
        ttk.Radiobutton(self.controls_frame, state=tk.ACTIVE, text="Show original", value="original", variable=self.to_show, command=self.toggle_view).pack(pady=5)

        self.add_window_info = tk.IntVar()
        ttk.Checkbutton(self.controls_frame, text=f"Include window info", variable=self.add_window_info).pack(pady=10)

        save_button = ttk.Button(self.controls_frame, text="Save current view", command=self.save_current_view)
        save_button.pack(pady=20)

        # Label to display pixel value
        self.pixel_value_label = ttk.Label(self.controls_frame, text="Pixel Value: N/A")
        self.pixel_value_label.pack(pady=20)

        """
        self.hu_range = tk.Frame(self.controls_frame)
        self.hu_range.pack(pady=10)
        self.left_hu_border = ttk.Entry(self.hu_range)
        self.left_hu_border.grid(row=0, column=0, padx=5, sticky="ns")
        self.right_hu_border = ttk.Entry(self.hu_range)
        self.right_hu_border.grid(row=0, column=1, padx=5, sticky="ns")
        """

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

        vcmd = self.root.register(self.validate_input)
        self.start_slice = ttk.Entry(self.controls_frame, validate="key", validatecommand=(vcmd, '%P'))
       

        # Artifact labeling
        ttk.Label(self.controls_frame, text="Start Slice:").pack(pady=5)
        #self.start_slice = ttk.Entry(self.controls_frame)
        self.start_slice = ttk.Entry(self.controls_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.start_slice.pack(fill="x")
        self.start_slice.bind('<Escape>', self.lose_focus)

        ttk.Label(self.controls_frame, text="End Slice:").pack(pady=5)
        #self.end_slice = ttk.Entry(self.controls_frame)
        self.end_slice = ttk.Entry(self.controls_frame, validate="key", validatecommand=(vcmd, '%P'))
        self.end_slice.pack(fill="x")
        self.end_slice.bind('<Escape>', self.lose_focus)
        
        #self.artifact_options = ["motion", "metal", "beam hardening"]
        self.artifact_vars = {}
        for index, artifact in enumerate(self.artifact_options.values(), start=1):
            if index > 9:  # We only support 1-9 key bindings
                break
            var = tk.IntVar()
            cb = ttk.Checkbutton(self.controls_frame, text=f"{index}. {artifact}", variable=var)
            cb.pack(pady=5)
            self.root.bind(str(index), self.generate_checkbox_callback(var, cb))
            self.artifact_vars[artifact] = var

        ttk.Button(self.controls_frame, text="Save Current Ranges", command=self.save_ranges).pack(pady=20)

        """
        self.instructions = ttk.Frame(self.root)
        self.instructions.grid(row=2, column=0, pady=10)

        self.instructions_text = tk.Label(root, text=f"Possible artifacts: ...", font=("Courier", 12))
        self.root.grid_rowconfigure(2, weight=1)
        """

    def update_window_level(self, _=None):
        self.window_level_label.config(text=f"Window Level: {int(self.window_level.get())}")
        self.update_image()

    def update_window_width(self, _=None):
        self.window_width_label.config(text=f"Window Width: {int(self.window_width.get())}")
        self.update_image()

    def save_ranges(self):
        self.final_labelling[get_folder_key(self.series_path)] = \
            list(map(ArtifactRange.toDict, self.artifact_ranges.values()))
        with open(self.output_path, mode="w", encoding="UTF-8") as output_file:
            json.dump(self.final_labelling, output_file, ensure_ascii=False, indent=4)
        print('Saved to disk')

    def lose_focus(self, event=None):
        self.root.focus_set()

    def generate_checkbox_callback(self, var, cb):
        def callback(event):
            focused_widget = self.root.focus_get()
            if focused_widget in [self.start_slice, self.end_slice]:
                return
            if var.get():
                var.set(0)
            else:
                var.set(1)
            cb.update()  # Reflect the state change visually
        return callback

    def validate_input(self, value):
        if not value or not self.ct_series:
            # this accounts for the field being empty which is a valid state (like deleting all content)
            return True
        if value.isdigit():
            num = int(value)
            if 0 <= num < len(self.ct_series):
                return True
        return False

    def add_artifact_range(self, event=None):
        start = int(self.start_slice.get())
        end = int(self.end_slice.get())
    
        selected_artifacts = [artifact for artifact, var in self.artifact_vars.items() if var.get()]
        selected_artifacts.sort()
        if not selected_artifacts:
            print("No artifact selected.")
            return

        artifact_range = ArtifactRange(start, end, selected_artifacts)
        if str(artifact_range) in self.artifact_ranges:
            print(f'[{get_folder_key(self.series_path, short=True)}] Range {artifact_range} has been already added.')
            return
        print(f'[{get_folder_key(self.series_path, short=True)}] Range {artifact_range} added')
        self.artifact_ranges[str(artifact_range)] = artifact_range
        #self.artifact_ranges.append(artifact_range)
        
        # Add to the Listbox using the string representation of the ArtifactRange object
        self.ranges_listbox.insert(tk.END, str(artifact_range))
    
    def remove_artifact_range(self):
        try:
            selected_idx = self.ranges_listbox.curselection()[0]
            selected_str = self.ranges_listbox.get(selected_idx)
            # Remove the selected ArtifactRange object from the list
            print(f'[{get_folder_key(self.series_path, short=True)}] Range {selected_str} removed')
            #del self.artifact_ranges[selected_idx]
            self.artifact_ranges.pop(selected_str)
            
            # Remove from the Listbox
            self.ranges_listbox.delete(selected_idx)
            
            # After deletion, set the next item as active.
            # If the last item was deleted, set the previous item as active.
            if self.ranges_listbox.size() > 0:
                new_index = min(selected_idx, self.ranges_listbox.size() - 1)
                self.ranges_listbox.activate(new_index)
                self.ranges_listbox.selection_set(new_index)
        except Exception as e:
            print(e)

    def set_start_slice(self, event):
        slice_idx = self.slider.get()
        end_slice = self.end_slice.get()
        if end_slice and slice_idx > int(end_slice):
            self.end_slice.delete(0, tk.END)
            self.end_slice.insert(0, slice_idx)
        self.start_slice.delete(0, tk.END)
        self.start_slice.insert(0, slice_idx)

    def set_end_slice(self, event):
        slice_idx = self.slider.get()
        start_slice = self.start_slice.get()
        if start_slice and slice_idx < int(start_slice):
            self.start_slice.delete(0, tk.END)
            self.start_slice.insert(0, slice_idx)
        self.end_slice.delete(0, tk.END)
        self.end_slice.insert(0, slice_idx)

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
    
        new_value = self.slider.get() + increment*2
        if 0 <= new_value < len(self.ct_series):  # Ensure the new value is within bounds
            self.slider.set(new_value)
        self.update_image()

    def bksp_remove_artifact_range(self, event):
        focused_widget = self.root.focus_get()
        if focused_widget in [self.start_slice, self.end_slice]:
            return
        self.remove_artifact_range()

    def prev_image(self, event):
        target_list = self.get_current_viewtype()
        focused_widget = self.root.focus_get()
        if focused_widget in [self.start_slice, self.end_slice]:
            return
        new_value = self.slider.get() - 1
        if 0 <= new_value < len(target_list):  # Ensure the new value is within bounds
            self.slider.set(new_value)
            self.update_image()

    def next_image(self, event):
        target_list = self.get_current_viewtype()
        focused_widget = self.root.focus_get()
        if focused_widget in [self.start_slice, self.end_slice]:
            return
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
        img = self.get_current_img()
        self.tk_img = ImageTk.PhotoImage(image=img)
        self.img_label.config(image=self.tk_img)

    def load_ct_series(self, path=None):
        if not path:
            path = filedialog.askdirectory(title="Select Folder Containing CT Series", initialdir=os.path.realpath(__file__))
            self.root.focus_set()
            self.root.lift()
            if not path:
                return


        ct_series_old = self.ct_series
        series_path_old = self.series_path
        artifact_ranges_old = self.artifact_ranges
        self.ct_series = []
        self.series_path = None
        self.is_numpy = False
        self.root.lift()

        series = {} # SeriesInstanceUID


        #studies = set() # StudyInstanceUID

        for root_dir, _, files in os.walk(path):
            if 'npy' in root_dir:
                images = []  # a list of tuples (idx, img_array)
                self.is_numpy = True
                for file in files:
                    image = np.load(os.path.join(root_dir, file))
                    slice_idx = int(file.split('.')[0])
                    images.append((slice_idx, image))
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
                            series[dcm.SeriesInstanceUID].append(dcm)
                        else:
                            series[dcm.SeriesInstanceUID] = [dcm]
                        #studies.add(dcm.StudyInstanceUID)
                        #series.add(dcm.SeriesInstanceUID)
                        if not self.series_path:
                            self.series_path = root_dir
                    except:
                        pass

        if self.is_numpy:
            # idk why but .npy series start from neck (0 slice) and end up in stomach (last slice)
            # that is why these sequences are sorted in reverse order
            images.sort(key=lambda x: x[0], reverse=True)
            self.ct_series = [img for idx, img in images]
        else:
            if not series:
                showerror("Error", f"No series in {path} has been found")
                self.ct_series = ct_series_old
                self.series_path = series_path_old
                self.artifact_ranges = artifact_ranges_old
                return
    
            chosen_series = ask_option(self.root, list(series.keys()), "Choose the series") if len(series) > 1 else list(series.keys())[0]
            if chosen_series is None:
                self.ct_series = ct_series_old
                self.series_path = series_path_old
                self.artifact_ranges = artifact_ranges_old
                return
            series[chosen_series].sort(key=lambda x: float(x.ImagePositionPatient[2]))
            series[chosen_series] = [pydicom.pixel_data_handlers.apply_modality_lut(dcm.pixel_array, dcm).astype('float32') for dcm in series[chosen_series]]
            for image in series[chosen_series]:
                # Making all paddings have the same value
                image[image.astype(int) == -3024] = -2048.
            self.ct_series = series[chosen_series]
        
        self.mip_series = [createMIP(np.stack(self.ct_series, axis=0))]

        target_list = self.get_current_viewtype()

        self.slider.config(to=len(target_list)-1)

        self.artifact_ranges = self.construct_artifact_ranges()

        if self.ranges_listbox.size() > 0:
            self.ranges_listbox.delete(0, tk.END)

        for text_repr in self.artifact_ranges.keys():
            self.ranges_listbox.insert(tk.END, text_repr)

        possible_artifacts = '\n'.join(self.series_labels[get_folder_key(self.series_path)]) if get_folder_key(self.series_path) in self.series_labels else "not available"
        self.folder_key_label.config(text=get_folder_key(self.series_path)+'\n'+"Possible artifacts: \n"+possible_artifacts, font=("Courier", 12))
        self.update_image()
        self.root.focus_set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAVAI - CT Annontation, Viewing and Analysing Instrument')
    parser.add_argument(
        'artifacts',
        help='path to the json file with artifact types')
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
    viewer = CTViewer(root, series_path=args.series, labels_path=args.labels, artifacts_path=args.artifacts, checkpoint_path=args.checkpoint, output_path=args.destination)
    root.mainloop()
