from tkinter import simpledialog


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
