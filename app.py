import tkinter as tk

import pca_1d
import pca_2d


def __on_enter__(e):
    e.widget['background'] = 'blue'
def __on_leave__(e):
    e.widget['background'] = 'SystemButtonFace'

def __linear_pca_analysis__():
    data = pca_1d.detect()
    pca_1d.plot(data)

def __flat_pca_analysis__():
    data = pca_2d.detect()
    pca_2d.plot(data)

def __build_window__():
    root = tk.Tk()
    root.geometry("640x400")
    root.title("Face Recognition System")

    return root

def __build_entry__(parent, text, default_value):
    label = tk.Label(parent, text=text, font='Times 12')
    label.pack()
    default_entry_value = tk.StringVar(parent, value=default_value)
    entry = tk.Entry(parent, textvariable=default_entry_value)
    entry.pack()

def __build_button__(parent, text, command):
    plot_button1 = tk.Button(parent, text=text, width=40, height=2, pady=10, command=lambda: command())
    plot_button1.pack()
    plot_button1.bind("<Enter>", __on_enter__)
    plot_button1.bind("<Leave>", __on_leave__)

def run():
    root = __build_window__()

    __build_entry__(root, "Введите кол-во эталонов:", '4')
    __build_entry__(root, "Введите кол-во главных компонент:", '4')


    __build_button__(root, "Анализ одномерного PCA", __linear_pca_analysis__)
    __build_button__(root, "Анализ двумерного PCA", __flat_pca_analysis__)

    root.mainloop()