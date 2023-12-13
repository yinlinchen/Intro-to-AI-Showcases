import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

from imageSearcher import loadFolder, findImage, featureMatching, findColorImage, findShape


class ImageComparerApp:
    def __init__(self, root):
        # Initialize parameters for application window.
        self.root = root
        self.root.title('Image Similarity Searcher')

        self.select_button = tk.Button(self.root, text='Select Image', command=self.select_image)
        self.select_button.pack()

        self.selected_image_label = tk.Label(self.root)
        self.selected_image_label.pack()

        self.selected_image_name_label = tk.Label(self.root, text="")
        self.selected_image_name_label.pack()

        self.num_results_label = tk.Label(self.root, text="Number of Results:")
        self.num_results_label.pack()
        self.num_results_entry = tk.Entry(self.root)
        self.num_results_entry.pack()

        self.num_results_entry.insert(0, "5")

        self.select_folder_button = tk.Button(self.root, text='Select Folder', command=self.select_folder)
        self.select_folder_button.pack(pady=(20, 80))

        self.comparelabel = tk.Label(self.root, text="Choose Image Comparison Algorithms", font=("Helvetica", 16, "bold"))
        self.comparelabel.pack(pady=(0, 30))

        self.compare_hist_gray_button = tk.Button(self.root, text='Compare Histogram (Gray)', command=lambda: self.show_results(self.selected_image_path, 'gray'))
        self.compare_hist_gray_button.pack()

        self.compare_hist_color_button = tk.Button(self.root, text='Compare Histogram (Color)', command=lambda: self.show_results(self.selected_image_path, 'color'))
        self.compare_hist_color_button.pack()

        self.compare_feature_button = tk.Button(self.root, text='Compare Feature', command=lambda: self.show_results(self.selected_image_path, 'feature'))
        self.compare_feature_button.pack()

        self.compare_shape_button = tk.Button(self.root, text='Compare Shape', command=lambda: self.show_results(self.selected_image_path, 'shape'))
        self.compare_shape_button.pack()


        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack()

        self.images_folder = None
        self.selected_image_path = None

    def select_image(self):
        # Select an image and display it
        file_path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg"), 
                                                          ("JPEG Files", "*.jpeg"),
                                                          ("PNG Files", "*.png")])
        if file_path:
            self.selected_image_path = file_path
            self.display_selected_image(file_path)

    def select_folder(self):
        # select the folder containing images
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.images_folder = folder_path

    def display_selected_image(self, file_path):
        #display the selected image in window
        img = Image.open(file_path)
        img = img.resize((250, 250), Image.Resampling.LANCZOS)
        photo_img = ImageTk.PhotoImage(img)

        self.selected_image_label.config(image=photo_img)
        self.selected_image_label.photo_img = photo_img

        self.selected_image_name_label.config(text=os.path.basename(file_path))

    def show_results(self, input_image_path, method):
        #show comparison results
        try:
            top_n = int(self.num_results_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the number of results.")
            return

        images_data = loadFolder(self.images_folder, method)

        if method == 'gray':
            similar_images = findImage(input_image_path, images_data, top_n=top_n)
        elif method == 'color':
            similar_images = findColorImage(input_image_path, images_data, top_n=top_n)
        elif method == 'feature':
            similar_images = featureMatching(input_image_path, self.images_folder, top_n=top_n)
        elif method == 'shape':
            similar_images = findShape(input_image_path, images_data, top_n=top_n)
        else:
            similar_images = []

        self.display_results(similar_images)

    def display_results(self, similar_images):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for i, (path, score) in enumerate(similar_images):
            frame = tk.Frame(self.results_frame)
            frame.pack(side='left')

            tk.Label(frame, text=f'Rank {i+1}: {score:.2f}').pack()

            img = Image.open(path)
            img = img.resize((100, 100))
            img = ImageTk.PhotoImage(img)

            label = tk.Label(frame, image=img)
            label.image = img
            label.pack()

            tk.Label(frame, text=os.path.basename(path)).pack()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1000x800")
    app = ImageComparerApp(root)
    root.mainloop()
