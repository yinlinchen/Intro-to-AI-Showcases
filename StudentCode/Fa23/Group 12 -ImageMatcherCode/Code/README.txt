Image Matcher

Description:

The Image Matcher is a Python application which allows user to search
for similar images using different computer vision algorithms.
We utilizing tkinter library for fornt-end of our program. It provides
a interface to compare images based on histogram comparison, feature matching,
and shape matching.


Requirements:

1. Python3
2. Libraries:
    (a) tkinter
    (b) PIL
    (c) opencv
    (d) numpy


Installation:

1. pip install numpy
2. pip install opencv-python
3. pip install pillow
4. pip install tk

Run:

1. Go to Code directory.
2. Run command: python3 searcherWindow.py


Instructions:

1. Select an image you want to search for.
2. Select number of results you wnat to display.
3. Select a folder of images to compare against to selected image in step 1.
4. Select image compare methods:
    (a) Histogram (Gray-scale)
    (b) Histogram (Color)
    (c) Feature Matching
    (d) Shape Comparison
5. Display results ranked by similarity.



