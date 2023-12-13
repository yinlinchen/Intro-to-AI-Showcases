import cv2
import os
import numpy as np


def computeHistogram(image_path):
    """
    Computes and normalizes the grayscale histogram of an image.
    
    image_path: Path to a image.

    Returns array of the normalized grayscale histogram.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_normalized = cv2.normalize(hist, hist).flatten()
    return hist_normalized
    

def loadFolder(folder, method):
    """
    Loads images from folder and computes their data based on the specified methods.

    folder: Directory path containing images.
    method: gray, color or shape.

    Returns image paths as keys and computed data as values:
    """
    images_data = {}
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            data = None
            if method == 'gray':
                data = computeHistogram(path)
            elif method == 'color':
                data = computeColorHistogram(path)
            elif method == 'shape':
                data = calculateHu(path)

            if data is not None:
                images_data[path] = data
    return images_data

def findImage(input_image_path, images_data, top_n):
    """
    Finds and sorts images based on similarity score using grayscale histogram computation.

    input_image_path: Path to the input image.
    images_data: image data.
    top_n: Number results to return.

    Returns image path, similarity score.
    """
    input_hist = computeHistogram(input_image_path)
    similar_images = []

    for path, hist in images_data.items():
        comparison = cv2.compareHist(input_hist, hist, cv2.HISTCMP_CORREL)
        similar_images.append((path, comparison))

    similar_images.sort(key=lambda x: x[1], reverse=True)
    return similar_images[:top_n]

def featureMatching(input_image_path, images_folder, top_n):
    """
    Feature matching using ORB algorithm to find similar images.

    input_image_path: Path to the input image.
    images_folder: path to images.
    top_n: Number of top similar images to return based on feature matching.

    Returns image path, similarity score.
    """
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(input_image, None)

    matchingResults = []

    for filename in os.listdir(images_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(images_folder, filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            kp2, des2 = orb.detectAndCompute(image, None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            good = []
            
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good.append([m])

            matchingResults.append((path, len(good)))

    matchingResults.sort(key=lambda x: x[1], reverse=True)

    return matchingResults[:top_n]

def computeColorHistogram(image_path):
    """
    Computes and normalizes the color histogram of an image.

    image_path: Path to the image.

    Returns list of normalized colored histograms.
    """
    image = cv2.imread(image_path)
    color_hist = []

    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])

        channel_hist_normalized = cv2.normalize(channel_hist, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).flatten()
        color_hist.extend(channel_hist_normalized)

    return color_hist

def findColorImage(input_image_path, images_data, top_n):
    """
    Finds and sorts images based on color histogram similarity score.

    input_image_path: Path to the input image.
    images_data: image.
    top_n: Numbers of returned images.
    """
    input_hist = computeColorHistogram(input_image_path)
    input_hist = np.array(input_hist, dtype=np.float32)
    similar_images = []

    for path, hist in images_data.items():
        try:
            hist = np.array(hist, dtype=np.float32)
            comparison = cv2.compareHist(input_hist, hist, cv2.HISTCMP_CORREL)
            similar_images.append((path, comparison))
        except Exception as e:
            print(f"Error comparing {input_image_path} and {path}: {e}")

    similar_images.sort(key=lambda x: x[1], reverse=True)
    return similar_images[:top_n]


#Shape Matching
def calculateHu(image_path):
    """
    Calculates the Hu Moments of the image.

    image_path: Path to the image.

    Returns array of Hu Moments:
    """
    image = cv2.imread(image_path, 0)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)

    epsilon = 1e-7
    for i in range(0, 7):
        hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + epsilon)
    return hu_moments.flatten()

def findShape(input_image_path, images_data, top_n):
    """
    Finds images based on shape similarity score using Hu Moments.

    input_image_path: Path to the input image.
    images_data: image data.
    top_n: Numbers of returned images.
    """

    input_hu_moments = calculateHu(input_image_path)
    similar_images = []

    for path, hu_moments in images_data.items():

        distance = cv2.norm(input_hu_moments, hu_moments, cv2.NORM_L2)
        similar_images.append((path, distance))

    similar_images.sort(key=lambda x: x[1])
    return similar_images[:top_n]
