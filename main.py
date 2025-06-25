import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_image(image_path, resize_dim=(200, 200)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resize_dim)
    return image

def extract_colors(image, num_colors=5):
    img_data = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_data)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def show_palette(colors):
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.subplot(1, len(colors), i + 1)
        plt.imshow([[color / 255]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img_path = "img/image.png"
    image = load_image(img_path)
    colors = extract_colors(image, num_colors=6)
    show_palette(colors)
