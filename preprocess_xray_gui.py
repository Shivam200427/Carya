import cv2
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt


def pick_image():
    """
    Opens a file dialog to pick an image from the local device.
    Returns the selected image file path.
    """
    Tk().withdraw()  # Hide main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not file_path:
        raise ValueError("No image selected.")
    return file_path


def enhance_xray(img_path, target_size=(512, 512)):
    """
    Enhances a chest X-ray image for better visibility (no cropping).

    Steps:
    1. Adjust brightness and contrast (percentile windowing).
    2. Apply CLAHE for local contrast enhancement.
    3. Auto-balance brightness (gamma correction).
    4. Normalize, resize, and scale to [0,1].
    """
    # Step 1: Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable")

    # Step 2: Adjust levels using percentile windowing
    p_low, p_high = np.percentile(img, (1, 99))
    img = np.clip(img, p_low, p_high)
    img = ((img - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    # Step 3: Apply CLAHE (local contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Step 4: Auto-brightness balance using gamma correction
    mean_intensity = np.mean(img)
    gamma = 1.0
    if mean_intensity < 100:  # dark image
        gamma = 1.5
    elif mean_intensity > 150:  # bright image
        gamma = 0.7
    img = np.power(img / 255.0, gamma) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Step 5: Normalize and resize
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Step 6: Scale to [0,1] for model
    img = img.astype(np.float32) / 255.0

    return img


if __name__ == "__main__":
    # Step 1: Pick image
    path = pick_image()
    print("Selected image:", path)

    # Step 2: Enhance X-ray
    enhanced = enhance_xray(path, target_size=(512, 512))

    # Step 3: Display before and after
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original X-ray")

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Gamma Corrected X-ray")

    plt.show()
