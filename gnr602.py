import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

# -----------------------------
# Core Functions (Your Code)
# -----------------------------

def rgb2gray(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

def gradient_x(img):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return ndimage.convolve(img, kernel_x)

def gradient_y(img):
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return ndimage.convolve(img, kernel_y)

def compute_harris(gray_img, k=0.04, window_size=5):
    blur_img = ndimage.gaussian_filter(gray_img, sigma=1.0)

    Ix = gradient_x(blur_img)
    Iy = gradient_y(blur_img)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    Sxx = ndimage.uniform_filter(Ixx, size=window_size)
    Syy = ndimage.uniform_filter(Iyy, size=window_size)
    Sxy = ndimage.uniform_filter(Ixy, size=window_size)

    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy

    R = det - k * (trace**2)
    return R

def multi_scale_harris(gray_img, scales=[1.0, 0.5, 0.25]):
    combined_R = np.zeros_like(gray_img, dtype=float)

    for scale in scales:
        scaled_img = ndimage.zoom(gray_img, zoom=scale, order=1) if scale != 1.0 else gray_img

        R_scale = compute_harris(scaled_img)

        # Local maxima
        local_max = ndimage.maximum_filter(R_scale, size=3)
        R_scale[R_scale != local_max] = 0

        # Normalize
        R_max = np.max(R_scale)
        if R_max > 0:
            R_scale = R_scale / R_max

        # Resize back
        if scale != 1.0:
            R_resized = ndimage.zoom(R_scale, zoom=1/scale, order=0)
            R_resized = R_resized[:gray_img.shape[0], :gray_img.shape[1]]
        else:
            R_resized = R_scale

        combined_R = np.maximum(combined_R, R_resized)

    return combined_R

def simulate_low_res(gray_img, scale=0.3):
    small = ndimage.zoom(gray_img, scale)
    upsampled = ndimage.zoom(
        small,
        (gray_img.shape[0] / small.shape[0],
         gray_img.shape[1] / small.shape[1])
    )
    return upsampled

def get_corners_absolute(R, abs_threshold=0.25):
    R_min = R.min()
    R_max = R.max()
    R_norm = (R - R_min) / (R_max - R_min + 1e-8)

    corner_map = np.zeros_like(R)
    corner_map[R_norm > abs_threshold] = 1
    return corner_map


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Harris Corner Detector", layout="wide")

st.title("🏙️ Building Corner Detection using Harris")
st.write("Upload an image and compare Normal vs Multi-Scale Harris on High & Low Resolution.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

threshold = 0.25

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = rgb2gray(input_img)
    gray_low = simulate_low_res(gray, scale=0.3)

    # Compute
    c_hr_norm = get_corners_absolute(compute_harris(gray), threshold)
    c_hr_multi = get_corners_absolute(multi_scale_harris(gray), threshold)

    c_lr_norm = get_corners_absolute(compute_harris(gray_low), threshold)
    c_lr_multi = get_corners_absolute(multi_scale_harris(gray_low), threshold)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def show(ax, img, corners, title):
        ax.imshow(img, cmap='gray')
        ax.imshow(np.ma.masked_where(corners == 0, corners),
                  cmap='autumn', alpha=0.8)
        ax.set_title(title)
        ax.axis('off')

    show(axes[0, 0], gray, c_hr_norm, "High-Res: Normal Harris")
    show(axes[0, 1], gray, c_hr_multi, "High-Res: Multi-Scale")

    show(axes[1, 0], gray_low, c_lr_norm, "Low-Res: Normal Harris")
    show(axes[1, 1], gray_low, c_lr_multi, "Low-Res: Multi-Scale")

    st.pyplot(fig)
