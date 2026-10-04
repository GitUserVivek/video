import numpy as np
import cv2
import os

# ============================================================
# CONFIGURATION
# ============================================================

NPZ_FILE = "prompt_embeds.npz"
OUTPUT_VIDEO = "/kaggle/working/output.mp4"

FPS = 30


# ============================================================
# 1. LOAD NPZ
# ============================================================

print("Loading:", NPZ_FILE)

data = np.load(NPZ_FILE, allow_pickle=False)

print("\nArrays found in NPZ:")
for name in data.files:
    arr = data[name]
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")


# ============================================================
# 2. FIND A LIKELY VIDEO/FRAME ARRAY
# ============================================================

video_array = None
array_name = None

for name in data.files:
    arr = data[name]

    # Common video formats:
    # (frames, height, width)
    # (frames, height, width, channels)
    if arr.ndim in [3, 4]:
        video_array = arr
        array_name = name
        break

if video_array is None:
    raise ValueError(
        "Could not find a 3D or 4D array that looks like video frames."
    )

print(f"\nUsing array: {array_name}")
print("Shape:", video_array.shape)
print("Dtype:", video_array.dtype)


# ============================================================
# 3. DETERMINE FRAME FORMAT
# ============================================================

frames = video_array

print("\nOriginal shape:", frames.shape)


# ------------------------------------------------------------
# Case A: (N, H, W)
# Grayscale video
# ------------------------------------------------------------

if frames.ndim == 3:

    num_frames, height, width = frames.shape

    channels = 1

    print("Detected: grayscale video")


# ------------------------------------------------------------
# Case B: (N, H, W, C)
# RGB / RGBA video
# ------------------------------------------------------------

elif frames.ndim == 4:

    num_frames, height, width, channels = frames.shape

    print(f"Detected: {channels}-channel video")


# ============================================================
# 4. CONVERT DATA TYPE TO uint8
# ============================================================

print("\nConverting frames to uint8...")

if frames.dtype != np.uint8:

    print(
        "Original value range:",
        frames.min(),
        "to",
        frames.max()
    )

    # If values are floating point in [0, 1]
    if frames.max() <= 1.0:

        frames = (frames * 255).clip(0, 255).astype(np.uint8)

    else:

        # Normalize arbitrary numerical range to 0-255
        min_val = frames.min()
        max_val = frames.max()

        if max_val == min_val:
            frames = np.zeros_like(frames, dtype=np.uint8)
        else:
            frames = (
                (frames - min_val)
                / (max_val - min_val)
                * 255
            ).clip(0, 255).astype(np.uint8)


# ============================================================
# 5. CREATE MP4 VIDEO
# ============================================================

print("\nCreating video...")

# MP4 codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (width, height),
    True
)

if not video_writer.isOpened():
    raise RuntimeError("Could not create video writer.")


# ============================================================
# 6. WRITE FRAMES
# ============================================================

for i in range(num_frames):

    frame = frames[i]

    # Grayscale -> BGR
    if frame.ndim == 2:

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_GRAY2BGR
        )

    # RGB -> BGR
    elif frame.shape[-1] == 3:

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

    # RGBA -> BGR
    elif frame.shape[-1] == 4:

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGBA2BGR
        )

    else:
        raise ValueError(
            f"Unsupported number of channels: {frame.shape[-1]}"
        )

    video_writer.write(frame)

    if (i + 1) % 100 == 0:
        print(f"Written {i + 1}/{num_frames} frames")


# ============================================================
# 7. FINISH
# ============================================================

video_writer.release()

print("\n========================================")
print("VIDEO CREATED SUCCESSFULLY")
print("========================================")
print("Output:", OUTPUT_VIDEO)
print("Frames:", num_frames)
print("Resolution:", f"{width}x{height}")
print("FPS:", FPS)
print("Duration:", round(num_frames / FPS, 2), "seconds")
print("File size:", round(os.path.getsize(OUTPUT_VIDEO) / (1024**2), 2), "MB")
