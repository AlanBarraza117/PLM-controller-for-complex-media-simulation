import numpy as np
from PIL import Image
import ctypes
import matplotlib.pyplot as plt
import time
import os
from PLMController import PLMController

# Resolve paths relative to this script's location (works wherever you run from)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLMCTRL_DIR = os.path.dirname(SCRIPT_DIR)  # plmctrl root (has BitpackHologramsCS.hlsl)

# --- Config ---
IMG_PATH = os.path.join(SCRIPT_DIR, r'..\holograms\pickle_rick.png')
N = 1358   # PLM width
M = 800    # PLM height
NUM_HOLO = 24      # 24 holograms per RGB frame (bitpacked)
GS_ITER = 50       # Gerchberg-Saxton iterations

# --- 1. Load & preprocess target image ---
img = Image.open(IMG_PATH).convert('L')  # grayscale

# Resize to fit MxN (preserving aspect ratio, then pad)
img.thumbnail((N, M), Image.LANCZOS)
target = np.array(img, dtype=np.float32)
target = (target - target.min()) / (target.max() - target.min() + 1e-10)

# Pad to MxN (center the image)
ph, pw = target.shape
y_off = (M - ph) // 2
x_off = (N - pw) // 2
target_full = np.zeros((M, N), dtype=np.float32)
target_full[y_off:y_off+ph, x_off:x_off+pw] = target

# Target amplitude = sqrt(target intensity)
target_amp = np.sqrt(target_full)

# --- 2. Gerchberg-Saxton algorithm (Fourier CGH) ---
# Phase pattern starts random
phase = np.random.uniform(0, 2*np.pi, (M, N))

for it in range(GS_ITER):
    # SLM plane: amplitude=1, phase from current estimate
    slm_field = np.exp(1j * phase)
    
    # Propagate to far field (FFT)
    far_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
    far_amp = np.abs(far_field)
    far_phase = np.angle(far_field)
    
    # Apply target amplitude constraint, keep phase
    far_field_new = target_amp * np.exp(1j * far_phase)
    
    # Back-propagate to SLM plane
    slm_field_new = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(far_field_new)))
    
    # Extract phase, keep amplitude at 1
    phase = np.angle(slm_field_new)
    
    mse = np.mean((far_amp / (far_amp.max() + 1e-10) - target_amp)**2)
    if it % 10 == 0:
        print(f"  GS iteration {it}/{GS_ITER} — MSE: {mse:.6f}")

# Add off-axis carrier (blazed grating) to shift reconstruction away from DC
# This separates the image from the zero-order bright spot
CARRIER_FRINGES = 150  # number of fringe periods across the PLM (shift in pixels)
x = np.arange(N)[np.newaxis, :]
carrier = 2 * np.pi * CARRIER_FRINGES * x / N
phase = np.mod(phase + carrier, 2 * np.pi)

# Final phase in [0, 2π], normalize to [0, 1]
phase_norm = phase / (2 * np.pi)

# --- 3. Duplicate to fill 24 holograms ---
phase_3d = np.tile(phase_norm[np.newaxis, :, :], (NUM_HOLO, 1, 1)).astype(np.float32)

# --- 4. Visualize ---
plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.imshow(target_full, cmap='gray')
plt.title("Target image (padded)")
plt.subplot(1, 3, 2)
plt.imshow(phase_norm, cmap='gray')
plt.title("CGH phase pattern")
plt.subplot(1, 3, 3)
recon = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j * phase))))
recon_int = np.abs(recon)**2
# Show cropped region around the +1 order (shifted off-axis from DC)
shift = CARRIER_FRINGES
c = N // 2 + shift  # center of +1 order
crop = recon_int[:, c-200:c+200]
plt.imshow(crop, cmap='gray', extent=[-200, 200, M//2, -M//2])
plt.title(f"+1 order ({shift}px from DC)")
plt.xlabel("pixels"); plt.ylabel("pixels")
plt.tight_layout()
plt.savefig('cgh_preview.png', dpi=150)
plt.show()
print("Preview saved to cgh_preview.png")

# --- 5. Send to PLM ---
# DLL needs to find BitpackHologramsCS.hlsl in CWD — switch to plmctrl root
os.chdir(PLMCTRL_DIR)

MAX_FRAMES = 64
DLL_PATH = os.path.join(SCRIPT_DIR, r'..\bin\plmctrl.dll')

plm = PLMController(MAX_FRAMES, N, M, DLL_PATH, x0=2560, y0=0)

plm.set_windowed(True)
plm.start_ui()

plm.open()
plm.play()

# Wait for UI to be ready
time.sleep(2)

# Phase levels & phase map (from your setup)
phase_levels = np.array([0.004, 0.017, 0.036, 0.058, 0.085, 0.117, 0.157,
                         0.217, 0.296, 0.4, 0.5, 0.605, 0.713, 0.82,
                         0.922, 0.981, 1.0], dtype=np.float32)
plm.set_lookup_table(phase_levels)

phase_map = np.array([
    [0,0,0,0], [1,0,0,0], [0,1,0,0], [1,1,0,0],
    [0,0,1,0], [1,0,1,0], [0,1,1,0], [1,1,1,0],
    [0,0,0,1], [1,0,0,1], [0,1,0,1], [1,1,0,1],
    [0,0,1,1], [1,0,1,1], [0,1,1,1], [1,1,1,1],
])
phase_map_order = (12, 8, 4, 14, 0, 6, 10, 2, 13, 5, 9, 1, 15, 7, 11, 3)
phase_map = phase_map[phase_map_order, :]
plm.set_phase_map(phase_map)

# Configure PLM (only needed once per boot — skip if already configured)
# HDMI = 1
# CONTINUOUS = 1
# print("Configuring PLM (skip if already configured)...")
# plm.configure(play_mode=CONTINUOUS, connection_type=HDMI)

# Bitpack and insert
print("Bitpacking and inserting CGH frame...")
plm.bitpack_and_insert_gpu(phase_3d, 0)

# Set sequence and play
seq = np.arange(MAX_FRAMES, dtype=np.uint64)
plm.set_frame_sequence(seq)
plm.start_sequence(MAX_FRAMES)
print("Playing CGH on PLM!")

input("Press Enter to stop cleanup...")

plm.stop()
plm.stop_ui()
plm.lib.Close()
print("Done")