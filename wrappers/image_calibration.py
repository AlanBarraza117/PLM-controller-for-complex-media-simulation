"""
PLM Twyman-Green Interferometer - Image Plane Calibration
==========================================================
Displays the left half of the PLM at mirror level 0 (state 0000)
and the right half at mirror level 15 (state 1111), creating a
phase step that shifts the interference fringes laterally.
"""

import numpy as np
import time
import os
from PLMController import PLMController

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PLMCTRL_DIR  = os.path.dirname(SCRIPT_DIR)
DLL_PATH     = os.path.join(SCRIPT_DIR, r'..\bin\plmctrl.dll')

# ── PLM dimensions ───────────────────────────────────────────────────────────
N = 1358   # mirror columns (width)
M = 800    # mirror rows    (height)

# ── Connection mode ──────────────────────────────────────────────────────────
# Set DISPLAYPORT = True  for 1.44 kHz (60 Hz × 24),
#     DISPLAYPORT = False for 720 Hz   (30 Hz × 24, HDMI)
DISPLAYPORT = False
CONNECTION_TYPE = 2 if DISPLAYPORT else 1   # 2 = DisplayPort, 1 = HDMI
CONTINUOUS      = 1                         # loop forever
NUM_HOLO        = 24
MAX_FRAMES      = 1                         # one frame is enough for a static pattern

# ── Lookup table: 16 normalised phase values (0 → 1) ─────────────────────────
# These are the measured phase shifts for each mirror level at your wavelength.
# Replace with your own measured values after calibration.
# The array must have exactly 16 entries, all in [0, 1].
phase_levels = np.array([
    0.000, 0.017, 0.036, 0.058, 0.085, 0.117, 0.157,
    0.217, 0.296, 0.400, 0.500, 0.605, 0.713, 0.820,
    0.922, 0.981,
], dtype=np.float32)

# ── Phase map: 16-level → 2×2 electrode encoding ─────────────────────────────
# Rows correspond to the 16 mirror levels (in your reordered sequence).
# Each row gives the [e1, e2, e3, e4] on/off state of the four electrodes.
phase_map_base = np.array([
    [0,0,0,0], [1,0,0,0], [0,1,0,0], [1,1,0,0],
    [0,0,1,0], [1,0,1,0], [0,1,1,0], [1,1,1,0],
    [0,0,0,1], [1,0,0,1], [0,1,0,1], [1,1,0,1],
    [0,0,1,1], [1,0,1,1], [0,1,1,1], [1,1,1,1],
], dtype=np.int32)

phase_map_order = (12, 8, 4, 14, 0, 6, 10, 2, 13, 5, 9, 1, 15, 7, 11, 3)
phase_map = phase_map_base[phase_map_order, :]   # shape (16, 4), int32

# ── Build the split phase pattern ────────────────────────────────────────────
# Shape required by BitpackHologramsGPU: (N, M, num_holograms)
#   N = 1358 (columns / PLM width)
#   M = 800  (rows    / PLM height)
#
# Left half  (columns   0 .. N//2-1) → level  0 → normalised value 0.0
# Right half (columns N//2 .. N-1  ) → level 15 → normalised value 1.0
#
# We use the *normalised* lookup-table values so the C++ bitpacker can
# quantise correctly; 0.0 snaps to level 0, 1.0 snaps to level 15.

phase = np.zeros((NUM_HOLO, M, N), dtype=np.float32)
phase[:, :, N//2:] = phase_levels[15]  # right half → level 15 boundary

# ── Initialise PLM ───────────────────────────────────────────────────────────
os.chdir(PLMCTRL_DIR)

plm = PLMController(MAX_FRAMES, N, M, DLL_PATH, x0=2560, y0=0)

# 1. Open USB comms first
plm.open()
time.sleep(1)

# 2. Upload lookup table and phase map BEFORE starting the display
plm.set_lookup_table(phase_levels)
plm.set_phase_map(phase_map)

# 3. Configure the PLM hardware (only needed once per boot)
try:
    plm.set_source(0, 1)                   # Parallel RGB, 24-bit
    plm.set_port_swap(0, 0)                # ABC → ABC
    plm.set_port_swap(1, 0)
    plm.set_pixel_mode(CONNECTION_TYPE)
    time.sleep(1)
    plm.set_connection_type(CONNECTION_TYPE)
    time.sleep(5)
    plm.set_video_pattern_mode()
    time.sleep(3)
    plm.update_lut(CONTINUOUS, CONNECTION_TYPE)
    time.sleep(2)
except Exception as e:
    print(f"[Info] Hardware config skipped (already configured?): {e}")

# 4. Start the display window (must come AFTER configure for DisplayPort)
plm.set_windowed(True)
plm.start_ui()
time.sleep(1)

# 5. Start playback
plm.play()
time.sleep(1)

# ── Pack and upload the hologram frame ───────────────────────────────────────
frame = plm.bitpack_holograms_gpu(phase)   # returns one bitpacked RGBA frame

# Insert at slot 0
plm.insert_frames(frame, offset=0, format=1)   # format=1 → RGBA

# Tell the PLM to loop over just this one frame
plm.set_frame_sequence(np.array([0], dtype=np.uint64))
plm.start_sequence(MAX_FRAMES)

# ── Interactive section ───────────────────────────────────────────────────────
print("=" * 60)
print("Split-chip pattern active:")
print("  Left  half → mirror level  0  (phase ~ 0.0)")
print("  Right half → mirror level 15  (phase ~ 1.0)")
print()
print("You should see the interference fringes shift laterally")
print("on the right half of the PLM chip relative to the left.")
print()
print("Press Enter to exit...")
input()

# ── Cleanup ───────────────────────────────────────────────────────────────────
plm.stop()
plm.stop_ui()
plm.lib.Close()
print("Done.")