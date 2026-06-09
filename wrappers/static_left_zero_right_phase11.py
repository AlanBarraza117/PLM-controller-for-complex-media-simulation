import numpy as np
import time
import os
from PLMController import PLMController

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLMCTRL_DIR = os.path.dirname(SCRIPT_DIR)

N = 1358
M = 800
DLL_PATH = os.path.join(SCRIPT_DIR, r'..\bin\plmctrl.dll')

phase_levels = np.array([0.004, 0.017, 0.036, 0.058, 0.085, 0.117, 0.157,
                         0.217, 0.296, 0.4, 0.5, 0.605, 0.713, 0.82,
                         0.922, 0.981, 1.0], dtype=np.float32)

phase_map = np.array([
    [0,0,0,0], [1,0,0,0], [0,1,0,0], [1,1,0,0],
    [0,0,1,0], [1,0,1,0], [0,1,1,0], [1,1,1,0],
    [0,0,0,1], [1,0,0,1], [0,1,0,1], [1,1,0,1],
    [0,0,1,1], [1,0,1,1], [0,1,1,1], [1,1,1,1],
])
phase_map_order = (12, 8, 4, 14, 0, 6, 10, 2, 13, 5, 9, 1, 15, 7, 11, 3)
phase_map = phase_map[phase_map_order, :]

os.chdir(PLMCTRL_DIR)
plm = PLMController(1, N, M, DLL_PATH, x0=2560, y0=0)
plm.set_windowed(True)
plm.set_lookup_table(phase_levels)
plm.set_phase_map(phase_map)
plm.start_ui()
time.sleep(1)

plm.open()
time.sleep(1)
plm.play()
time.sleep(1)

try:
    plm.set_source(0, 1)
    plm.set_port_swap(0, 0)
    plm.set_port_swap(1, 0)
    plm.set_pixel_mode(1)
    time.sleep(1)
    plm.set_connection_type(1)
    time.sleep(5)
    plm.set_video_pattern_mode()
    time.sleep(3)
    plm.update_lut(1, 1)
except Exception as e:
    print(f"Config skipped (already configured?): {e}")

# Create static image: left half = zeros, right half = phase_levels[11]
phase = np.zeros((M, N), dtype=np.float32)
phase[:, N//2:] = phase_levels[11]

# Replicate across all 24 holograms
phase = np.tile(phase[np.newaxis, :, :], (24, 1, 1))

# Bitpack and insert
frame = plm.bitpack_holograms_gpu(phase)
plm.insert_frames(frame, 0, format=1)
plm.set_frame(0)

print("---")
print("Static image: left=zeros, right=phase_levels[11] (24 holograms)")
print("Press Enter to exit...")
input()

plm.stop()
plm.stop_ui()
plm.lib.Close()
print("Done")
