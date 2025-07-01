import matplotlib.pyplot as plt
import numpy as np

# Für Liniendiagramm (FPS über Zeit)
times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
fps_v8 = [6.23, 5.64, 6.15, 6.16, 4.98, 5.24, 5.06, 4.87, 5.04, 5.35, 6.19]

# --- Liniendiagramm erstellen ---

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(times, fps_v8, marker='o', linestyle='-', color='skyblue', label='YOLOv8-face')

ax.set_xlabel('Zeit (Sekunden)')
ax.set_ylabel('FPS')
ax.set_title('FPS-Verlauf über Zeit')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig('src/final_report/data/yolo_FPS_ueber_Zeit.png')