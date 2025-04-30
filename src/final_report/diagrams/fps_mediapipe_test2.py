import matplotlib.pyplot as plt
import numpy as np

# Für Liniendiagramm (FPS über Zeit)
times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
fps_conf_05 = [29.21, 27.36, 23.58, 30.53, 29.67, 25.72, 30.13, 28.34, 28.6, 30.1, 30.5]
fps_conf_08 = [20.57, 19.19, 21.21, 21.76, 20.89, 21.27, 21.01, 19.34, 20.12, 19.92, 21.37]

# --- Liniendiagramm erstellen ---

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(times, fps_conf_05, marker='o', linestyle='-', color='skyblue', label='Confidence 50%')
ax.plot(times, fps_conf_08, marker='s', linestyle='--', color='salmon', label='Confidence 80%')

ax.set_xlabel('Zeit (Sekunden)')
ax.set_ylabel('FPS')
ax.set_title('FPS-Verlauf über Zeit')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig('src/final_report/data/FPS_ueber_Zeit.png')
