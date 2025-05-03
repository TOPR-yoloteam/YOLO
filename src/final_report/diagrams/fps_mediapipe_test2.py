import matplotlib.pyplot as plt
import numpy as np

# Für Liniendiagramm (FPS über Zeit)
times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
fps_conf_05 = [10.41, 18.6, 12.82, 18.07, 18.99, 17.06, 20.3, 14.74, 15.77, 19.88, 18.5]
fps_conf_08 = [21.2, 20.28, 18.47, 17.47, 18.67, 19.5, 20.97, 20.06, 20.51, 18.07, 19.05]

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
