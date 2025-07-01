import matplotlib.pyplot as plt
import numpy as np

times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

# MediaPipe
fps_mp = [20.47, 21.96, 18.62, 20.1, 20.76, 21.09, 19.06, 22.24, 20.21, 20.53, 22.25]
stability_mp = [98, 97, 97, 98, 98, 99, 97, 97, 98, 98, 97]

# --- Liniendiagramm erstellen ---

fig, ax1 = plt.subplots(figsize=(10, 6))

# FPS für YOLOv8 und MediaPipe
ax1.plot(times, fps_mp, marker='s', linestyle='-', color='red', label='MediaPipe - FPS')

# Achsenbeschriftungen und Titel
ax1.set_xlabel('Zeit (Sekunden)', fontsize=14)
ax1.set_ylabel('FPS', fontsize=14)
ax1.set_title('FPS MediaPipe', fontsize=16)

ax1.set_ylim(0, 30)  # Bereich von 0 bis 25 für FPS-Werte
ax1.set_yticks(np.arange(0, 31, 2.5))  # Ticks von 0 bis 25 in Schritten von 5

# Zweite Y-Achse für Stabilität
ax2 = ax1.twinx()
ax2.plot(times, stability_mp, marker='D', linestyle='--', color='firebrick', label='MediaPipe - Stabilität')
ax2.set_ylabel('Stabilität (%)', fontsize=14)
ax2.set_ylim(80, 100)

# Gitterlinien
ax1.grid(True, linestyle='--', alpha=0.7)

# Legende anzeigen
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower center', fontsize=12, ncol=2)

# Layout und speichern
plt.tight_layout()
plt.savefig('src/final_report/data/fps_mediapipe_detection.png')