import matplotlib.pyplot as plt

# Daten
modelle = ['YOLOv8n', 'YOLOv8n-face', 'YOLOv11n-face']
ap_scores = [50.3, 64.8, 68.2]  # Average Precision
fps_scores = [80, 72, 65]       # Frames per Second

# Balkendiagramm
x = range(len(modelle))
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Modelle')
ax1.set_ylabel('AP@0.5 (%)', color=color)
ax1.bar(x, ap_scores, color=color, width=0.4, align='center', label="AP@0.5")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(x)
ax1.set_xticklabels(modelle)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('FPS', color=color)
ax2.plot(x, fps_scores, color=color, marker='o', label="FPS")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Vergleich der YOLO-Modelle')
plt.show()
