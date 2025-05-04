import matplotlib.pyplot as plt

# Zeitpunkte und Messdaten aus deinem Summary
time = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
fps = [22.32, 21.29, 21.0, 23.43, 20.33, 20.65, 22.26, 20.94, 23.65, 23.34, 21.86]
processing_time = [22.82, 20.47, 21.36, 20.69, 27.85, 24.33, 28.03, 28.89, 25.94, 22.81, 23.59]

# Diagramm erstellen
plt.figure(figsize=(12, 6))

# FPS plotten
plt.plot(time, fps, label='FPS', marker='o')

# Processing Time plotten
plt.plot(time, processing_time, label='Verarbeitungszeit (ms)', marker='s')

# Achsenbeschriftung und Titel
plt.xlabel('Zeit (s)')
plt.ylabel('Wert')
plt.title('FPS und Verarbeitungszeit über 60 Sekunden')

# Legende und Gitter
plt.legend()
plt.grid(True)

# Anzeigen
plt.savefig('src/final_report/data/FPS_and_Processing_Time.png')
