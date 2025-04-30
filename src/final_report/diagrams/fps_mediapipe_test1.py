import matplotlib.pyplot as plt
import numpy as np

# --- Daten vorbereiten ---

# Für Balkendiagramm (Durchschnittswerte)
labels = ['FPS Ø', 'Verarbeitungszeit Ø (ms)', 'Stabilität (%)']
confidence_05 = [27.16, 8.81, 97.73]
confidence_08 = [20.6, 12.68, 99.0]



# --- Balkendiagramm erstellen ---

x = np.arange(len(labels))  # x-Positionen für Gruppen
width = 0.35  # Balkenbreite

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, confidence_05, width, label='Confidence 50%', color='skyblue')
bars2 = ax.bar(x + width/2, confidence_08, width, label='Confidence 80%', color='salmon')

# Achsentitel und Beschriftungen
ax.set_ylabel('Wert')
ax.set_title('Vergleich: Confidence 50% vs 80%')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Werte über Balken schreiben
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 Punkte hoch
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('src/final_report/data/Vergleich_Confidence_50_vs_80.png')

