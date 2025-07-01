import matplotlib.pyplot as plt

metriken = {
    'Precision': 0.89,
    'Recall': 0.85,
    'F1 Score': 0.87,
    'FPS': 24.5
}

plt.figure(figsize=(8,5))
plt.bar(metriken.keys(), metriken.values(), color=['blue', 'green', 'orange', 'red'])
plt.title('Leistungsmetriken Gesichtserkennung')
plt.ylim(0,1.2)
plt.ylabel('Wert')
plt.show()