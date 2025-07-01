import matplotlib.pyplot as plt
import numpy as np

# Sample data - Confidence scores for different detection scenarios
# This simulates varying confidence values for different face detection scenarios
detection_types = ['Frontal', 'Profile', 'Partially\nObscured', 'Low\nLight', 'Distance\n(5m+)', 'Multiple\nFaces']

# Mean confidence scores for YOLOv8-face detector
mean_scores = [0.92, 0.78, 0.68, 0.63, 0.57, 0.81]

# Standard deviation to simulate variance in detections
std_dev = [0.06, 0.09, 0.12, 0.14, 0.15, 0.08]

# Generate sample data points that would represent individual detections
num_samples = 30
np.random.seed(42)  # For reproducibility

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Create violin plots
violin_parts = ax.violinplot(
    [np.random.normal(m, s, num_samples) for m, s in zip(mean_scores, std_dev)],
    showmeans=False,
    showmedians=True
)

# Customize violin plots
for pc in violin_parts['bodies']:
    pc.set_facecolor('skyblue')
    pc.set_edgecolor('navy')
    pc.set_alpha(0.7)

for partname in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
    vp = violin_parts[partname]
    vp.set_edgecolor('navy')
    vp.set_linewidth(1.5)

# Add scatter points representing individual detections (jittered for visibility)
for i, (m, s) in enumerate(zip(mean_scores, std_dev)):
    # Generate sample data
    sample_data = np.random.normal(m, s, num_samples)
    
    # Clip values to be between 0 and 1 (as confidence scores)
    sample_data = np.clip(sample_data, 0, 1)
    
    # Create jittered x-positions
    x_jitter = np.random.normal(i+1, 0.07, size=num_samples)
    
    # Plot individual points
    ax.scatter(x_jitter, sample_data, s=15, alpha=0.4, c='navy', edgecolor='none')

# Add horizontal line for threshold (typical face detection threshold)
ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Typical Detection Threshold (0.6)')

# Improve visualization
ax.set_xticks(np.arange(1, len(detection_types) + 1))
ax.set_xticklabels(detection_types)
ax.set_ylabel('Confidence Score (0-1)', fontsize=12)
ax.set_xlabel('Face Detection Scenario', fontsize=12)
ax.set_title('YOLOv8-face Detection Confidence by Scenario', fontsize=14)
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='lower left')

# Annotate average confidence
for i, (m, s) in enumerate(zip(mean_scores, std_dev)):
    ax.text(i+1, 1.01, f'Avg: {m:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('yolo_confidence_scores.png')
plt.close()