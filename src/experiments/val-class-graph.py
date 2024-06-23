import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
noise_levels = ['Noise - 10', 'Noise - 25', 'Noise - 50']
models = ['SPLIT 1', 'SPLIT 2', 'SPLIT 3', 'SPLIT 4', 'SPLIT 5', 'ENSEMBLE MODEL']

# Model scores for each noise level
scores = [
    [0.7429082146751868, 0.7155313584366787, 0.5789198022853745],
    [0.7394807787916569, 0.7149130284196775, 0.6011295589620325],
    [0.7459131901262521, 0.7126438658807377, 0.5961999186527971],
    [0.7439725140718846, 0.7125720952288261, 0.5919346200402515],
    [0.7453449884004522, 0.7073461337968795, 0.6087636806357666],
    [0.7750279836400245, 0.7552396213870599, 0.6867963740667701]
]

# Create a DataFrame
data = {
    'Noise Level': noise_levels * len(models),
    'Model': [model for model in models for _ in noise_levels],
    'mAP Score': [score for sublist in scores for score in sublist]
}
df = pd.DataFrame(data)

# Create a grouped barplot with extensive seaborn customization
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
sns.set_palette("pastel")

ax = sns.barplot(x='Noise Level', y='mAP Score', hue='Model', data=df, dodge=True, edgecolor=".2")

# Customize the plot
ax.set_ylim(0.5, 0.8)
ax.set_title('Classification Noise Levels: mAP@50 Scores', fontsize=18, weight='bold')
ax.set_xlabel('Noise Level', fontsize=14)
ax.set_ylabel('mAP Scores', fontsize=14)
ax.legend(title='Model', title_fontsize='13', fontsize='12')

# Adding data labels
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

sns.despine(left=True, bottom=True)
plt.grid(True, linestyle='--', alpha=0.7)

# Ensure x-axis labels are horizontal
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
