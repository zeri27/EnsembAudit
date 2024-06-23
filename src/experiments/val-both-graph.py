import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
noise_levels = ['Noise - 10', 'Noise - 25', 'Noise - 50']
models = ['SPLIT 1', 'SPLIT 2', 'SPLIT 3', 'SPLIT 4', 'SPLIT 5', 'ENSEMBLE MODEL']

# Model scores for each noise level
scores = [
    [0.7377864617063206, 0.699955745946481, 0.5876695503953062],
    [0.7384981515155086, 0.701902070858773, 0.574462598569774],
    [0.7450991879108878, 0.7018620397570816, 0.5867280688115626],
    [0.7325042473059751, 0.7019739030514378, 0.5815053220737509],
    [0.7359919764380607, 0.6948462361867345, 0.5735300160277343],
    [0.7751002353890366, 0.7522993521380739, 0.6720263332155824]
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
ax.set_title('Both Noise Levels: mAP@50 Scores', fontsize=18, weight='bold')
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
