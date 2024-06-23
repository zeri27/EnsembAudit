import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
noise_levels = ['Noise - 10', 'Noise - 25', 'Noise - 50']
models = ['SPLIT 1', 'SPLIT 2', 'SPLIT 3', 'SPLIT 4', 'SPLIT 5', 'ENSEMBLE MODEL']

# Model scores for each noise level

scores = [
    [0.7487080552141292, 0.7367871746586128, 0.7216621611361209],
    [0.7531927844674263, 0.7413235935719709, 0.7170312756761873],
    [0.7556328420298424, 0.7356777637145376, 0.721841939291888],
    [0.7491688093885412, 0.7407774899215966, 0.7134369398920251],
    [0.7500653967692634, 0.7349172872300148, 0.7200900926652697],
    [0.7848924977106877, 0.7727972126990517, 0.7660946349382296]
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
ax.set_title('Localization Noise Levels: mAP@50 Scores', fontsize=18, weight='bold')
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
