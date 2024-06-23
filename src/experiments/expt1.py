import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = {
    'Model': [
        'Split 1', 'Split 2', 'Split 3', 'Split 4', 'Split 5', 'Ensemble',
        'Split 1', 'Split 2', 'Split 3', 'Split 4', 'Split 5', 'Ensemble'
    ],
    'Type': [
        'Scratch', 'Scratch', 'Scratch', 'Scratch', 'Scratch', 'Scratch',
        'Pretrained', 'Pretrained', 'Pretrained', 'Pretrained', 'Pretrained', 'Pretrained'
    ],
    'mAP Score': [
        0.6400574856695403, 0.630962600322583, 0.6334676193495812, 0.6301978225996827, 0.636904749903971, 0.6889654115269502,
        0.7574412686120604, 0.762736075793012, 0.7550524513914086, 0.76230508593289, 0.7580481440628779, 0.7883097455952949
    ]
}

df = pd.DataFrame(data)

# Create a grouped barplot with extensive seaborn customization
plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")
sns.set_palette("pastel")

ax = sns.barplot(x='Model', y='mAP Score', hue='Type', data=df, dodge=True, edgecolor=".2")

# Customize the plot
ax.set_ylim(0.6, 0.8)
ax.set_title('Comparison of mAP Scores for Scratch and Pretrained Models', fontsize=18, weight='bold')
ax.set_xlabel('Model Splits and Ensemble', fontsize=14)
ax.set_ylabel('mAP Scores', fontsize=14)
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Model Type', title_fontsize='13', fontsize='12')

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
