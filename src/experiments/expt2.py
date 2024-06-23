import matplotlib.pyplot as plt

# Data
noise_levels = [10, 25, 50]

precision = [0.6346153846153846, 0.6752688172043011, 0.8099496925656792]
recall = [0.7734375, 0.7584541062801933, 0.6637654603756299]
map_score = [0.36113911300830803, 0.29482637065287454, 0.2817951528849413]

actual_errors = [1750, 4375, 8751]
detected_errors = [371, 690, 2130]
true_errors = [87, 288, 1397]

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 8))

bar_width = 0.2
index = range(len(noise_levels))

# Plotting Precision, Recall, and mAP Score
bar1 = ax1.bar(index, precision, bar_width, label='Precision', color='b')
bar2 = ax1.bar([i + bar_width for i in index], recall, bar_width, label='Recall', color='g')
bar3 = ax1.bar([i + 2 * bar_width for i in index], map_score, bar_width, label='mAP Score', color='r')

ax1.set_xlabel('Noise Level')
ax1.set_ylabel('Score')
ax1.set_title('Performance Metrics by Noise Level')
ax1.set_xticks([i + bar_width for i in index])
ax1.set_xticklabels(noise_levels)
ax1.legend(loc='upper left')

# Adding values on top of bars for Precision, Recall, mAP Score
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

# Creating a second y-axis for actual errors, detected errors, true errors
ax2 = ax1.twinx()
bar4 = ax2.bar([i - bar_width for i in index], actual_errors, bar_width, label='Actual Errors', color='navy', alpha=0.7)
bar5 = ax2.bar(index, detected_errors, bar_width, label='Detected Errors', color='darkorchid', alpha=0.7)
bar6 = ax2.bar([i + bar_width for i in index], true_errors, bar_width, label='True Errors', color='darkgoldenrod', alpha=0.7)

ax2.set_ylabel('Count')
ax2.legend(loc='upper right')

# Adding values on top of bars for Actual Errors, Detected Errors, True Errors
def add_count_labels(bars, offset):
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_count_labels(bar4, 0)
add_count_labels(bar5, 10)
add_count_labels(bar6, 0)

plt.tight_layout()
plt.show()
