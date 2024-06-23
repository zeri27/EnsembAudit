import matplotlib.pyplot as plt

# Data for Localization Noise
noise_levels = [10, 25, 50]

precision = [0.5535714285714286, 0.4888888888888889, 0.531615925058548]
recall = [0.6595744680851063, 0.5906040268456376, 0.6358543417366946]
f1_score = [0.6019417475728156, 0.5349544072948328, 0.5790816326530613]
map_score = [0.2867261904761905, 0.28089812882223597, 0.4128274060546387]

actual_errors = [1750, 4375, 8751]
detected_errors = [248, 286, 423]
true_errors = [40, 104, 262]

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

bar_width = 0.2
index = range(len(noise_levels))

# Plotting Precision, Recall, F1-score, and mAP Score
bar1 = ax1.bar(index, precision, bar_width, label='Precision', color='b')
bar2 = ax1.bar([i + bar_width for i in index], recall, bar_width, label='Recall', color='g')
bar3 = ax1.bar([i + 2 * bar_width for i in index], f1_score, bar_width, label='F1-score', color='r')
bar4 = ax1.bar([i + 3 * bar_width for i in index], map_score, bar_width, label='mAP Score', color='c')

ax1.set_ylabel('Score')
ax1.set_title('Performance Metrics by Localization Noise Level')
ax1.set_xticks([i + 1.5 * bar_width for i in index])
ax1.set_xticklabels(noise_levels)
ax1.legend(loc='upper left')

# Adding values on top of bars for Precision, Recall, F1-score, mAP Score
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
add_labels(bar4)

# Creating a second y-axis for actual errors, detected errors, true errors
bar5 = ax2.bar(index, actual_errors, bar_width, label='Actual Errors', color='navy', alpha=0.7)
bar6 = ax2.bar([i + bar_width for i in index], detected_errors, bar_width, label='Detected Errors', color='darkorchid', alpha=0.7)
bar7 = ax2.bar([i + 2 * bar_width for i in index], true_errors, bar_width, label='True Errors', color='darkgoldenrod', alpha=0.7)

ax2.set_xlabel('Localization Noise Level')
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

add_count_labels(bar5, 0)
add_count_labels(bar6, 10)
add_count_labels(bar7, 0)

plt.tight_layout()
plt.show()
