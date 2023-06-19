import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)  # for reproducibility


def classification():
    # Creating the dictionary with languages as keys and accuracies as values
    languages_and_accuracies = {'ZH': 0.99 * 100,
                                'EML': 0.99 * 100,
                                'FUR': 1.0 * 100,
                                'LIJ': 0.98 * 100,
                                'LMO': 0.99 * 100,
                                'NAP': 0.99 * 100,
                                'PMS': 1.0 * 100,
                                'ROA TARA': 1.0 * 100,
                                'SC': 0.97 * 100,
                                'SCN': 0.99 * 100,
                                'VEC': 0.99 * 100,
                                'ACH': 0.99 * 100,
                                'DRE': 0.99 * 100,
                                'GRO': 0.99 * 100,
                                'HAM': 0.95 * 100,
                                'HOL': 0.94 * 100,
                                'MAR': 0.99 * 100,
                                'MKB': 0.97 * 100,
                                'MON': 0.99 * 100,
                                'NNI': 0.99 * 100,
                                'NPR': 0.99 * 100,
                                'OFL': 0.98 * 100,
                                'OFR': 0.99 * 100,
                                'OVY': 0.99 * 100,
                                'OWL': 0.99 * 100,
                                'SUD': 0.97 * 100,
                                'TWE': 0.99 * 100,
                                }

    italian = ["EML", "FUR", 'LIJ', 'LMO', 'NAP', 'PMS', 'ROA TARA', "SC", 'SCN', 'VEC']

    data = pd.DataFrame(list(languages_and_accuracies.items()), columns=['Language', 'Classification Accuracy'])

    # Setting up the Seaborn style
    sns.set_style("whitegrid")

    # Create a colorful bar plot
    plt.figure(figsize=(10, 6))

    # Create a color palette based on the languages
    palette = ['#fc6c00' if lang == 'ZH' else '#fc9d59' if lang in italian else '#fc8d90' for lang in data['Language']]

    bar_plot = sns.barplot(x='Language', y='Classification Accuracy', data=data, palette=palette)

    # Rotate x-labels if they overlap
    plt.xticks(rotation=45, fontsize=9)

    # Adjust the y-axis range to include higher values
    plt.ylim(90, 101)

    # Add the accuracy values on top of the bars with adjusted position
    for p, accuracy in zip(bar_plot.patches, data['Classification Accuracy']):
        bar_plot.annotate(f'{int(accuracy)}', (p.get_x() + p.get_width() / 2, p.get_height() + 0.2), ha='center', va='bottom', rotation=45)

    # Add the title and labels
    # plt.xlabel('Language')
    plt.ylabel('Accuracy')

    # Show the plot
    plt.show()


#classification()


def Sufficiency():
    dialects = ['ZH', 'LIJ']
    k_values = [1, 3, 5]
    accuracy_values = [[76.5, 96.7, 97.8],
                       [87.8, 95.8, 97.9]]

    # Create an empty figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set the x-axis and y-axis limits
    ax.set_xlim(-0.5, len(dialects) - 0.5)
    ax.set_ylim(70, 100)

    # x-axis tick positions and labels
    x_ticks = range(len(k_values))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(k_values)

    # Plot the bars for each dialect and k
    bar_width = 0.35
    for i in range(len(dialects)):
        ax.bar([x + i * bar_width for x in x_ticks], accuracy_values[i], width=bar_width, label=dialects[i])

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Number of Top k Explanations')
    ax.set_ylabel('Accuracy')

    # Set the title
    ax.set_title('Number of Top k Explanations vs Accuracy')

    # Show the legend
    ax.legend()

    # Show the grid
    ax.grid(True)

    # Show the plot
    plt.show()

#Sufficiency()




# Updating language pairs
language_pairs = ['CN', 'TW', 'SC', 'IT']
explanation_methods = ['LOO', 'SelfExplain']
evaluation_metrics = ['Strong indicator', 'Combined methods', 'PR']

# Updating data for the language pairs
data = {
    'CN': {'LOO': [100, 100, 100], 'SelfExplain': [100, 100, 100]},
    'TW': {'LOO': [70, 80, 100], 'SelfExplain': [70, 55, 100]},
    'SC': {'LOO': [20, 40, 80], 'SelfExplain': [20, 20, 85]},
    'IT': {'LOO': [60, 55, 46], 'SelfExplain': [60, 55, 57]}
}

bar_width = 0.15
x = np.arange(len(language_pairs))

fig, ax = plt.subplots()

# Using 'Spectral' seaborn color palette
colors = sns.color_palette("Spectral", len(explanation_methods) * len(evaluation_metrics))

for i, method in enumerate(explanation_methods):
    for j, metric in enumerate(evaluation_metrics):
        percentages = [data[lp][method][j] for lp in language_pairs]
        bars = ax.bar(x + i * bar_width * len(evaluation_metrics) + j * bar_width, percentages,
                      width=bar_width, label=f'{method} - {metric}', color=colors[i * len(evaluation_metrics) + j])


        # Adding data labels on top of each bar
        for bar in bars:
            yval = bar.get_height()

            ax.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom',rotation=45,
                    ha='center')  # positioning text in the center of the bar

# make the y-axis a little bit longer to make space for the text
ax.set_ylabel('Percentage of Important Features')
ax.set_xticks(x + bar_width * len(evaluation_metrics) / 2)
# move x-ticks to the right a little bit
ax.set_xticks(x + bar_width * len(evaluation_metrics))

ax.set_xticklabels(language_pairs)
 # Move the legend box outside, at the bottom of the plot area, and arrange it over 3 columns
ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2, frameon=False)

# Adjust layout to make space for the legend
plt.subplots_adjust(bottom=0.3)


# Extend the y-axis limit a little bit to accommodate the text
plt.ylim(0, max([max(data[lp][method]) for lp in language_pairs for method in explanation_methods]) + 12)

plt.show()
