"""Explore labels of the training data."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data_directory = Path('./data')
train = pd.read_csv(data_directory / 'train.csv')

# Save label EDA
output_directory = Path('./reports/output')
with open(output_directory / 'label_eda.txt', 'w') as f:
    # Print high-level information about training data
    print("Exploration", file=f)
    print(train.head(), file=f)
    print(f"\nTraining set has {train.shape[0]:,} rows", file=f)
    print(f"{train['toxic'].sum()/train.shape[0]:.2%} Toxic", file=f)
    print(f"{train['severe_toxic'].sum()/train.shape[0]:.2%} Severe Toxic",
          file=f)
    print(f"{train['obscene'].sum()/train.shape[0]:.2%} Obscene", file=f)
    print(f"{train['threat'].sum()/train.shape[0]:.2%} Threat", file=f)
    print(f"{train['insult'].sum()/train.shape[0]:.2%} Insult", file=f)
    print(f"{train['identity_hate'].sum()/train.shape[0]:.2%} Identity Hate",
          file=f)
    print("\nNumber missing in each column:", file=f)
    print(train.isna().sum(), file=f)

# Calculate label percentages (each obs can have multiple)
label_freq = train.iloc[:, 2:].sum() / train.shape[0]
plt.figure(figsize=(6.5, 3))
plt.style.use('default')
label_bar = sns.barplot(label_freq.index, label_freq.values,
                        color=sns.color_palette()[0])
plt.title("Label Frequency")
plt.xlabel("Label")
yticks = label_bar.get_yticks()
label_bar.set_yticklabels(['{:.0%}'.format(i) for i in yticks])
plt.ylabel("Percentage Frequency")
plt.tight_layout()
output_directory = Path('./reports/figures')
plt.savefig(output_directory / 'label_freq.png')
