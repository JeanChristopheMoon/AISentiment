import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your labeled data
file_path = "structured_labeled_headlines.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please ensure the classification step has completed.")

# Read the CSV
df = pd.read_csv(file_path)

# Set style for better visuals
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Function to plot label distribution
def plot_label_distribution(label_column, title, color_palette):
    label_counts = df[label_column].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette=color_palette, legend=False)
    plt.title(f"{title}", fontsize=16)
    plt.ylabel("Number of Headlines")
    plt.xlabel(label_column.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 🔵 Topic Distribution
plot_label_distribution("topic", "🧠 Topic Distribution", "mako")

# 🟢 Tone Distribution
plot_label_distribution("tone", "🎭 Tone Distribution", "crest")

# 🟣 Frame Distribution
plot_label_distribution("frame", "🧱 Frame Distribution", "viridis")
