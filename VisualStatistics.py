import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the saved file (labeled_headlines.json or structured_labeled_headlines.csv)
df_labels = pd.read_csv("structured_labeled_headlines.csv")  # Or use labeled_headlines.json

# Setting up Seaborn style for plots
sns.set(style="whitegrid")

# Function to create bar plots
def plot_label_distribution(df, label_column, title, filename):
    plt.figure(figsize=(10, 6))
    label_counts = df[label_column].value_counts()
    
    # Create a bar plot
    sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis")
    
    # Title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(label_column, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Save the plot as an image
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.show()

# Plot distributions for Topic, Tone, and Frame
plot_label_distribution(df_labels, 'topic', 'Distribution of Topics', 'topic_distribution')
plot_label_distribution(df_labels, 'tone', 'Distribution of Tones', 'tone_distribution')
plot_label_distribution(df_labels, 'frame', 'Distribution of Frames', 'frame_distribution')
