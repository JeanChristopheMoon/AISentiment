import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your labeled data with dates
file_path = "labeled_with_dates.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please ensure the labeled data with dates is available.")

# Read the CSV
df = pd.read_csv(file_path)

# Convert 'date' to datetime format and create a 'day' column
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["day"] = df["date"].dt.date

# Check for missing tone values
print("Missing tone values:", df["tone"].isna().sum())
print("Unique tone values:", df["tone"].unique())

# Drop rows with missing tone values
df = df.dropna(subset=["tone"])

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
    plt.show()  # Display the plot

# Function to plot time series distribution
def plot_time_series_distribution(label_column, title, color_palette):
    # Group data by label and day
    trend_data = df.groupby([label_column, "day"]).size().reset_index(name="count")
    
    # If no data is available, exit the function
    if trend_data.empty:
        print(f"No {label_column} data available")
        return
    
    # Pivot the data to create columns for each label value
    data_pivot = trend_data.pivot(index="day", columns=label_column, values="count").fillna(0)
    print(f"\nPivoted data for {label_column}:")
    print(data_pivot.head())
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    data_pivot.plot(kind="line", marker="o", figsize=(12, 6), colormap=color_palette)
    plt.title(f"{title}", fontsize=16)
    plt.ylabel("Count")
    plt.xlabel("Day")
    plt.xticks(rotation=45)
    plt.legend(title=label_column.capitalize())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()  # Display the plot

# 🔵 Topic Distribution
plot_label_distribution("topic", "🧠 Topic Distribution", "mako")

# 🟢 Tone Distribution
plot_label_distribution("tone", "🎭 Tone Distribution", "crest")

# 🟣 Frame Distribution
plot_label_distribution("frame", "🧱 Frame Distribution", "viridis")

# 📊 Time Series Distribution for Tone
plot_time_series_distribution("tone", "🎭 Tone Distribution Over Time", "crest")
