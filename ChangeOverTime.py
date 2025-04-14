import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read the labeled headlines data
df = pd.read_csv("labeled_with_dates.csv")

# Ensure the 'date' column exists
if 'date' not in df.columns:
    print("WARNING: No 'date' column found in your CSV. Make sure your original classification script included dates.")
    exit()

# Convert 'date' to datetime format and create a 'day' column
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["day"] = df["date"].dt.date

# Check for missing or NaN values in tone and frame columns
print("Missing tone values:", df["tone"].isna().sum())
print("Missing frame values:", df["frame"].isna().sum())

# Check unique values for tone and frame
print("Unique tone values:", df["tone"].unique())
print("Unique frame values:", df["frame"].unique())

# Drop rows with missing tone or frame values
df = df.dropna(subset=["tone", "frame"])

# Plot the trends of categories over time
def plot_category_trends(col_name, title):
    plt.figure(figsize=(12, 6))
    
    # Group data by the category and day
    trend_data = df.groupby([col_name, "day"]).size().reset_index(name="count")
    
    # Debug: Check the grouped data
    print("\nGrouped " + str(col_name) + " data:")
    print(trend_data.head())
    
    # If no data is available for this category, skip plotting
    if trend_data.empty:
        print("No data available for " + str(col_name))
        return
    
    # Pivot the data to create columns for each category
    trend_data_pivot = trend_data.pivot(index="day", columns=col_name, values="count").fillna(0)
    
    # Check the pivoted data
    print("\nPivoted data for " + str(col_name) + ":")
    print(trend_data_pivot.head())
    
    # Plot the data
    if not trend_data_pivot.empty:
        ax = trend_data_pivot.plot(kind="line", marker="o", figsize=(12, 6))
        plt.title(title + " Over Time")
        plt.ylabel("Count")
        plt.xlabel("Day")
        plt.xticks(rotation=45)
        plt.legend(title=col_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot to a file
        plt.savefig(str(col_name) + "_distribution_over_time.png")
        print("Plot saved as " + str(col_name) + "_distribution_over_time.png")
        
        # Show the plot
        plt.show()
    else:
        print("No data to plot for " + str(col_name))

# Plot the Topic, Tone, and Frame distributions over time
print("\n=== Plotting Topic Distribution ===")
plot_category_trends("topic", "Topic Distribution")

print("\n=== Plotting Tone Distribution ===")
plot_category_trends("tone", "Tone Distribution")

print("\n=== Plotting Frame Distribution ===")
plot_category_trends("frame", "Frame Distribution")

print("\nAnalysis complete! Check the current directory for saved plot images.")
