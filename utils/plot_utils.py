import pandas as pd
import matplotlib.pyplot as plt

def plot_prediction_distribution(df: pd.DataFrame, selected_ids: list[str]):
    """
    Plot predicted vs true label distribution for selected videos.

    Args:
        df: pandas DataFrame with at least columns ["video_id", "predicted_label", "true_label"]
        selected_ids: list of video_id strings to include in the plot

    Returns:
        A matplotlib figure object.
    """
    # Filter
    sub_df = df[df["video_id"].isin(selected_ids)]

    # Count distributions
    pred_counts = sub_df["predicted_label"].value_counts().sort_index()
    true_counts = sub_df["true_label"].value_counts().sort_index()
    labels = sorted(set(pred_counts.index).union(true_counts.index))

    # Create aligned count lists
    pred = [pred_counts.get(label, 0) for label in labels]
    true = [true_counts.get(label, 0) for label in labels]

    # Plot
    fig, ax = plt.subplots(figsize=(4, 3))
    width = 0.35
    x = range(len(labels))
    ax.bar([i - width/2 for i in x], true, width, label="True", color="green")
    ax.bar([i + width/2 for i in x], pred, width, label="Predicted", color="blue")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution (True vs. Predicted)")
    ax.legend()

    plt.tight_layout()
    return fig
