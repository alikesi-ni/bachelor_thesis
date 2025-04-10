import warnings
import matplotlib.pyplot as plt

class ColorPalette:
    # High-contrast distinct colors
    COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
    ]

    @staticmethod
    def assign_color_map(color_counts):
        sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
        color_map = {}
        for i, (color_id, _) in enumerate(sorted_colors):
            if i < len(ColorPalette.COLORS):
                color_map[color_id] = ColorPalette.COLORS[i]
            else:
                warnings.warn(f"Ran out of colors! Assigning numeric labels for remaining colors.")
                color_map[color_id] = None  # fallback: no color
        return color_map