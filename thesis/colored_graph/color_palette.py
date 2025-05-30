import warnings

class ColorPalette:
    # High-contrast distinct colors
    COLORS = [
        # Original 20
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",

        # New 20 with more hue variation
        "#1b9e77",  # greenish teal
        "#d95f02",  # orange-brown
        "#7570b3",  # blue-violet
        "#e7298a",  # vibrant pink
        "#66a61e",  # olive green
        "#e6ab02",  # golden yellow
        "#a6761d",  # earthy brown
        "#666666",  # deep gray
        "#17becf",  # cyan
        "#bcbd22",  # lime yellow
        "#f781bf",  # pastel pink
        "#999999",  # neutral gray
        "#386cb0",  # saturated blue
        "#f0027f",  # magenta pink
        "#fdc086",  # peach
        "#beaed4",  # lavender
        "#7fc97f",  # mint green
        "#ffff99",  # pale yellow
        "#fdcdac",  # light salmon
        "#b3cde3"  # pale blue
    ]

    @staticmethod
    def map_color_id_to_hex_color_by_frequency(color_counts: {}):
        sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
        color_map = {}
        for i, (color_id, _) in enumerate(sorted_colors):
            if i < len(ColorPalette.COLORS):
                color_map[color_id] = ColorPalette.COLORS[i]
            else:
                warnings.warn(f"Ran out of colors! Assigning numeric labels for remaining colors.")
                color_map[color_id] = None  # fallback: no color
        return color_map

    @staticmethod
    def map_color_id_to_hex_color_consecutively(color_ids: []):
        color_map = {}
        for color_id in color_ids:
            if color_id < len(ColorPalette.COLORS):
                color_map[color_id] = ColorPalette.COLORS[color_id]
            else:
                warnings.warn(f"Ran out of colors! Assigning numeric labels for remaining colors.")
                color_map[color_id] = None  # fallback: no color
        return color_map