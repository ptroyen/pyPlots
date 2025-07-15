# pyPlots/pyplots/plot_saver.py
import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt # For type hinting and figure saving

from datetime import datetime

# Helper to generate timestamped filename
def _generate_timestamped_filename(prefix, extension):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"

def save_figure_image(fig, output_file=None, format='png'):
    """
    Saves a matplotlib figure to an image file (PNG, PDF, SVG, etc.).
    This is generally preferred over pickling for long-term storage or sharing.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        output_file (str, optional): The path to save the image. If None,
                                     a timestamped filename is generated.
        format (str, optional): The format of the image (e.g., 'png', 'pdf', 'svg').
                                Defaults to 'png'.
    """
    if output_file is None:
        output_file = _generate_timestamped_filename("plot_image", f".{format}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    fig.savefig(output_file, format=format, bbox_inches='tight') # bbox_inches for tight layout
    print(f"Figure image saved to {output_file}")


def save_pickled_figure(fig, output_file=None):
    """
    Saves a matplotlib figure using pickle.
    WARNING: Pickled figures are not recommended for long-term storage
    or portability due to security risks and compatibility issues across
    different Matplotlib/Python versions. Use save_figure_image or
    save_plot_data_and_settings_to_json for more robust storage.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        output_file (str, optional): The path to save the pickle file. If None,
                                     a timestamped filename is generated.
    Returns:
        str: The path to the saved file.
    """
    if output_file is None:
        output_file = _generate_timestamped_filename("plot_fig", ".pkl")
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(fig, f)
    print(f"Figure (pickled) saved to {output_file}")
    return output_file


def extract_plot_data_and_settings(fig):
    """
    Extract data and settings from a matplotlib figure for JSON storage.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to extract data from
        
    Returns:
        tuple: (data_dict, curve_props, settings) or (None, None, None) if extraction fails
    """
    ax = fig.axes[0] if fig.axes else None
    if not ax:
        print("Warning: No axes found in figure, cannot extract data.")
        return None, None, None
    
    lines = ax.get_lines()
    if not lines:
        print("Warning: No data lines found in figure.")
        return None, None, None
    
    # Extract x-data (assuming common x-axis)
    x_data, _ = lines[0].get_data()
    data_dict = {"x": x_data.tolist()}
    
    # Extract y-data and properties for each curve
    curve_props = []
    for i, line in enumerate(lines):
        _, y_data = line.get_data()
        data_dict[f"y{i+1}"] = y_data.tolist()
        
        curve_props.append({
            "label": line.get_label() if line.get_label() != '_nolegend_' else f"Curve {i+1}",
            "color": line.get_color(),
            "linestyle": line.get_linestyle(),
            "marker": line.get_marker(),
            "markersize": line.get_markersize(),
            "markevery": line.get_markevery()
        })
    
    # Extract plot settings
    settings = {
        "figsize": fig.get_size_inches().tolist(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "title": ax.get_title(),
        "xscale": ax.get_xscale(),
        "yscale": ax.get_yscale(),
        "xlim": list(ax.get_xlim()),
        "ylim": list(ax.get_ylim()),
        "grid": ax.grid(),
        "save_img_fmts": ["png", "pdf"]
    }
    
    # Add legend settings
    legend = ax.get_legend()
    if legend:
        settings["legend"] = "on"
        settings["legend_params"] = {"frameon": legend.get_frame_on()}
    else:
        settings["legend"] = "off"
    
    # Check for scienceplots
    from .plotting import SCIENCEPLOTS_AVAILABLE
    if SCIENCEPLOTS_AVAILABLE:
        for style in ["ieee", "nature", "science"]:
            if style in plt.style.available:
                settings["style"] = style
                settings["use_scienceplot"] = True
                break
    
    # Check for LaTeX
    settings["use_latex"] = plt.rcParams.get("text.usetex", False)
    
    return data_dict, curve_props, settings


def save_plot_to_json(fig, output_file=None):
    """
    Extract data and settings from a matplotlib figure and save to JSON.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to save
        output_file (str, optional): Path to save JSON file
        
    Returns:
        str: Path to the saved file
    """
    if output_file is None:
        output_file = _generate_timestamped_filename("plot", ".json")
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Extract data and settings
    data_dict, curve_props, settings = extract_plot_data_and_settings(fig)
    if data_dict is None:
        return None
    
    # Create output structure
    plot_data = {
        "data": data_dict,
        "curves": curve_props,
        "settings": settings
    }
    
    # Save to JSON
    try:
        with open(output_file, 'w') as f:
            json.dump(plot_data, f, indent=2)
        print(f"Plot saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving plot to {output_file}: {e}")
        return None
    
def save_plot_styling_to_json(plot_settings, output_file=None):
    """
    Saves only the styling settings from a plot configuration to a JSON file.
    
    Args:
        plot_settings (dict): Dictionary of plot settings
        output_file (str, optional): Path to save JSON file. If None, generates a timestamped name.
        
    Returns:
        str: Path to the saved JSON file
    """
    if output_file is None:
        output_file = _generate_timestamped_filename("plot_styling", ".json")
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Extract only styling-related settings
    styling_settings = {
        "use_latex": plot_settings.get("use_latex", False),
        "scienceplot_style": plot_settings.get("scienceplot_style"),
        "figsize": plot_settings.get("figsize"),
        "xlabel": plot_settings.get("xlabel"),
        "ylabel": plot_settings.get("ylabel"),
        "title": plot_settings.get("title"),
        "xscale": plot_settings.get("xscale"),
        "yscale": plot_settings.get("yscale"),
        "xlim": plot_settings.get("xlim"),
        "ylim": plot_settings.get("ylim"),
        "legend": plot_settings.get("legend"),
        "legend_params": plot_settings.get("legend_params"),
        "grid": plot_settings.get("grid"),
        "style": plot_settings.get("style"),
        "save_img_fmts": ["png", "pdf"]  # Default image formats
    }
    
    # Remove None values
    styling_settings = {k: v for k, v in styling_settings.items() if v is not None}
    
    try:
        with open(output_file, 'w') as f:
            json.dump(styling_settings, f, indent=2)
        print(f"Saved styling settings to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving styling settings to {output_file}: {e}")
        return None

def save_multiplot_to_json(datasets_data, plot_settings, output_file=None):
    """
    Saves a multi-plot configuration with all datasets and settings to a JSON file.
    
    Args:
        datasets_data (list): List of dataset information dictionaries
        plot_settings (dict): Dictionary of common plot settings
        output_file (str, optional): Path to save JSON file. If None, generates a timestamped name.
        
    Returns:
        str: Path to the saved JSON file
    """
    if output_file is None:
        output_file = _generate_timestamped_filename("multiplot_data", ".json")
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # Combine datasets and settings
    combined_data = {
        "datasets": datasets_data,
        "plot_settings": plot_settings
    }
    
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Multi-plot data and settings saved to {output_file}")
    return output_file


