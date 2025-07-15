# pyPlots/pyplots/multi_plot.py

import json
import numpy as np
import matplotlib.pyplot as plt
import os # For os.path.exists and os.makedirs if saving directly from here
from datetime import datetime # For generating timestamps

# Import functions from your existing modules
from .data_io import read_data
from .plotting import create_plot # We will use this to draw individual lines
from .plot_saver import save_figure_image, save_pickled_figure, save_multiplot_to_json, save_plot_styling_to_json


def _detect_config_type(config):
    """Detect if config is standard config or saved JSON plot."""
    if not isinstance(config, dict):
        return "invalid"
    
    # Check for saved JSON plot structure
    if "version" in config and "datasets" in config and \
       all("data" in ds and "settings" in ds for ds in config["datasets"]):
        return "saved_json"
    
    # Check for standard config structure
    if "datasets" in config and "plot_settings" in config:
        return "standard"
    
    return "invalid"


def _strip_save_settings(plot_settings):
    """Remove save-related settings to prevent recursive saving."""
    save_keys = [
        "save_json", 
        "save_settings", 
        "save_pkl", 
        "save_img_name", 
        "save_img_fmts"
    ]
    return {k: v for k, v in plot_settings.items() if k not in save_keys}


def _convert_saved_json_to_config(saved_json):
    """Convert saved JSON plot format to standard config format."""
    config = {
        "datasets": [],
        "plot_settings": _strip_save_settings(saved_json.get("plot_settings", {}))
    }
    
    for dataset in saved_json["datasets"]:
        dataset_config = {
            "embedded_data": np.array(dataset["data"]),
            **dataset["settings"]
        }
        if "file_header" in dataset:
            dataset_config["file_header"] = dataset["file_header"]
        if "columns_map" in dataset:
            dataset_config["columns_map"] = dataset["columns_map"]
        
        config["datasets"].append(dataset_config)
    
    return config


def _save_enhanced_json(datasets_data, plot_settings, filepath):
    """Save plot data and settings in enhanced JSON format."""
    # Strip save-related settings before saving
    cleaned_settings = _strip_save_settings(plot_settings)
    
    json_data = {
        "version": "1.0",
        "datasets": datasets_data,
        "plot_settings": cleaned_settings  # Use cleaned settings
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved enhanced plot data to {filepath}")
    except Exception as e:
        print(f"Error saving enhanced JSON: {e}")

def load_plot_config(config_filepath):
    """Loads plot configuration from a JSON file."""
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Configuration file not found at {config_filepath}")
    
    try:
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not decode JSON from {config_filepath}. Check file format. Error: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading config file {config_filepath}: {e}")


def load_plot_from_json(json_filepath):
    """
    Load and recreate a plot from a saved JSON file.
    
    Args:
        json_filepath (str): Path to the JSON plot file
        
    Returns:
        matplotlib.figure.Figure: The recreated figure
    """
    if not os.path.exists(json_filepath):
        raise FileNotFoundError(f"JSON file not found at {json_filepath}")
    
    try:
        with open(json_filepath, 'r') as f:
            plot_data = json.load(f)
        
        # Check if this is a multi-plot JSON format
        if "datasets" in plot_data and "plot_settings" in plot_data:
            # Handle multi-plot format
            config = {
                "datasets": [],
                "plot_settings": plot_data["plot_settings"].copy()  # Make sure to copy
            }
            
            # Make sure style settings are preserved
            if "style" in plot_data["plot_settings"]:
                config["plot_settings"]["style"] = plot_data["plot_settings"]["style"]
            if "use_latex" in plot_data["plot_settings"]:
                config["plot_settings"]["use_latex"] = plot_data["plot_settings"]["use_latex"]
            
            # Process each dataset
            for dataset_info in plot_data["datasets"]:
                # Convert list data back to numpy array if needed
                data_array = np.array(dataset_info["data"]) if isinstance(dataset_info["data"], list) else dataset_info["data"]
                
                # Create a dataset config with embedded data
                dataset_config = {
                    "embedded_data": data_array,
                    "xcol": dataset_info["settings"]["xcol"],
                    "ycols": dataset_info["settings"]["ycols"],
                    "labels": dataset_info["settings"].get("labels", []),
                    "colors": dataset_info["settings"].get("colors", []),
                    "markers": dataset_info["settings"].get("markers", []),
                    "linestyles": dataset_info["settings"].get("linestyles", []),
                    "markersize": dataset_info["settings"].get("markersize"),
                    "markevery": dataset_info["settings"].get("markevery"),
                    "file_header": dataset_info.get("file_header"),
                    "columns_map": dataset_info.get("columns_map")
                }
                
                config["datasets"].append(dataset_config)
            
            # Create the plot from reconstructed config
            return plot_from_config(config)
        
        # Original code for single-plot format
        # Apply style settings
        settings = plot_data.get("settings", {})
        
        # Handle scienceplots
        if settings.get("style") in ["ieee", "nature", "science"]:
            from .plotting import SCIENCEPLOTS_AVAILABLE
            if SCIENCEPLOTS_AVAILABLE:
                try:
                    plt.style.use(['science', settings["style"]])
                except Exception as e:
                    print(f"Warning: Could not apply scienceplots style: {e}")
        
        # Handle LaTeX
        if settings.get("use_latex", False):
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
            })
        
        # Create figure and axis
        figsize = settings.get("figsize", (10, 6))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data
        data = plot_data.get("data", {})
        curves = plot_data.get("curves", [])
        
        if "x" not in data:
            raise ValueError("No x-data found in the plot file")
        
        x_data = np.array(data["x"])
        
        # Plot each curve
        for i, props in enumerate(curves):
            y_key = f"y{i+1}"
            if y_key in data:
                y_data = np.array(data[y_key])
                
                ax.plot(x_data, y_data,
                       label=props.get("label"),
                       color=props.get("color"),
                       linestyle=props.get("linestyle"),
                       marker=props.get("marker"),
                       markersize=props.get("markersize"),
                       markevery=props.get("markevery"),
                       alpha=0.7)
        
        # Apply axis settings
        ax.set_xlabel(settings.get("xlabel", ""))
        ax.set_ylabel(settings.get("ylabel", ""))
        ax.set_title(settings.get("title", ""))
        
        if "xlim" in settings:
            ax.set_xlim(settings["xlim"])
        if "ylim" in settings:
            ax.set_ylim(settings["ylim"])
        
        ax.set_xscale(settings.get("xscale", "linear"))
        ax.set_yscale(settings.get("yscale", "linear"))
        
        # Grid
        ax.grid(settings.get("grid", True))
        
        # Legend
        if settings.get("legend") == "on" and ax.get_legend_handles_labels()[0]:
            legend_params = settings.get("legend_params", {"frameon": False})
            ax.legend(**legend_params)
        
        return fig
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not decode JSON from {json_filepath}. Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading plot from JSON file: {e}")


def _parse_image_name_and_formats(save_img_name, save_img_fmts):
    """Helper to parse image name and formats."""
    base_name = save_img_name
    formats = []
    
    if save_img_name:
        # Split name and check extension
        base_name, ext = os.path.splitext(save_img_name)
        if ext and ext.lower() in ['.png', '.pdf', '.svg', '.jpg', '.jpeg', '.tiff', '.tif']:
            formats = [ext[1:].lower()]
    
    # If explicit formats provided, use those instead
    if save_img_fmts and isinstance(save_img_fmts, list):
        formats = [fmt.lower() for fmt in save_img_fmts 
                  if fmt.lower() in ['png', 'pdf', 'svg', 'jpg', 'jpeg', 'tiff', 'tif']]
    
    # Default to PNG if no valid formats found
    if not formats:
        formats = ['png']
        
    return base_name, formats

def _generate_default_image_name(config_file=None):
    """Generate default image name based on config or timestamp."""
    if config_file:
        base_name = os.path.splitext(os.path.basename(config_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"plot_{base_name}_{timestamp}"
    return f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def plot_from_config(config, return_datasets=False):
    """Reads data from multiple files based on config and plots them."""
    # Detect and convert config if needed
    config_type = _detect_config_type(config)
    if config_type == "invalid":
        raise ValueError("Invalid configuration format")
    elif config_type == "saved_json":
        config = _convert_saved_json_to_config(config)

    if "datasets" not in config or not isinstance(config["datasets"], list):
        raise ValueError("Configuration must contain a 'datasets' list.")
    
    common_plot_settings = config.get("plot_settings", {})
    
    # Reset any existing style and rcParams
    plt.style.use('default')
    plt.rcParams.update(plt.rcParamsDefault)
    
    # 1. Apply scienceplot style first if specified
    style = common_plot_settings.get("scienceplot_style")
    if style and style.lower() in ['ieee', 'nature', 'science']:
        from .plotting import SCIENCEPLOTS_AVAILABLE
        if SCIENCEPLOTS_AVAILABLE:
            try:
                if style.lower() == 'science':
                    plt.style.use('science')
                    print("Applied basic science style")
                else:
                    plt.style.use(['science', style.lower()])
                    print(f"Applied scienceplot style: {style}")
            except Exception as e:
                print(f"Warning: Could not apply scienceplots style: {e}")

    # 2. LaTeX setting overrides any style-based LaTeX settings
    use_latex = common_plot_settings.get("use_latex", False)
    if use_latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
        print(f"LaTeX rendering enabled")
    else:
        # Explicitly disable LaTeX, overriding any style settings
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif"
        })
        print(f"LaTeX rendering disabled")

    # Create figure and continue with rest of plotting...
    fig, ax = plt.subplots(figsize=common_plot_settings.get("figsize", (10, 6)))
    
    # Initialize datasets storage
    all_datasets_data = []
    plotted_any = False


    # --- Plot Data from Each Dataset in the Config ---
    for i, dataset_config in enumerate(config["datasets"]):
        print(f"Processing dataset {i+1}/{len(config['datasets'])}: {dataset_config.get('file', 'N/A')}")

        # Validate dataset configuration
        if not isinstance(dataset_config, dict):
            print(f"Warning: Dataset entry {i+1} is not an object. Skipping.")
            continue
        if "file" not in dataset_config and "embedded_data" not in dataset_config:
            print(f"Warning: Dataset entry {i+1} is missing 'file' or 'embedded_data' key. Skipping.")
            continue
        if "xcol" not in dataset_config or "ycols" not in dataset_config:
            print(f"Warning: Dataset entry {i+1} ({dataset_config.get('file', 'N/A')}) is missing 'xcol' or 'ycols' key(s). Skipping.")
            continue
        if not isinstance(dataset_config["ycols"], list):
            print(f"Warning: 'ycols' for dataset entry {i+1} ({dataset_config.get('file', 'N/A')}) must be a list. Skipping.")
            continue

        # Add this condition at the beginning of the dataset processing loop
        if "embedded_data" in dataset_config:
            # Use embedded data directly
            data_to_plot = dataset_config["embedded_data"]
            file_settings = {
                "header": dataset_config.get("file_header"),
                "columns_map": dataset_config.get("columns_map")
            }
            filepath = f"embedded_dataset_{i}"  # For logging purposes
            print(f"Using embedded data: {filepath}")
        else:
            # Original code for loading from file
            filepath = dataset_config["file"]
            try:
                result, file_settings = read_data(filepath, 
                                      delimiter=dataset_config.get("delimiter"),
                                      has_header=dataset_config.get("has_header", False),
                                      header_line_idx=dataset_config.get("header_line_idx", 0),
                                      comment_header_idx=dataset_config.get("comment_header_idx"))

                if isinstance(result, plt.Figure):
                    print(f"Warning: Input file {filepath} is a pickled figure (.pkl). For multi-plot, raw data is required. Skipping.")
                    continue

                data_to_plot = result
                print(f"Successfully read data from {filepath}. Shape: {data_to_plot.shape}")
            except FileNotFoundError as fnfe:
                print(f"Error: Dataset file not found at '{filepath}'. Skipping. ({fnfe})")
                continue
            except ValueError as ve:
                print(f"Error reading/parsing data from '{filepath}': {ve}. Skipping.")
                continue
            except Exception as e:
                print(f"An unexpected error occurred while processing dataset {filepath}: {e}. Skipping.")
                continue
            
        # Save dataset for potential JSON output
        dataset_info = {
            "filename": filepath,
            "data": data_to_plot.tolist(),  # Convert to list for JSON
            "settings": dataset_config,
            "file_header": file_settings.get("header", None),
            "columns_map": file_settings.get("columns_map", None)
        }
        all_datasets_data.append(dataset_info)
        
        # Resolve column names to indices if needed
        xcol_resolved = dataset_config["xcol"]
        ycols_resolved = dataset_config["ycols"]
        
        # If we have a columns_map from header, try to resolve column names
        if file_settings.get("columns_map"):
            columns_map = file_settings["columns_map"]
            
            # Resolve x column if it's a string name
            if isinstance(xcol_resolved, str) and not xcol_resolved.isdigit():
                if xcol_resolved in columns_map:
                    xcol_resolved = columns_map[xcol_resolved]
                    print(f"Resolved x-column name '{dataset_config['xcol']}' to index {xcol_resolved}")
                else:
                    print(f"Warning: x-column name '{xcol_resolved}' not found in header. Using as index.")
                    xcol_resolved = int(xcol_resolved)
            else:
                xcol_resolved = int(xcol_resolved)
            
            # Resolve y columns if they're string names
            resolved_ycols = []
            for ycol in ycols_resolved:
                if isinstance(ycol, str) and not ycol.isdigit():
                    if ycol in columns_map:
                        resolved_ycols.append(columns_map[ycol])
                        print(f"Resolved y-column name '{ycol}' to index {columns_map[ycol]}")
                    else:
                        print(f"Warning: y-column name '{ycol}' not found in header. Using as index.")
                        resolved_ycols.append(int(ycol))
                else:
                    resolved_ycols.append(int(ycol))
            ycols_resolved = resolved_ycols

        # Use resolved columns for plotting
        x_data = data_to_plot[:, xcol_resolved]
        
        plotted_dataset_lines = 0
        for j, ycol_idx in enumerate(ycols_resolved):
            if ycol_idx < data_to_plot.shape[1] and ycol_idx >= 0:
                y_data = data_to_plot[:, ycol_idx]
                label = dataset_config.get("labels", [])[j] if j < len(dataset_config.get("labels", [])) else f'{filepath} - Col {ycol_idx}'

                linestyle = dataset_config.get("linestyles", [])[j] if j < len(dataset_config.get("linestyles", [])) else '-'
                marker = dataset_config.get("markers", [])[j] if j < len(dataset_config.get("markers", [])) else None
                color = dataset_config.get("colors", [])[j] if j < len(dataset_config.get("colors", [])) else None
                markersize = dataset_config.get("markersize", None)
                markevery = dataset_config.get("markevery", None)  # Add this line

                ax.plot(x_data, y_data,
                        label=label,
                        linestyle=linestyle,
                        marker=marker,
                        color=color,
                        markersize=markersize,
                        markevery=markevery,  # Add this parameter
                        alpha=0.7) # Added alpha for better visibility of overlapping lines
                
                plotted_dataset_lines += 1
                plotted_any = True
            else:
                print(f"Warning: Y-column index {ycol_idx} is out of bounds for {filepath}. Skipping this column.")

        if plotted_dataset_lines == 0:
            print(f"Warning: No valid columns plotted for dataset {filepath}.")

    # --- Apply Common Plot Settings to the single Axes ---
    if plotted_any:
        ax.set_xlabel(common_plot_settings.get("xlabel", ""))
        ax.set_ylabel(common_plot_settings.get("ylabel", ""))
        ax.set_title(common_plot_settings.get("title", ""))

        xlim = common_plot_settings.get("xlim")
        if xlim and isinstance(xlim, list) and len(xlim) == 2:
            ax.set_xlim(xlim)
        ylim = common_plot_settings.get("ylim")
        if ylim and isinstance(ylim, list) and len(ylim) == 2:
            ax.set_ylim(ylim)

        xscale = common_plot_settings.get("xscale", "linear")
        if xscale in ['linear', 'log', 'symlog', 'logit']:
            ax.set_xscale(xscale)
        else:
            print(f"Warning: Invalid xscale '{xscale}' in config. Using 'linear'.")
            ax.set_xscale('linear')

        yscale = common_plot_settings.get("yscale", "linear")
        if yscale in ['linear', 'log', 'symlog', 'logit']:
            ax.set_yscale(yscale)
        else:
            print(f"Warning: Invalid yscale '{yscale}' in config. Using 'linear'.")
            ax.set_yscale('linear')

        legend_setting = common_plot_settings.get("legend", "on").lower()
        if ax.get_legend_handles_labels()[0]: # Check if there are any labels to show
            if legend_setting == 'on':
                # Default legend params
                legend_kwargs = {'frameon': False}  # Default to no frame
                # Update with custom params if provided
                legend_kwargs.update(common_plot_settings.get("legend_params", {}))
                ax.legend(**legend_kwargs)
            elif legend_setting == 'off':
                if ax.get_legend():
                    ax.get_legend().remove()

        grid_setting = common_plot_settings.get("grid", True)
        if isinstance(grid_setting, bool):
            ax.grid(grid_setting)
        else:
            print(f"Warning: Invalid grid setting '{grid_setting}' in config. Must be boolean (true/false). Using default (True).")
            ax.grid(True)

        # --- Save Outputs ---
        # First check for new multi-format image saving settings
        save_img_name = common_plot_settings.get("save_img_name")
        save_img_fmts = common_plot_settings.get("save_img_fmts", [])

        # Get base name and formats
        if save_img_name or save_img_fmts:
            base_name, formats = _parse_image_name_and_formats(save_img_name, save_img_fmts)
        else:
            # Generate default name if neither provided
            base_name = _generate_default_image_name(config.get("config_file"))
            formats = ['png']

        # Save in all specified formats
        for fmt in formats:
            try:
                output_path = f"{base_name}.{fmt}"
                save_figure_image(fig, output_path, format=fmt)
                print(f"Saved figure as {output_path}")
            except Exception as e:
                print(f"Error saving figure to {output_path}: {str(e)}")
                # Save pickle if requested
                save_pkl_filepath = common_plot_settings.get("save_pkl")
                if save_pkl_filepath:
                    try:
                        save_pickled_figure(fig, save_pkl_filepath)
                    except Exception as e:
                        print(f"Error saving figure to {save_pkl_filepath} (PKL): {str(e)}")
                
        # Save data and settings to JSON if requested
        save_json_filepath = common_plot_settings.get("save_json")
        if save_json_filepath and all_datasets_data:
            try:
                _save_enhanced_json(all_datasets_data, common_plot_settings, save_json_filepath)
            except Exception as e:
                print(f"Error saving enhanced JSON: {e}")
                    
        # Save only styling settings if requested
        save_settings_filepath = common_plot_settings.get("save_settings")
        if save_settings_filepath:
            save_plot_styling_to_json(common_plot_settings, save_settings_filepath)

        # IMPORTANT: Return proper values based on return_datasets flag
        if return_datasets:
            if plotted_any:
                return fig, all_datasets_data
            else:
                print("Warning: No data was plotted")
                return None, []
        else:
            return fig if plotted_any else None