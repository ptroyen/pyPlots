# pyPlots/pyplots/data_io.py

import numpy as np
import json
import pickle
import matplotlib.pyplot as plt # Needed for isinstance(result, plt.Figure) check

def load_figure_from_pickle(filename):
    """Load a previously saved matplotlib figure from a pickle file."""
    # Warning: Pickled figures are not recommended for long-term storage
    # or portability due to security risks and compatibility issues.
    # Consider saving data and plot settings to JSON instead.
    try:
        with open(filename, 'rb') as f:
            fig = pickle.load(f)
        return fig
    except Exception as e:
        raise IOError(f"Error loading pickled figure from {filename}: {e}")

def read_data(filename):
    """
    Read data from file, supporting plain text (space-separated, no header),
    JSON, and pickled Matplotlib figures.

    Returns:
        tuple: (data_array, plot_settings_dict) or (matplotlib.Figure, None)
    """
    filename_str = str(filename) # Convert to string
    if filename_str.endswith('.pkl'):
        # For .pkl files, we return the figure object itself and no settings
        return load_figure_from_pickle(filename_str), None
    elif filename_str.endswith('.json'):
        with open(filename_str, 'r') as f:
            json_data = json.load(f)
        # Ensure 'data' and 'plot_settings' keys exist, provide defaults if not
        data_array = np.array(json_data.get('data', []))
        plot_settings = json_data.get('plot_settings', {})
        if data_array.size == 0 and not plot_settings:
            raise ValueError(f"JSON file '{filename_str}' contains no valid data or plot settings.")
        return data_array, plot_settings
    else:
        # Plain text file reading logic (space-separated)
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comment lines (starting with '#')
                if line and not line.startswith('#'):
                    try:
                        row = [float(x) for x in line.split()] # Assumes space-separated values
                        data.append(row)
                    except ValueError as e:
                        print(f"Warning: Skipping malformed line in '{filename}': '{line}'. Error: {e}")
                        continue
        
        if not data:
            raise ValueError(f"No valid data found in file '{filename}' after skipping comments and empty lines.")
        
        # Returns numpy array and empty settings dictionary for text files
        return np.array(data), {}