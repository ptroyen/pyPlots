# pyplots/__init__.py

"""
pyPlots: A versatile Python library for post-processing and plotting scientific data.

This package provides utilities for reading data from various formats,
creating customizable Matplotlib plots, and saving plot data and settings
for reproducibility and long-term storage.
"""

# Import core functionalities to make them directly accessible
# via 'from pyplots import function_name'
from .data_io import read_data
from .plotting import create_plot
from .plot_saver import save_plot_to_json, save_figure_image, extract_plot_data_and_settings

# You might also want to expose the main CLI entry point if it's convenient
# from .cli import main as cli_main

# Define the package version (optional, but good practice)
# This allows users to check pyplots.__version__
__version__ = "0.1.0"

# Optional: Define what is imported by default when doing 'from pyplots import *'
# __all__ = [
#     "read_data",
#     "create_plot",
#     "save_plot_data_and_settings_to_json",
#     "save_figure_image",
#     "extract_plot_data_and_settings_from_figure",
#     # "cli_main",
# ]