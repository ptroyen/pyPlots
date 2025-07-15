# pyPlots/pyplots/config_module.py

import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: Import scienceplots if it's installed
try:
    import scienceplots
    _SCIENCEPLOTS_AVAILABLE_GLOBAL = True # Use a module-level global for availability
except ImportError:
    _SCIENCEPLOTS_AVAILABLE_GLOBAL = False

class PlotConfig:
    """Manages plot styling and configuration for scientific plots."""
    
    def __init__(self, project_name=None, config_file=None):
        # Default configuration
        self.config = {
            "style": {
                "use_scienceplot": True,
                "scienceplot_style": "ieee",
                "use_latex": True,
                "figure_dpi": 300,
                "figsize": [8, 6]  # Add figsize to default config
            },
            "fonts": {
                "family": "serif",
                "size": 10,
                "title_size": 12,
                "label_size": 10,
                "legend_size": 9,
                "tick_size": 8
            },
            "markers": {
                "default": "o",
                "size": 5,
                "every": 1
            },
            "lines": {
                "width": 1.5,
                "style": "-"
            },
            "colors": None  # Use matplotlib defaults
        }
        
        # Load project configuration if specified
        if project_name:
            self.load_project_config(project_name)
        
        # Override with specific config file if provided
        if config_file:
            self.load_config_file(config_file)
    
    def load_project_config(self, project_name):
        """Load configuration from project name or path"""
        # Check if it's a full path
        if os.path.isfile(project_name):
            config_path = project_name
        # Check if it's a json file in current directory
        elif os.path.isfile(f"{project_name}.json"):
            config_path = f"{project_name}.json"
        # Check standard locations
        elif os.path.isfile(os.path.expanduser(f"~/.pyplots/{project_name}.json")):
            config_path = os.path.expanduser(f"~/.pyplots/{project_name}.json")
        elif os.path.isfile(f"./config/{project_name}.json"):
            config_path = f"./config/{project_name}.json"
        else:
            raise FileNotFoundError(f"Could not find project config for '{project_name}'")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        self._update_config(self.config, config_data)

    def load_config_file(self, config_file):
        """Load configuration from a JSON file"""
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
            
            # Deep update of configuration
            self._update_config(self.config, custom_config)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading configuration from {config_file}: {e}")
    
    def _update_config(self, target, source):
        """Recursively update nested dictionary"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target:
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def apply_style(self):
        """Apply the configured style to matplotlib"""
        # Reset to default style first
        plt.style.use('default')
        
        # Apply SciencePlots if enabled
        # Use the module-level global for scienceplots availability
        if self.config["style"]["use_scienceplot"] and _SCIENCEPLOTS_AVAILABLE_GLOBAL:
            style = self.config["style"].get("scienceplot_style") # Use .get() for safety
            
            # CRITICAL FIX: Check if style is a non-empty string and valid
            if isinstance(style, str) and style.lower() != 'none': # Check for 'none' explicitly
                if style in ['ieee', 'nature', 'science']:
                    try:
                        plt.style.use(['science', style])
                        print(f"Applied scienceplot style: {style}")
                    except Exception as e:
                        print(f"Warning: Could not apply scienceplot style '{style}': {e}")
                else: # Custom style that might not be in the direct list, but is a string
                    try:
                        plt.style.use(style) # Try to apply directly
                        print(f"Applied custom scienceplot style: {style}")
                    except Exception as e:
                        print(f"Warning: Could not apply custom scienceplot style '{style}': {e}. Applying basic 'science' style instead.")
                        try:
                            plt.style.use('science') # Fallback
                            print("Applied basic science style as fallback.")
                        except Exception as inner_e:
                            print(f"Further warning: Could not even apply basic 'science' style: {inner_e}")
            else:
                # If style is None, 'none', or invalid, just apply basic 'science' if use_scienceplot is true
                try:
                    plt.style.use('science')
                    print("Applied basic science style (no specific sub-style selected or style was 'none').")
                except Exception as e:
                    print(f"Warning: Could not apply basic 'science' style: {e}")
        
        # Configure LaTeX
        if self.config["style"]["use_latex"]:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
            print("LaTeX rendering enabled")
        else: # Ensure LaTeX is turned off if not requested
            plt.rcParams.update({"text.usetex": False})
        
        # Apply font settings
        plt.rcParams.update({
            "font.size": self.config["fonts"]["size"],
            "axes.titlesize": self.config["fonts"]["title_size"],
            "axes.labelsize": self.config["fonts"]["label_size"],
            "legend.fontsize": self.config["fonts"]["legend_size"],
            "xtick.labelsize": self.config["fonts"]["tick_size"],
            "ytick.labelsize": self.config["fonts"]["tick_size"],
            "figure.dpi": self.config["style"]["figure_dpi"]
        })
        
        # Return self for chaining
        return self
    
    def get_line_props(self, index=0):
        """Get line properties for the given index"""
        return {
            "linewidth": self.config["lines"]["width"],
            "linestyle": self.config["lines"]["style"],
            "marker": self.config["markers"]["default"],
            "markersize": self.config["markers"]["size"],
            "markevery": self.config["markers"]["every"]
        }
    
    def get_figsize(self):
        """Get the figure size from the config."""
        if "style" in self.config and "figsize" in self.config["style"]:
            figsize = self.config["style"]["figsize"]
            # Convert to tuple if it's a list (for consistency with matplotlib)
            return tuple(figsize) if isinstance(figsize, list) else figsize
        return (8, 6)  # Default figsize

    def plot(self, *args, **kwargs):
        """
        Compatibility method to support older code that might call plot_config.plot()
        Delegates to plt.plot()
        """
        import matplotlib.pyplot as plt
        return plt.plot(*args, **kwargs)

# Helper function to create a configured figure
def create_figure(figsize=None, return_axes=False, plot_config=None, **kwargs):
    """
    Create a figure with default or specified size and a PlotConfig.
    
    Args:
        figsize: Tuple of (width, height) in inches
        return_axes: If True, returns (fig, ax), otherwise (fig, config)
        plot_config: An existing PlotConfig object, or None to create a new one
        **kwargs: Additional arguments for PlotConfig if creating a new one
        
    Returns:
        If return_axes=True: tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        If return_axes=False: tuple of (matplotlib.figure.Figure, PlotConfig)
    """
    # Use the provided config or create a new one
    if plot_config is None:
        config = PlotConfig(**kwargs)
    else:
        config = plot_config
    
    # If figsize is explicitly specified, update the config to match
    if figsize is not None:
        # Convert to list if it's a tuple (for consistency in config)
        config.config["style"]["figsize"] = list(figsize) if isinstance(figsize, tuple) else figsize
    
    config.apply_style()
    
    # Use the updated figsize from config (now includes any override)
    actual_figsize = config.get_figsize()
    
    # Always create with subplots for consistency
    fig, ax = plt.subplots(figsize=actual_figsize)
    
    # Return based on the flag
    if return_axes:
        return fig, ax
    else:
        return fig, config