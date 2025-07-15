# pyPlots/pyplots/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import os

from .config_module import PlotConfig, create_figure

try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False

def create_plot(data, args, existing_figure=None, plot_config=None):
    """
    Creates a Matplotlib plot based on numerical data and plotting arguments.
    """
    # Ensure data is a NumPy array for consistent indexing and shape
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if data is None or data.size == 0:
        raise ValueError("No valid data provided for plotting.")

    # Determine figsize based on args, then config if present
    figsize = None
    if hasattr(args, 'figsize') and args.figsize:
        figsize = tuple(args.figsize)
    elif plot_config: # If plot_config exists and args.figsize isn't set, get from config
        figsize = plot_config.get_figsize()
    
    # Create or use figure and axes
    if existing_figure:
        fig = existing_figure
        ax = fig.gca() 
        ax.clear()  # Clear existing content if re-plotting on the same axes
        # Note: Global styles are usually set once by config.apply_style() in cli.py.
        # Re-applying here might be redundant or problematic if it tries to reset global styles.
        # If specific plot_config styles need to apply per-figure, that logic might be in create_figure.
    else:
        # Use create_figure from config_module if plot_config is available
        if plot_config:
            # Request axes explicitly with return_axes=True
            fig, ax = create_figure(figsize=figsize, plot_config=plot_config, return_axes=True)
        else:
            # Legacy path for when no PlotConfig is involved
            if figsize is None: # Default if no config and no arg
                figsize = (10, 6)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            # Apply scienceplots style if requested and available (legacy path)
            # This logic should ideally be managed by PlotConfig.apply_style()
            if SCIENCEPLOTS_AVAILABLE and getattr(args, 'style', 'default') != 'default':
                # Check for explicit 'none'
                if getattr(args, 'scienceplot', 'none').lower() != 'none':
                    try:
                        style_to_use = getattr(args, 'scienceplot', getattr(args, 'style', 'default'))
                        if style_to_use != 'default': # Don't try to apply 'default' as a scienceplot style
                            plt.style.use(style_to_use)
                    except Exception as e:
                        print(f"Warning: Could not apply scienceplots style '{style_to_use}': {e}")
                elif getattr(args, 'style', 'default') == 'science':
                     try:
                        plt.style.use('science')
                     except Exception as e:
                        print(f"Warning: Could not apply scienceplots style 'science': {e}")


    # Get column indices (already resolved in cli.py to integers)
    xcol = getattr(args, 'xcol', 0)
    ycols = getattr(args, 'ycols', [1])

    # Validate column indices against data shape
    num_cols = data.shape[1]
    if xcol >= num_cols:
        raise IndexError(f"X-column index {xcol} out of bounds for data with {num_cols} columns.")
    
    valid_ycols = []
    for ycol in ycols:
        if ycol >= num_cols:
            print(f"Warning: Y-column index {ycol} out of bounds for data with {num_cols} columns. Skipping.")
        else:
            valid_ycols.append(ycol)
    ycols = valid_ycols # Update to only include valid columns

    if not ycols:
        raise ValueError("No valid Y-columns to plot after validation.")

    # Prepare plot styles from args (ensure they are iterable, not None)
    # This is the crucial part for the original len() error
    labels = getattr(args, 'labels', [])
    if labels is None: labels = [] # Defensive check
    
    linestyles = getattr(args, 'linestyles', [])
    if linestyles is None: linestyles = [] # Defensive check
    
    markers = getattr(args, 'markers', [])
    if markers is None: markers = [] # Defensive check
    
    colors = getattr(args, 'colors', [])
    if colors is None: colors = [] # Defensive check
    
    markersize = getattr(args, 'markersize', None)


    for i, ycol in enumerate(ycols):
        x_data = data[:, xcol]
        y_data = data[:, ycol]

        plot_kwargs = {}
        
        # Get default properties from plot_config if available
        if plot_config:
            plot_kwargs.update(plot_config.get_line_props(i))
        
        # CLI arguments override configuration properties
        # Only apply if the CLI arg is explicitly set and not empty/None
        if labels and i < len(labels): plot_kwargs['label'] = labels[i]
        if linestyles and i < len(linestyles): plot_kwargs['linestyle'] = linestyles[i]
        if markers and i < len(markers): plot_kwargs['marker'] = markers[i]
        if colors and i < len(colors): plot_kwargs['color'] = colors[i]
        if markersize is not None: plot_kwargs['markersize'] = markersize

        ax.plot(x_data, y_data, **plot_kwargs)

    # Set plot labels, limits, scales, and title
    # Defensive getattr with meaningful defaults if not found
    ax.set_xlabel(getattr(args, 'xlabel', ''))
    ax.set_ylabel(getattr(args, 'ylabel', ''))
    ax.set_title(getattr(args, 'title', ''))

    xlim = getattr(args, 'xlim', None)
    if xlim is not None and len(xlim) == 2:
        ax.set_xlim(xlim)
    ylim = getattr(args, 'ylim', None)
    if ylim is not None and len(ylim) == 2:
        ax.set_ylim(ylim)

    ax.set_xscale(getattr(args, 'xscale', 'linear'))
    ax.set_yscale(getattr(args, 'yscale', 'linear'))

    # Add legend if enabled
    if getattr(args, 'legend', 'on').lower() == 'on':
        handles, labels_from_lines = ax.get_legend_handles_labels()
        if handles: # Only show legend if there are items to show
            # Get legend parameters from plot_config if available
            legend_kwargs = {'frameon': False}  # Default to no frame
            if plot_config and hasattr(plot_config, 'config') and 'legend_params' in plot_config.config:
                legend_kwargs.update(plot_config.config['legend_params'])
            ax.legend(**legend_kwargs)

    # Add grid if enabled
    if getattr(args, 'grid', False):
        ax.grid(True)
        
    fig.tight_layout() # Adjust layout to prevent labels/titles overlapping

    return fig