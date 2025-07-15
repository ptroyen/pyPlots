# pyPlots/pyplots/cli.py

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

# Import functions from your new modules
from .data_io import read_data
from .plotting import create_plot, SCIENCEPLOTS_AVAILABLE
from .plot_saver import save_figure_image, save_pickled_figure, \
    save_plot_to_json, save_multiplot_to_json, extract_plot_data_and_settings
from .multi_plot import load_plot_config, plot_from_config  # Remove load_plot_from_json
from .config_module import PlotConfig # Import PlotConfig
import os

# Helper function (no changes)
def resolve_columns(xcol_arg, ycols_args, columns_map):
    resolved_xcol = None
    resolved_ycols = []

    if columns_map: # Header was read and mapped
        try:
            # Try to resolve xcol as a name first
            if isinstance(xcol_arg, str) and xcol_arg in columns_map:
                resolved_xcol = columns_map[xcol_arg]
            else: # If not a name, try converting to int
                resolved_xcol = int(xcol_arg)
        except (KeyError, ValueError):
            raise ValueError(f"Invalid x-column specified: '{xcol_arg}'. Not found in header or not a valid index.")

        for ycol_arg in ycols_args:
            try:
                if isinstance(ycol_arg, str) and ycol_arg in columns_map:
                    resolved_ycols.append(columns_map[ycol_arg])
                else:
                    resolved_ycols.append(int(ycol_arg))
            except (KeyError, ValueError):
                raise ValueError(f"Invalid y-column specified: '{ycol_arg}'. Not found in header or not a valid index.")
    else: # No header, assume numeric indices
        try:
            resolved_xcol = int(xcol_arg)
            resolved_ycols = [int(y) for y in ycols_args]
        except ValueError:
            raise ValueError("Data file has no header. Please specify x and y columns by integer index.")
            
    return resolved_xcol, resolved_ycols


def plot_single_file_cli(args):
    """Handles the single file plotting logic."""
    data = None
    fig = None
    file_settings = {}  # Initialize in case read_data doesn't return it structured
    effective_args = vars(args).copy()  # Will hold our resolved args

    # Load styling settings if provided
    if hasattr(args, 'settings') and args.settings:
        try:
            with open(args.settings, 'r') as f:
                styling_settings = json.load(f)
                
            # Apply styling settings (but don't override explicit CLI arguments)
            for key, value in styling_settings.items():
                current_arg_value = getattr(args, key, None) # Get current value, default to None if not present

                is_default = False
                if key in ['xlabel', 'ylabel']:
                    is_default = (current_arg_value == '' or current_arg_value is None)
                elif key in ['labels', 'markers', 'linestyles', 'colors']:
                    # Check if it's None OR an empty list/tuple
                    is_default = (current_arg_value is None or 
                                  (isinstance(current_arg_value, (list, tuple)) and len(current_arg_value) == 0))
                else:
                    # For other types, check if attribute exists and is None (or not set)
                    is_default = (not hasattr(args, key) or current_arg_value is None)

                if is_default:
                    setattr(args, key, value)
                    effective_args[key] = value
                    
            print(f"Applied styling settings from {args.settings}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading settings file: {e}")

    # Create PlotConfig based on CLI arguments
    plot_config = None
    if (hasattr(args, 'project') and args.project) or \
       (hasattr(args, 'config') and args.config) or \
       (hasattr(args, 'use_latex') and args.use_latex) or \
       (hasattr(args, 'scienceplot') and args.scienceplot) or \
       (hasattr(args, 'figsize') and args.figsize): # Added figsize to trigger config
        
        # Create base config without applying style yet
        project_name = getattr(args, 'project', None)
        config_file = getattr(args, 'config', None)
        plot_config = PlotConfig(project_name=project_name, config_file=config_file)
        
        # Override with specific CLI args
        if hasattr(args, 'use_latex') and args.use_latex:
            plot_config.config["style"]["use_latex"] = True
        
        if hasattr(args, 'scienceplot') and args.scienceplot:
            if args.scienceplot == "none":
                plot_config.config["style"]["use_scienceplot"] = False
            else:
                plot_config.config["style"]["use_scienceplot"] = True
                plot_config.config["style"]["scienceplot_style"] = args.scienceplot
        
        # Apply the figsize if specified
        if hasattr(args, 'figsize') and args.figsize:
            plot_config.config["style"]["figsize"] = args.figsize
        
        # Apply the configuration to matplotlib - only once!
        # This will set the global style parameters
        plot_config.apply_style()

    # Read data from input file
    file_result, file_settings = read_data(
        args.input,
        delimiter=args.delimiter if hasattr(args, 'delimiter') else None,
        has_header=args.header if hasattr(args, 'header') else False,
        # Simplified: pass header_line_idx directly - data_io will handle it
        header_line_idx=args.header_line if hasattr(args, 'header_line') else None,
        comment_header_idx=args.comment_header if hasattr(args, 'comment_header') else None
    )

    # Process and plot data
    if isinstance(file_result, plt.Figure):
        # Handle pickled figure case
        fig = file_result
        
        # Check if we need to replot (any plotting parameters provided)
        # Simplified check for replotting
        # Ensure that args.ycols and args.xcol are checked for their default values properly
        is_ycols_default = not hasattr(args, 'ycols') or args.ycols == ['1']
        is_xcol_default = not hasattr(args, 'xcol') or args.xcol == '0'

        need_replot = not (is_ycols_default and is_xcol_default) or \
                      (hasattr(args, 'labels') and args.labels) or \
                      (hasattr(args, 'linestyles') and args.linestyles) or \
                      (hasattr(args, 'markers') and args.markers) or \
                      (hasattr(args, 'colors') and args.colors) or \
                      (hasattr(args, 'style') and args.style != 'default') or \
                      (hasattr(args, 'use_latex') and args.use_latex) or \
                      (hasattr(args, 'scienceplot') and args.scienceplot != 'none')
                      
        if need_replot:
            print("Input is a pickled figure, but plotting parameters were provided. Extracting data and re-plotting.")
            extracted_data, extracted_settings = extract_plot_data_and_settings(fig)
            
            if extracted_data is not None:
                data = extracted_data
                
                # Merge settings from extracted_settings with CLI args (CLI args override)
                for key, value in extracted_settings.items():
                    # Only set if CLI arg wasn't explicitly provided or is default
                    current_arg_value = getattr(args, key, None)
                    is_cli_default = False
                    if key in ['ycols']:
                        is_cli_default = (current_arg_value == ['1'])
                    elif key in ['xcol']:
                        is_cli_default = (current_arg_value == '0')
                    elif key in ['style']:
                        is_cli_default = (current_arg_value == 'default')
                    elif key in ['use_latex', 'scienceplot']: # Handle boolean flags or string defaults
                        # For these, if the arg exists and is not its default false/none, then CLI should win.
                        # If it's the default, then the extracted setting can apply.
                        is_cli_default = (current_arg_value is False or current_arg_value is None or current_arg_value == 'none')
                    else:
                        is_cli_default = (current_arg_value is None)

                    if not hasattr(args, key) or is_cli_default:
                        setattr(args, key, value)
                        effective_args[key] = value # Update effective_args with merged values
                
                # Resolve column indices based on final args and extracted header
                resolved_xcol, resolved_ycols = resolve_columns(
                    args.xcol, args.ycols, extracted_settings.get('columns_map')
                )
                effective_args['xcol'] = resolved_xcol
                effective_args['ycols'] = resolved_ycols

                # Set default labels from header if available and not overridden by CLI
                if 'header' in extracted_settings and not (hasattr(args, 'labels') and args.labels):
                    default_labels = []
                    for y_idx in resolved_ycols:
                        try:
                            default_labels.append(extracted_settings['header'][y_idx])
                        except IndexError:
                            default_labels.append(f"Column {y_idx}")
                    effective_args['labels'] = default_labels
                
                # Create new plot with PlotConfig
                fig = create_plot(data, argparse.Namespace(**effective_args), 
                                  existing_figure=fig, plot_config=plot_config)
            else:
                print("Could not extract data from pickled figure. Displaying as-is.")
        else:
            print("Input is a pickled figure. Displaying as-is.")

    else:
        # Handle numerical data (from text or JSON)
        data = file_result
        
        # If using JSON, merge settings from file (CLI args still override)
        if args.input.endswith('.json'):
            for key, value in file_settings.items():
                current_arg_value = getattr(args, key, None)
                is_cli_default = False
                if key in ['ycols']:
                    is_cli_default = (current_arg_value == ['1'])
                elif key in ['xcol']:
                    is_cli_default = (current_arg_value == '0')
                elif key in ['style']:
                    is_cli_default = (current_arg_value == 'default')
                elif key in ['use_latex', 'scienceplot']:
                    is_cli_default = (current_arg_value is False or current_arg_value is None or current_arg_value == 'none')
                elif key in ['labels', 'markers', 'linestyles', 'colors']:
                    is_cli_default = (current_arg_value is None or 
                                      (isinstance(current_arg_value, (list, tuple)) and len(current_arg_value) == 0))
                else:
                    is_cli_default = (current_arg_value is None or (isinstance(current_arg_value, str) and current_arg_value == '')) # Added for general strings

                if not hasattr(args, key) or is_cli_default:
                    setattr(args, key, value)
                    effective_args[key] = value # Update effective_args with merged values
        
        # Resolve column indices based on final args and file header
        resolved_xcol, resolved_ycols = resolve_columns(
            args.xcol, args.ycols, file_settings.get('columns_map')
        )
        effective_args['xcol'] = resolved_xcol
        effective_args['ycols'] = resolved_ycols

        # Set default labels from header if available and not overridden by CLI
        if 'header' in file_settings and not (hasattr(args, 'labels') and args.labels):
            default_labels = []
            for y_idx in resolved_ycols:
                try:
                    default_labels.append(file_settings['header'][y_idx])
                except IndexError:
                    default_labels.append(f"Column {y_idx}")
            effective_args['labels'] = default_labels
        
        # Create plot with PlotConfig
        fig = create_plot(data, argparse.Namespace(**effective_args), plot_config=plot_config)
    
    # --- Save outputs if requested ---
    if fig:
        if hasattr(args, 'save_json') and args.save_json:
            # Prepare settings for JSON save
            # This part needs careful handling of `data` vs `extracted_data`
            data_to_save = data if data is not None else []
            
            settings_to_save = {k: v for k, v in effective_args.items()
                               if v is not None and k not in ['input', 'save_json', 'save_figure_pickle', 'save_image', 'image_format', 'func', 'project', 'config', 'save_settings', 'settings']}
            
            # Add header info if available
            if 'header' in file_settings:
                settings_to_save['header'] = file_settings['header']
            elif 'header' in extracted_settings: # For re-plotted pickled figures
                settings_to_save['header'] = extracted_settings['header']
            
            # Save data and settings
            save_plot_to_json(data_to_save, settings_to_save, args.save_json)

        if hasattr(args, 'save_figure_pickle') and args.save_figure_pickle:
            save_pickled_figure(fig, args.save_figure_pickle)
            print("WARNING: Saving Matplotlib figures as pickle files is NOT recommended for long-term storage or portability.")

        if hasattr(args, 'save_image') and args.save_image:
            format_arg = args.image_format if hasattr(args, 'image_format') else 'png'
            save_figure_image(fig, args.save_image, format=format_arg)

        # In the saving section, add:
        if hasattr(args, 'save_settings') and args.save_settings:
            # Prepare ONLY styling settings for JSON save (no data references)
            styling_settings = {
                # Visual styling
                'labels': effective_args.get('labels'),
                'colors': effective_args.get('colors'),
                'markers': effective_args.get('markers'),
                'linestyles': effective_args.get('linestyles'),
                'markersize': effective_args.get('markersize'),
                
                # Axes styling
                'xlabel': effective_args.get('xlabel'),
                'ylabel': effective_args.get('ylabel'),
                'xscale': effective_args.get('xscale'),
                'yscale': effective_args.get('yscale'),
                'xlim': effective_args.get('xlim'),
                'ylim': effective_args.get('ylim'),
                'legend': effective_args.get('legend'),
                'figsize': effective_args.get('figsize'),
                
                # Plot style and scientific rendering options
                'style': effective_args.get('style'),
                'scienceplot': effective_args.get('scienceplot'),
                'use_latex': effective_args.get('use_latex'),
                
                # Data source options (only relevant if they affect styling, e.g., comment_header defines a header)
                'delimiter': effective_args.get('delimiter'), # Include if it affects how subsequent plots are interpreted
                'has_header': effective_args.get('has_header'),
                'header_line': effective_args.get('header_line'),
                'comment_header': effective_args.get('comment_header')
            }
            
            # Remove None values and empty lists/tuples for cleaner JSON
            cleaned_settings = {}
            for k, v in styling_settings.items():
                if v is not None:
                    if isinstance(v, (list, tuple)) and len(v) == 0:
                        continue # Skip empty lists/tuples
                    if isinstance(v, str) and v == '':
                        continue # Skip empty strings
                    # Special handling for default values that shouldn't be saved if they truly are defaults
                    if k == 'style' and v == 'default':
                        continue
                    if k == 'scienceplot' and v == 'none':
                        continue
                    if k == 'use_latex' and v is False:
                        continue
                    if k == 'legend' and v == 'on': # Default to on, so no need to save unless explicitly 'off'
                        continue
                    
                    cleaned_settings[k] = v
            
            # Save styling settings
            with open(args.save_settings, 'w') as f:
                json.dump(cleaned_settings, f, indent=2)
            print(f"Saved styling settings to {args.save_settings}")

        plt.show()  # Display the plot interactively
    else:
        print("No figure was generated to display or save.")

def plot_multi_file_cli(args):
    """Handles the multi-file plotting logic from a config."""
    try:
        # Create PlotConfig based on project if specified
        plot_config = None
        if hasattr(args, 'project') and args.project:
            plot_config = PlotConfig(project_name=args.project)
            # Apply project styling to matplotlib
            plot_config.apply_style()
            print(f"Applied project configuration: {args.project}")
            
        # Load the multi-plot configuration
        config = load_plot_config(args.config_file)
        
        # Make sure plot_settings exists
        if 'plot_settings' not in config:
            config['plot_settings'] = {}
            
        # If we have a project config, merge its settings into the plot config
        if plot_config:
            # Add legend_params if not present
            if 'legend_params' not in config['plot_settings']:
                config['plot_settings']['legend_params'] = plot_config.config.get('legend_params', {'frameon': False})
                
            # Update style handling
            if hasattr(args, 'use_latex'):
                config['plot_settings']['use_latex'] = args.use_latex
                
            if hasattr(args, 'scienceplot') and args.scienceplot != 'none':
                config['plot_settings']['scienceplot_style'] = args.scienceplot
    
        # Ensure multi-plot CLI overrides are processed correctly
        if hasattr(args, 'comment_header') and args.comment_header is not None:
            # Apply comment header to all datasets
            for dataset in config['datasets']:
                dataset['comment_header_idx'] = args.comment_header
                # Remove has_header if we're using comment_header
                if 'has_header' in dataset:
                    del dataset['has_header']

        if hasattr(args, 'has_header') and args.has_header:
            # Apply has_header to all datasets
            for dataset in config['datasets']:
                dataset['has_header'] = True
                # Only set if comment_header_idx isn't already set
                if 'comment_header_idx' not in dataset:
                    dataset['header_line_idx'] = dataset.get('header_line_idx', 0)
        
        # Apply CLI overrides for data loading and column selection
        if hasattr(args, 'comment_header') and args.comment_header is not None:
            # Apply comment header to all datasets
            for dataset in config['datasets']:
                dataset['comment_header_idx'] = args.comment_header
                
        if hasattr(args, 'has_header') and args.has_header:
            # Apply has_header to all datasets
            for dataset in config['datasets']:
                dataset['has_header'] = True
                
        # Apply CLI overrides to config plot settings
        # These take precedence over both the config file and project settings
        if hasattr(args, 'xscale') and args.xscale:
            config['plot_settings']['xscale'] = args.xscale
            
        if hasattr(args, 'yscale') and args.yscale:
            config['plot_settings']['yscale'] = args.yscale
            
        if hasattr(args, 'legend') and args.legend:
            config['plot_settings']['legend'] = args.legend
            
        if hasattr(args, 'use_latex') and args.use_latex:
            config['plot_settings']['use_latex'] = True
        
        # Handle save options - override or add to config
        if hasattr(args, 'save_image') and args.save_image:
            # Extract format from extension if possible
            filename, extension = os.path.splitext(args.save_image)
            if extension and extension.lower() in ['.png', '.pdf', '.svg', '.jpg', '.tiff']:
                config['plot_settings']['image_format'] = extension[1:].lower()
            config['plot_settings']['save_image'] = args.save_image
            
        if hasattr(args, 'image_format') and args.image_format:
            config['plot_settings']['image_format'] = args.image_format
            
        if hasattr(args, 'save_figure_pickle') and args.save_figure_pickle:
            config['plot_settings']['save_pkl'] = args.save_figure_pickle
            
        if hasattr(args, 'save_json') and args.save_json:
            config['plot_settings']['save_json'] = args.save_json
            
        if hasattr(args, 'save_settings') and args.save_settings:
            config['plot_settings']['save_settings'] = args.save_settings

        # Handle the new multi-format image saving options
        if hasattr(args, 'save_image') and args.save_image:
            # If CLI specifies save_image, use that for save_img_name
            config['plot_settings']['save_img_name'] = args.save_image
            
            # If image_format is also specified, use that as the only format
            if hasattr(args, 'image_format') and args.image_format:
                config['plot_settings']['save_img_fmts'] = [args.image_format]
            # Otherwise default to PNG
            else:
                # Try to get format from extension, fallback to png
                _, ext = os.path.splitext(args.save_image)
                if ext.lower() in ['.png', '.pdf', '.svg', '.jpg', '.tiff']:
                    config['plot_settings']['save_img_fmts'] = [ext[1:].lower()]
                else:
                    config['plot_settings']['save_img_fmts'] = ['png']
        
        if hasattr(args, 'column_format') and args.column_format:
            config['plot_settings']['use_column_format'] = True

        # Create the plot, requesting the datasets data as well
        fig, datasets_data = plot_from_config(config,return_datasets=True)
        
        # --- Handle any post-processing or saving that wasn't done in plot_from_config ---
        # These would be for saving options that were passed from CLI but not in the core plot_from_config
        
        if fig:
            plt.show()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error in multi-plot operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during multi-plot: {e}")
        raise


def load_and_display_multiplot(args):
    """Loads and displays a plot from a saved JSON file."""
    try:
        # Load JSON file
        with open(args.json_file, 'r') as f:
            config = json.load(f)
        
        # Add CLI save settings if provided (don't use ones from loaded JSON)
        if hasattr(args, 'save_image') and args.save_image:
            config['plot_settings']['save_img_name'] = args.save_image
            if hasattr(args, 'image_format'):
                config['plot_settings']['save_img_fmts'] = [args.image_format]
        
        # Create plot
        fig = plot_from_config(config)
        plt.show()
        
    except Exception as e:
        print(f"Error loading plot: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='pyPlots: A versatile Python library for post-processing and plotting data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    plot_parser = subparsers.add_parser(
        'plot',
        help='Plot data from a single input file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    plot_parser.add_argument('--input', required=True,
                             help='Input file to plot (txt, json, or pkl)')
    plot_parser.add_argument('--project', help='Name of a project configuration to use')
    plot_parser.add_argument('--config', help='Path to a specific configuration file')
    plot_parser.add_argument('--save-json', help='Save plot data and settings to JSON file')
    plot_parser.add_argument('--save-figure-pickle', help='Save matplotlib figure to a pickle file (NOT RECOMMENDED)')
    plot_parser.add_argument('--save-image', help='Save plot as an image file (e.g., .png, .pdf, .svg)')
    plot_parser.add_argument('--image-format', default='png',
                             choices=['png', 'pdf', 'svg', 'jpg', 'tiff'],
                             help='Format for saved image (if --save-image is used)')
    
    plot_parser.add_argument("--use-latex", action="store_true", help="Enable LaTeX rendering")
    plot_parser.add_argument("--scienceplot", type=str, choices=["ieee", "nature", "science", "none"], 
                             help="SciencePlot style to use (or 'none' to disable)")

    plot_parser.add_argument('--xcol', type=str, default='0', help='Column for x-axis (0-indexed or header name).')
    plot_parser.add_argument('--ycols', type=str, nargs='+', default=['1'], help='Column(s) for y-axis (0-indexed or header name).')
    
    plot_parser.add_argument('--xlabel', default='', help='X-axis label')
    plot_parser.add_argument('--ylabel', default='', help='Y-axis label')
    plot_parser.add_argument('--xlim', type=float, nargs=2, help='X-axis limits [min max]')
    plot_parser.add_argument('--ylim', type=float, nargs=2, help='Y-axis limits [min max]')
    plot_parser.add_argument('--labels', nargs='+', help='Labels for y columns (one for each --ycols)')
    plot_parser.add_argument('--legend', default='on', help='Legend on/off (values: "on" or "off")')
    plot_parser.add_argument('--xscale', choices=['linear', 'log', 'symlog', 'logit'], default='linear', help='Scale for x-axis')
    plot_parser.add_argument('--yscale', choices=['linear', 'log', 'symlog', 'logit'], default='linear', help='Scale for y-axis')
    plot_parser.add_argument('--markers', nargs='+', help='Markers for each curve (o, s, ^, v, etc). Single value applies to all.')
    plot_parser.add_argument('--linestyles', nargs='+', help='Line styles for each curve (-, --, :, etc). Single value applies to all.')
    plot_parser.add_argument('--colors', nargs='+', help='Colors for each curve. Can be names ("red") or hex values ("#FF0000"). Single value applies to all.')
    plot_parser.add_argument('--markersize', type=float, help='Size of markers when using marker styles.')

    if SCIENCEPLOTS_AVAILABLE:
        plot_parser.add_argument('--style', default='default',
                                 choices=['default', 'science', 'ieee'],
                                 help='Apply a specific plot style (e.g., "science"). Requires scienceplots package.')
    else:
        plot_parser.add_argument('--style', default='default',
                                 help='Apply a specific plot style. "science" style requires scienceplots package. (scienceplots not found)',
                                 choices=['default'])
    
    plot_parser.add_argument('--delimiter', '-d', type=str, default=None,
                             help='Delimiter for text files (e.g., "," for CSV, "\\t" for tab-separated). '
                                  'If not specified or empty, splits by any whitespace (default).')
    plot_parser.add_argument('--header', action='store_true',
                             help='Indicate that the data file has a header row. '
                                  'When enabled, column names can be used instead of indices.')

    plot_parser.add_argument('--header-line', type=int, default=None,
                             help='0-indexed line number of the header row. '
                                  'If specified, this line will be treated as a header '
                                  'regardless of the --header flag.')

    plot_parser.add_argument('--comment-header', type=int, 
                             help='Use a comment line (starting with #) as header. '
                                  'Specify the 0-indexed position among comment lines. '
                                  'This has higher priority than --header-line.')
    
    plot_parser.add_argument('--figsize', type=float, nargs=2, 
                             help='Figure size in inches [width height] (e.g., 10 6)')
    
    plot_parser.add_argument('--save-settings', 
                             help='Save plot styling settings (without data references) to JSON file')
    plot_parser.add_argument('--settings', 
                             help='Load plot styling settings from JSON file')
    
    plot_parser.set_defaults(func=plot_single_file_cli)

    multiplot_parser = subparsers.add_parser(
        'multiplot',
        help='Plot data from multiple files on the same axes using a JSON configuration file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    multiplot_parser.add_argument('config_file',
                                  help='Path to the JSON configuration file specifying datasets and common plot settings.')
    multiplot_parser.add_argument('--project', 
                                  help='Name of a project configuration to use as base settings')
    multiplot_parser.add_argument('--save-image', help='Override the save path in the config')
    multiplot_parser.add_argument('--image-format', choices=['png', 'pdf', 'svg', 'jpg', 'tiff'], 
                                  help='Override the image format in the config')
    multiplot_parser.add_argument('--image-formats', nargs='+', 
                                  choices=['png', 'pdf', 'svg', 'jpg', 'tiff'],
                                  help='Save image in multiple formats')
    multiplot_parser.add_argument('--xscale', choices=['linear', 'log', 'symlog', 'logit'], 
                                  help='Override x-axis scale')
    multiplot_parser.add_argument('--yscale', choices=['linear', 'log', 'symlog', 'logit'], 
                                  help='Override y-axis scale')
    multiplot_parser.add_argument('--legend', choices=['on', 'off'], 
                                  help='Override legend display')
    multiplot_parser.add_argument("--use-latex", action="store_true", 
                                  help="Override to enable LaTeX rendering")
    multiplot_parser.add_argument('--save-json', 
                                  help='Save combined plot data and settings to JSON file')
    multiplot_parser.add_argument('--save-settings', 
                                  help='Save only styling settings to JSON file')
    multiplot_parser.add_argument('--save-figure-pickle', 
                                  help='Save matplotlib figure to a pickle file')
    multiplot_parser.add_argument('--comment-header', type=int, 
                                  help='Override comment header line number for all datasets')
    multiplot_parser.add_argument('--has-header', action='store_true',
                                  help='Override to indicate all data files have headers')
    multiplot_parser.add_argument('--column-format', action='store_true',
                                  help='Save JSON in column-oriented format for better readability')
    multiplot_parser.set_defaults(func=plot_multi_file_cli)

    load_parser = subparsers.add_parser(
        'loadplot', 
        help='Load and display a saved multi-plot JSON file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    load_parser.add_argument('json_file', help='Path to the JSON file to load')
    load_parser.add_argument('--save-image', help='Save the loaded plot as an image')
    load_parser.add_argument('--image-format', choices=['png', 'pdf', 'svg', 'jpg', 'tiff'],
                             help='Format for saved image (if --save-image is used)')
    load_parser.set_defaults(func=load_and_display_multiplot)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            print(f"An error occurred: {e}")
            # raise # Uncomment for full traceback during development: raise
    else:
        parser.print_help()

if __name__ == '__main__':
    main()