# pyplots/data_io.py

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- Assuming load_figure_from_pickle is defined here or properly imported ---
def load_figure_from_pickle(filepath):
    """Loads a Matplotlib figure from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            fig = pickle.load(f)
        return fig
    except Exception as e:
        raise IOError(f"Error loading pickled figure from {filepath}: {e}")
# --- End assumption ---


def read_data(filename, delimiter=None, has_header=False, header_line_idx=None, comment_header_idx=None):
    """
    Read data from a file.
    
    Args:
        filename (str): The path to the data file.
        delimiter (str, optional): The delimiter to use for text files.
        has_header (bool, optional): Whether the data file has a header row.
        header_line_idx (int, optional): The line number of the header row (0-indexed, after skipping comments).
        comment_header_idx (int, optional): Line number of a comment line to use as header (0-indexed, counting only comment lines).
        
    Returns:
        tuple: (data, settings) where data is a numpy array or Matplotlib figure, and settings is a dict with any metadata.
    """
    filepath = Path(filename)
    plot_settings = {}

    if filepath.suffix == '.pkl':
        return load_figure_from_pickle(filepath), None

    elif filepath.suffix == '.json':
        try:
            with open(filepath, 'r') as f:
                content = json.load(f)
            data = np.array(content.get('data', []))
            plot_settings = content.get('plot_settings', {})
            return data, plot_settings
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not decode JSON from {filepath}: {e}")
        except Exception as e:
            raise IOError(f"Error reading JSON file {filepath}: {e}")

    else:  # Assume text file (csv, txt, dat, etc.)
        # Read the whole file as lines first to handle comments and headers
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Identify comment lines
            comment_lines = []
            data_lines = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                if line.startswith('#'):
                    comment_lines.append((i, line))
                else:
                    data_lines.append((i, line))
                    
            # Initialize header variables
            header = None
            columns_map = None
            plot_settings = {}
            
            # Extract header from comments if requested (highest priority)
            if comment_header_idx is not None and comment_header_idx < len(comment_lines):
                _, comment_line = comment_lines[comment_header_idx]
                # Remove comment marker and split by delimiter
                header_line = comment_line.lstrip('#').strip()
                if delimiter:
                    header = [h.strip() for h in header_line.split(delimiter)]
                else:
                    header = [h.strip() for h in header_line.split()]
                    
                # Create columns_map
                columns_map = {col: i for i, col in enumerate(header)}
                plot_settings = {'header': header, 'columns_map': columns_map}
                
                # We've already processed the header from comments
                has_header = False
                header_line_idx = None
            
            # Process data lines
            data_list = []
            data_relevant_line_idx = 0

            for line_num, line in data_lines:
                stripped_line = line.strip()
                
                # Handle regular header line if requested
                # If header_line_idx is provided, treat it as a header regardless of has_header
                if header_line_idx is not None and data_relevant_line_idx == header_line_idx:
                    if delimiter:
                        header_elements = [s.strip().strip('"') for s in stripped_line.split(delimiter) if s.strip()]
                    else:
                        header_elements = [s.strip().strip('"') for s in stripped_line.split() if s.strip()]
                    
                    header = header_elements
                    columns_map = {name: i for i, name in enumerate(header)}
                    plot_settings['header'] = header
                    plot_settings['columns_map'] = columns_map
                    data_relevant_line_idx += 1
                    continue
                
                # Process data lines
                current_row_data = []
                
                # Improved delimiter handling
                if delimiter is None:
                    # Use any whitespace (space or tab) as delimiter
                    split_elements = stripped_line.split()
                else:
                    # Use specified delimiter (comma, tab, etc.)
                    split_elements = stripped_line.split(delimiter)

                for s in split_elements:
                    s_stripped = s.strip()
                    if not s_stripped:
                        continue
                    try:
                        current_row_data.append(float(s_stripped))
                    except ValueError:
                        print(f"Warning: Skipping non-numeric data '{s_stripped}' on line {line_num + 1} in {filepath}.")
                        pass

                if current_row_data:
                    data_list.append(current_row_data)
                else:
                    print(f"Warning: Entire line {line_num + 1} in {filepath} contained no valid numeric data and was skipped.")

                data_relevant_line_idx += 1

            if not data_list:
                raise ValueError(f"No valid data found in text file {filepath}.")

            data_array = np.array(data_list)
            if data_array.dtype == object:
                raise ValueError(f"Data rows have inconsistent column counts or contain non-numeric data that prevents uniform array creation in {filepath}. Consider using pandas for complex files.")

            return data_array, plot_settings

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise IOError(f"Error reading text file {filepath}: {e}")