import pytest
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from unittest.mock import MagicMock, patch

# Import functions directly from your pyplots submodules
from pyplots.data_io import read_data, load_figure_from_pickle
from pyplots.plotting import create_plot
from pyplots.multi_plot import load_plot_config, plot_from_config
from pyplots.plot_saver import save_pickled_figure
from pyplots.cli import resolve_columns # Import the helper function
from pyplots.cli import plot_single_file_cli # Add this import


# --- Fixtures for Paths & Setup ---
@pytest.fixture
def data_dir():
    # This path assumes 'tests' is a direct subdirectory of your project root.
    # Adjust if your test_cli_and_core.py is nested deeper.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_datasets')


@pytest.fixture
def simple_txt_file(data_dir):
    # Make sure this file exists in test_datasets and has 7 data rows
    return os.path.join(data_dir, 'simple_data.txt')

# NEW FIXTURES FOR NEW DATA FILES
@pytest.fixture
def data_with_header_tab_file(data_dir):
    return os.path.join(data_dir, 'data_with_header_tab.tsv')

@pytest.fixture
def data_with_header_comma_file(data_dir):
    return os.path.join(data_dir, 'data_with_header_comma.csv')

@pytest.fixture
def data_with_comments_header_file(data_dir):
    return os.path.join(data_dir, 'data_with_comments_header.txt')


@pytest.fixture
def minimal_multi_plot_config_file(data_dir):
    return os.path.join(data_dir, 'minimal_multi_plot_config.json')

@pytest.fixture
def dummy_json_data_file(tmp_path):
    json_path = tmp_path / "data.json"
    content = {"data": [[1,1], [2,2]], "plot_settings": {"title": "JSON Data"}}
    with open(json_path, 'w') as f:
        json.dump(content, f)
    return json_path

@pytest.fixture
def dummy_pkl_fig_file(tmp_path):
    pkl_path = tmp_path / "fig.pkl"
    fig, ax = plt.subplots()
    ax.plot([0,1], [0,1])
    save_pickled_figure(fig, str(pkl_path))
    plt.close(fig)
    return pkl_path

@pytest.fixture
def temp_settings_file(tmp_path):
    return str(tmp_path / "test_settings.json")

@pytest.fixture
def temp_settings_override(tmp_path):
    return str(tmp_path / "override_settings.json")

@pytest.fixture
def temp_plot_file(tmp_path):
    return str(tmp_path / "test_plot.png")

@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close('all')

@pytest.fixture(autouse=True)
def mock_plt_show(monkeypatch):
    """Prevent plt.show() from opening windows during tests."""
    monkeypatch.setattr(plt, 'show', lambda: None)


# --- MockArgs for plotting.create_plot and CLI simulation ---
class MockArgs:
    def __init__(self, **kwargs):
        # Store all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Set defaults for common attributes if not provided
        if 'xcol' not in kwargs: self.xcol = '0'
        if 'ycols' not in kwargs: self.ycols = ['1']
        if 'labels' not in kwargs: self.labels = []
        if 'legend' not in kwargs: self.legend = 'on'
        if 'xlabel' not in kwargs: self.xlabel = ''
        if 'ylabel' not in kwargs: self.ylabel = ''
        if 'xscale' not in kwargs: self.xscale = 'linear'
        if 'yscale' not in kwargs: self.yscale = 'linear'
        if 'linestyles' not in kwargs: self.linestyles = []
        if 'markers' not in kwargs: self.markers = []
        if 'colors' not in kwargs: self.colors = []
        if 'style' not in kwargs: self.style = 'default'
        if 'delimiter' not in kwargs: self.delimiter = None
        if 'header' not in kwargs: self.header = False
        if 'header_line' not in kwargs: self.header_line = 0

    def __getattr__(self, name):
        # This is called only for attributes that don't exist
        # Return appropriate defaults for known attributes
        if name in ['labels', 'linestyles', 'markers', 'colors']: return []
        if name in ['markersize', 'xlim', 'ylim', 'figsize']: return None
        if name in ['xlabel', 'ylabel']: return ''
        if name in ['xscale', 'yscale']: return 'linear'
        if name == 'legend': return 'on'
        if name == 'style': return 'default'
        if name == 'delimiter': return None
        if name == 'header': return False
        if name == 'header_line': return 0
        if name in ['save_settings', 'settings', 'save_image', 'save_json', 'save_figure_pickle']: return None
        
        # Raise attribute error for unknown attributes
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# --- Data_IO Tests ---
def test_read_data_txt_success(simple_txt_file):
    data, settings = read_data(simple_txt_file)
    assert isinstance(data, np.ndarray)
    # Corrected expected shape: simple_data.txt has 7 lines of data
    assert data.shape == (7, 2)
    assert np.allclose(data[1, 1], 1.0)
    assert settings == {}

def test_read_data_json_success(dummy_json_data_file):
    data, settings = read_data(dummy_json_data_file)
    assert isinstance(data, np.ndarray)
    assert np.array_equal(data, [[1,1], [2,2]])
    assert settings == {"title": "JSON Data"}

def test_read_data_pkl_success(dummy_pkl_fig_file):
    fig, settings = read_data(dummy_pkl_fig_file)
    assert isinstance(fig, plt.Figure)
    assert settings is None

def test_read_data_non_existent_file():
    with pytest.raises(FileNotFoundError):
        read_data("non_existent.txt")

def test_read_data_empty_txt(tmp_path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("# Only comments")
        # Expect an OSError, and check its message for the original ValueError's text
        # The 'match' string should be a substring of the actual error message.
        with pytest.raises(OSError, match="No valid data found"): # This is the crucial part
            read_data(str(empty_file))

# NEW DATA_IO TESTS (Updated expectations based on data_io.py changes)
def test_read_data_with_tab_delimiter_no_header(data_with_header_tab_file):
    # The 'Comment' column will be skipped because it's non-numeric
    data, settings = read_data(data_with_header_tab_file, delimiter='\t', has_header=False)
    assert isinstance(data, np.ndarray)
    # Expect 4 rows, 3 columns (Time, Value1, Value2) as 'Comment' is skipped
    assert data.shape == (4, 3) 
    assert np.allclose(data[0, 0], 0.0)
    assert np.allclose(data[0, 1], 10.0)
    assert settings == {} # No header info expected when has_header=False

def test_read_data_with_header_tab_delimiter(data_with_header_tab_file):
    data, settings = read_data(data_with_header_tab_file, delimiter='\t', has_header=True)
    assert isinstance(data, np.ndarray)
    # Expect 4 data rows, 3 numerical columns ('Comment' column is ignored in data array)
    assert data.shape == (4, 3)
    assert np.allclose(data[0, 0], 0.0)
    assert np.allclose(data[0, 1], 10.0)
    # Header and columns_map should contain all headers including 'Comment'
    assert settings['header'] == ['Time', 'Value1', 'Value2', 'Comment']
    assert settings['columns_map'] == {'Time': 0, 'Value1': 1, 'Value2': 2, 'Comment': 3}


def test_read_data_with_header_comma_delimiter(data_with_header_comma_file):
    data, settings = read_data(data_with_header_comma_file, delimiter=',', has_header=True)
    assert isinstance(data, np.ndarray)
    assert data.shape == (3, 3) # 3 data rows, 3 numerical columns
    assert np.allclose(data[0, 0], 1.0)
    assert np.allclose(data[0, 1], 1.1)
    assert settings['header'] == ['Index', 'Measurement A', 'Measurement B']
    assert settings['columns_map'] == {'Index': 0, 'Measurement A': 1, 'Measurement B': 2}

def test_read_data_header_line_idx(data_with_comments_header_file):
    filename = data_with_comments_header_file # This is now correctly getting the path from your fixture

    # Change the delimiter here:
    data, settings = read_data(filename, delimiter=None, has_header=True, header_line_idx=0) # <-- Change '\t' to None

    assert isinstance(data, np.ndarray)
    assert data.shape == (2, 3)
    assert np.allclose(data[0, 0], 1.0)
    assert np.allclose(data[0, 1], 10.0) # Check these values based on your file
    assert settings['header'] == ['Time', 'Data1', 'Data2']
    assert settings['columns_map'] == {'Time': 0, 'Data1': 1, 'Data2': 2}


# --- Plotting Tests (Focus on create_plot and resolve_columns) ---
def test_create_plot_basic_functionality():
    data = np.array([[0, 0], [1, 1], [2, 4]])
    args = MockArgs(xcol=0, ycols=[1], labels=['Data Series 1'], xlabel='X-axis', ylabel='Y-axis')
    fig = create_plot(data, args)

    assert fig is not None
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert len(ax.lines) == 1
    assert ax.get_xlabel() == 'X-axis'
    assert ax.get_ylabel() == 'Y-axis'
    assert ax.legend_ is not None
    assert ax.legend_.get_texts()[0].get_text() == 'Data Series 1'

def test_create_plot_multiple_ycols_and_labels():
    data = np.array([[0, 0, 10], [1, 1, 11]])
    args = MockArgs(xcol=0, ycols=[1, 2], labels=['Curve1', 'Curve2'])
    fig = create_plot(data, args)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_lines()[0].get_label() == 'Curve1'
    assert ax.get_lines()[1].get_label() == 'Curve2'
    assert ax.get_legend() is not None

def test_create_plot_empty_data_raises_error():
    data = np.array([])
    args = MockArgs(xcol=0, ycols=[1])
    with pytest.raises(ValueError, match="No valid data provided for plotting"):
        create_plot(data, args)

# NEW: Test resolve_columns helper function
def test_resolve_columns_by_index_strings():
    columns_map = {'A': 0, 'B': 1, 'C': 2}
    xcol_arg = '0'
    ycols_args = ['1', '2']
    resolved_xcol, resolved_ycols = resolve_columns(xcol_arg, ycols_args, columns_map)
    assert resolved_xcol == 0
    assert resolved_ycols == [1, 2]

def test_resolve_columns_by_name():
    columns_map = {'Time': 0, 'Value1': 1, 'Value2': 2}
    xcol_arg = 'Time'
    ycols_args = ['Value1', 'Value2']
    resolved_xcol, resolved_ycols = resolve_columns(xcol_arg, ycols_args, columns_map)
    assert resolved_xcol == 0
    assert resolved_ycols == [1, 2]

def test_resolve_columns_mixed_index_and_name():
    columns_map = {'Time': 0, 'Value1': 1, 'Value2': 2}
    xcol_arg = '0' # Index as string
    ycols_args = ['Value1', '2'] # Name and index as string
    resolved_xcol, resolved_ycols = resolve_columns(xcol_arg, ycols_args, columns_map)
    assert resolved_xcol == 0
    assert resolved_ycols == [1, 2]

def test_resolve_columns_invalid_name_raises_error():
    columns_map = {'A': 0, 'B': 1}
    with pytest.raises(ValueError, match="Invalid x-column specified"):
        resolve_columns('NonExistent', ['B'], columns_map)
    with pytest.raises(ValueError, match="Invalid y-column specified"):
        resolve_columns('A', ['NonExistent'], columns_map)

def test_resolve_columns_no_header_only_index_strings():
    columns_map = None # No header
    xcol_arg = '0'
    ycols_args = ['1']
    resolved_xcol, resolved_ycols = resolve_columns(xcol_arg, ycols_args, columns_map)
    assert resolved_xcol == 0
    assert resolved_ycols == [1]

def test_resolve_columns_no_header_name_raises_error():
    columns_map = None
    with pytest.raises(ValueError, match="Data file has no header. Please specify x and y columns by integer index."):
        resolve_columns('Time', ['Value'], columns_map)

# --- Multi_Plot Tests ---
def test_load_plot_config_success(minimal_multi_plot_config_file):
    config = load_plot_config(minimal_multi_plot_config_file)
    assert isinstance(config, dict)
    assert "datasets" in config
    assert len(config["datasets"]) == 1

def test_load_plot_config_non_existent():
    with pytest.raises(FileNotFoundError):
        load_plot_config("non_existent_config.json")

def test_plot_from_config_success(minimal_multi_plot_config_file, tmp_path):
    config = load_plot_config(minimal_multi_plot_config_file)
    config['datasets'][0]['file'] = os.path.join(os.path.dirname(minimal_multi_plot_config_file), 'simple_data.txt')
    config['plot_settings']['save_png'] = str(tmp_path / "test_output.png")

    fig = plot_from_config(config)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "Minimal Test Plot"
    assert len(ax.get_lines()) == 1
    assert os.path.exists(tmp_path / "test_output.png")

def test_plot_from_config_no_valid_datasets(tmp_path):
    config = {
        "datasets": [
            {"file": str(tmp_path / "non_existent_data.txt"), "xcol": 0, "ycols": [1]}
        ],
        "plot_settings": {}
    }
    fig = plot_from_config(config)
    assert fig is None

# --- CLI Tests ---
def test_cli_save_load_override_settings(simple_txt_file, temp_settings_file, temp_settings_override, temp_plot_file):
    """Test saving, loading, and overriding settings through the CLI interface."""
    # STEP 1: Create plot with specific settings and save settings file
    args1 = MockArgs(
        input=simple_txt_file,
        xcol='0',
        ycols=['1'],
        labels=['Test Data'],
        colors=['red'],
        markers=['o'],
        linestyles=['-'],
        markersize=8,
        xlabel='X Value',
        ylabel='Y Value',
        figsize=[10, 6],
        save_settings=temp_settings_file
    )
    
    # Mock the plot_single_file_cli call (suppressing plt.show)
    with patch('matplotlib.pyplot.show'):
        plot_single_file_cli(args1)
    
    # STEP 2: Verify settings file exists and contains expected values
    assert os.path.exists(temp_settings_file), "Settings file wasn't created"
    
    with open(temp_settings_file, 'r') as f:
        saved_settings = json.load(f)
    
    # Check that styling parameters were saved
    assert saved_settings['labels'] == ['Test Data']
    assert saved_settings['colors'] == ['red']
    assert saved_settings['markers'] == ['o']
    assert saved_settings['markersize'] == 8
    assert saved_settings['xlabel'] == 'X Value'
    assert saved_settings['ylabel'] == 'Y Value'
    
    # Check that data reference parameters were NOT saved
    assert 'input' not in saved_settings
    assert 'xcol' not in saved_settings
    assert 'ycols' not in saved_settings
    
    # STEP 3: Load settings without overrides
    args2 = MockArgs(
        input=simple_txt_file,
        xcol='0',
        ycols=['1'],
        settings=temp_settings_file,
        save_image=temp_plot_file
    )
    
    with patch('matplotlib.pyplot.show'):
        plot_single_file_cli(args2)
    
    # Verify plot was created
    assert os.path.exists(temp_plot_file), "Plot image wasn't created"
    
    # STEP 4: Load settings WITH overrides
    args3 = MockArgs(
        input=simple_txt_file,
        xcol='0',
        ycols=['1'],
        settings=temp_settings_file,
        colors=['blue'],       # Override colors
        yscale='log',          # Override scale
        save_settings=temp_settings_override
    )
    
    with patch('matplotlib.pyplot.show'):
        plot_single_file_cli(args3)
    
    # STEP 5: Verify overridden settings file
    assert os.path.exists(temp_settings_override), "Override settings file wasn't created"
    
    with open(temp_settings_override, 'r') as f:
        override_settings = json.load(f)
    
    # Check that overrides were applied
    assert override_settings['colors'] == ['blue']   # Should be overridden
    assert override_settings['yscale'] == 'log'      # Should be overridden
    assert override_settings['markers'] == ['o']     # Should be preserved from original
    assert override_settings['xlabel'] == 'X Value'  # Should be preserved from original