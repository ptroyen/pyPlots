import os
import json
import pytest
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Check if scienceplots is available
try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False

from pyplots.config_module import PlotConfig, create_figure

# --- Fixtures for test configuration files ---
@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_datasets')

@pytest.fixture
def simple_data_file(data_dir):
    file_path = os.path.join(data_dir, 'simple_data.txt')
    if not os.path.exists(file_path):
        pytest.skip(f"Test file not found: {file_path}")
    return file_path

@pytest.fixture
def latex_config_file(data_dir):
    file_path = os.path.join(data_dir, 'latex_config.json')
    if not os.path.exists(file_path):
        pytest.skip(f"Test file not found: {file_path}")
    return file_path

@pytest.fixture
def science_config_ieee_file(data_dir):
    file_path = os.path.join(data_dir, 'science_config_ieee.json')
    if not os.path.exists(file_path):
        pytest.skip(f"Test file not found: {file_path}")
    return file_path

@pytest.fixture
def science_config_nature_file(data_dir):
    file_path = os.path.join(data_dir, 'science_config_nature.json')
    if not os.path.exists(file_path):
        pytest.skip(f"Test file not found: {file_path}")
    return file_path

# Basic PlotConfig tests remain unchanged
def test_plotconfig_default_initialization():
    """Test that PlotConfig initializes with correct default values."""
    config = PlotConfig()
    assert config.config["style"]["use_scienceplot"] == True
    assert config.config["style"]["scienceplot_style"] == "ieee"
    assert config.config["style"]["use_latex"] == False
    assert config.config["fonts"]["size"] == 10
    assert config.config["lines"]["width"] == 1.5

def test_plotconfig_load_config_file(tmp_path):
    """Test loading configuration from a file."""
    # Create a test config file
    config_file = tmp_path / "test_config.json"
    test_config = {
        "style": {
            "figure_dpi": 600,
            "use_latex": True
        },
        "fonts": {
            "size": 12
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    # Load the config
    config = PlotConfig(config_file=str(config_file))
    
    # Check that values were updated
    assert config.config["style"]["figure_dpi"] == 600
    assert config.config["style"]["use_latex"] == True
    assert config.config["fonts"]["size"] == 12
    # Check that other values remain at defaults
    assert config.config["style"]["use_scienceplot"] == True
    assert config.config["style"]["scienceplot_style"] == "ieee"

def test_create_figure():
    """Test the create_figure function."""
    # Basic test
    fig, config = create_figure()
    assert isinstance(fig, plt.Figure)
    assert isinstance(config, PlotConfig)
    # Check figure size
    assert fig.get_size_inches()[0] == 8
    assert fig.get_size_inches()[1] == 6
    plt.close(fig)
    
    # Test with custom size
    fig, config = create_figure(figsize=(10, 4))
    assert fig.get_size_inches()[0] == 10
    assert fig.get_size_inches()[1] == 4
    plt.close(fig)

def test_get_line_props():
    """Test the get_line_props method."""
    config = PlotConfig()
    props = config.get_line_props()
    assert props["linewidth"] == 1.5
    assert props["linestyle"] == "-"
    assert props["marker"] == "o"
    assert props["markersize"] == 5
    assert props["markevery"] == 1

def test_get_figsize():
    """Test the get_figsize method."""
    config = PlotConfig()
    figsize = config.get_figsize()
    assert isinstance(figsize, tuple)
    assert len(figsize) == 2
    assert figsize[0] == 8  # Default width
    assert figsize[1] == 6  # Default height

def test_figsize_from_config_file(tmp_path):
    """Test loading figure size from a config file."""
    # Create a test config file with custom figsize
    config_file = tmp_path / "figsize_config.json"
    test_config = {
        "style": {
            "figsize": [12, 8]  # Custom figure size
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(test_config, f)
    
    # Load the config
    config = PlotConfig(config_file=str(config_file))
    
    # Check that figure size was loaded correctly
    figsize = config.get_figsize()
    assert figsize == (12, 8)
    
    # Test that it's applied when creating a figure
    fig, _ = create_figure(config_file=str(config_file))
    assert fig.get_size_inches()[0] == 12
    assert fig.get_size_inches()[1] == 8
    plt.close(fig)

def test_figsize_override_in_create_figure():
    """Test that figsize parameter overrides config values in create_figure."""
    # Create a config with default values
    config = PlotConfig()
    
    # Default figsize should be (8, 6)
    fig1, _ = create_figure()
    assert fig1.get_size_inches()[0] == 8
    assert fig1.get_size_inches()[1] == 6
    plt.close(fig1)
    
    # Override with custom figsize
    fig2, cfg2 = create_figure(figsize=(14, 10))
    assert fig2.get_size_inches()[0] == 14
    assert fig2.get_size_inches()[1] == 10
    # Config object should be updated too
    assert cfg2.config["style"]["figsize"] == [14, 10]
    plt.close(fig2)
    
    # Create a config file with one figsize, then override it
    tmp_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    config_file = tmp_dir / "test_override.json"
    with open(config_file, 'w') as f:
        json.dump({"style": {"figsize": [9, 7]}}, f)
    
    # Load from file but override figsize
    fig3, cfg3 = create_figure(config_file=str(config_file), figsize=(16, 9))
    assert fig3.get_size_inches()[0] == 16
    assert fig3.get_size_inches()[1] == 9
    # Config should reflect the override
    assert cfg3.config["style"]["figsize"] == [16, 9]
    plt.close(fig3)
    
    # Clean up
    config_file.unlink()
    tmp_dir.rmdir()

def test_latex_config():
    """Test LaTeX configuration is properly applied."""
    # Save original rcParams to restore after test
    original_rcParams = plt.rcParams.copy()  # Make a full copy
    
    try:
        # Test LaTeX enabled
        plt.rcdefaults()  # Reset to defaults first
        config = PlotConfig()
        config.config["style"]["use_latex"] = True
        config.config["style"]["use_scienceplot"] = False  # Disable scienceplots to avoid interference
        config.apply_style()
        
        # Check rcParams were updated
        assert plt.rcParams['text.usetex'] == True
        # font.family can be either a string or a list containing 'serif'
        font_family = plt.rcParams['font.family']
        assert font_family == 'serif' or (isinstance(font_family, list) and 'serif' in font_family)
        
        # Test LaTeX disabled - reset matplotlib completely first
        plt.rcdefaults()  # Reset to defaults
        config = PlotConfig()
        config.config["style"]["use_latex"] = False
        config.config["style"]["use_scienceplot"] = False  # Disable scienceplots to avoid interference
        config.apply_style()
        
        # Check rcParams were updated
        assert plt.rcParams['text.usetex'] == False
    finally:
        # Restore original rcParams
        plt.rcParams.update(original_rcParams)

# Updated test using fixture for LaTeX config
def test_latex_config_from_file(latex_config_file):
    """Test LaTeX configuration is properly loaded from file and applied."""
    # Skip if fixture doesn't exist
    if not latex_config_file:
        pytest.skip("latex_config.json not found")
        
    # Save original rcParams to restore after test
    original_usetex = plt.rcParams.get('text.usetex', False)
    
    try:
        # Load config from file
        config = PlotConfig(config_file=latex_config_file)
        config.apply_style()
        
        # Verify LaTeX settings from file were applied
        assert plt.rcParams['text.usetex'] == True
        # font.family can be either a string or a list containing 'serif'
        font_family = plt.rcParams['font.family']
        assert font_family == 'serif' or (isinstance(font_family, list) and 'serif' in font_family)
    finally:
        # Restore original rcParams
        plt.rcParams['text.usetex'] = original_usetex

@pytest.mark.skipif(not SCIENCEPLOTS_AVAILABLE, 
                    reason="scienceplots package not installed")
def test_scienceplot_styles():
    """Test different SciencePlots styles are applied correctly."""
    # Test IEEE style
    config = PlotConfig()
    config.config["style"]["use_scienceplot"] = True
    config.config["style"]["scienceplot_style"] = "ieee"
    config.apply_style()
    
    # Create a simple plot to verify style application
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Since we can't easily verify the style was applied correctly,
    # just check it doesn't raise exceptions
    plt.close(fig)
    
    # Test Nature style
    config = PlotConfig()
    config.config["style"]["use_scienceplot"] = True
    config.config["style"]["scienceplot_style"] = "nature"
    config.apply_style()
    
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    plt.close(fig)

# Updated tests using fixtures for science config files
@pytest.mark.skipif(not SCIENCEPLOTS_AVAILABLE, 
                    reason="scienceplots package not installed")
def test_scienceplot_ieee_from_file(science_config_ieee_file):
    """Test IEEE SciencePlots style is properly loaded from file and applied."""
    # Skip if fixture doesn't exist
    if not science_config_ieee_file:
        pytest.skip("science_config_ieee.json not found")
        
    # Load config from file
    config = PlotConfig(config_file=science_config_ieee_file)
    config.apply_style()
    
    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3], **config.get_line_props())
    
    # Check that the plot was created without errors
    assert len(ax.get_lines()) == 1
    plt.close(fig)

@pytest.mark.skipif(not SCIENCEPLOTS_AVAILABLE, 
                    reason="scienceplots package not installed")
def test_scienceplot_nature_from_file(science_config_nature_file):
    """Test Nature SciencePlots style is properly loaded from file and applied."""
    # Skip if fixture doesn't exist
    if not science_config_nature_file:
        pytest.skip("science_config_nature.json not found")
        
    # Load config from file
    config = PlotConfig(config_file=science_config_nature_file)
    config.apply_style()
    
    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3], **config.get_line_props())
    
    # Check that the plot was created without errors
    assert len(ax.get_lines()) == 1
    plt.close(fig)

# Test integration with real data files
def test_integration_with_existing_files(simple_data_file, science_config_nature_file):
    """Test integration with plotting using existing test files."""
    # Skip if either fixture doesn't exist
    if not simple_data_file or not science_config_nature_file:
        pytest.skip("Required test files not found")
        
    # Load config from file
    config = PlotConfig(config_file=science_config_nature_file)
    config.apply_style()
    
    # Create a plot
    fig, ax = plt.subplots()
    
    # Read data from file
    data = np.loadtxt(simple_data_file)
    
    # Plot with config's line properties
    ax.plot(data[:, 0], data[:, 1], **config.get_line_props())
    ax.set_xlabel("X Value")
    ax.set_ylabel("Y Value")
    
    # Verify line properties were applied from the config file
    line = ax.get_lines()[0]
    
    # Verify the marker is what we expect from science_config_nature.json
    # (Adjust these assertions based on actual values in your config file)
    if SCIENCEPLOTS_AVAILABLE:
        assert line.get_marker() == '^'  # Assuming the config has "default": "^"
        assert line.get_linewidth() == 1.2  # Assuming the config has "width": 1.2
    
    plt.close(fig)