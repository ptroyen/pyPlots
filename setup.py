from setuptools import setup, find_packages

setup(
    name='pyPlots',
    version='0.1.0',
    author='Sagar Pokharel',
    author_email='ptroyen@gmai.com',
    description='A versatile Python library for post-processing and plotting data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ptroyen/pyPlots',
    packages=find_packages(),
    # Core dependencies go here.
    # We are directly listing them instead of reading from requirements.txt
    # for cleaner setup.py definition.
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.5.0',
    ],
    # Define optional dependencies under extras_require
    extras_require={
        'styles': [ # Users can install this group with pip install pyPlots[styles]
            'scienceplots==2.1.1', # Pin to exact version or use >=2.1.1
        ],
        # Add other optional groups if needed, e.g., 'hdf5': ['h5py']
    },
    entry_points={
        'console_scripts': [
            'pyplots=pyplots.cli:main', # Makes 'pyplots' command available
        ],
    },
    classifiers=[
        # Use broader Python version classifiers.
        # python_requires='>=3.7' already handles the minimum version.
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha', # Keep this if still in early development
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.7', # Specifies the minimum required Python version
)