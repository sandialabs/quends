# Quantification of Uncertainty in ENsembles of Data Streams (QUENDS)

#### Evans Etrue Howard, Abeyah Calpatura, Pieterjan Robbe, Bert Debusschere


## Overview
This project focuses on uncertainty quantification in plasma turbulent simulations. It includes modules for loading and processing NetCDF and CSV datasets, estimating steady states, computing effective sample sizes, and running uncertainty quantification analyses. The project is structured into multiple Python scripts, each handling different aspects of the analysis. The following Python scripts are included:

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [For Developers](#for-developers)
- [Module Descriptions](#module-descriptions)
- [Examples](#examples)
- [Summary](#summary)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
quends/
├── base/
│   ├── DataStream.py
│   └── DataStream methods:
│       ├── __init__()
│       ├── additional_data()
│       ├── compute_statistics()
│       ├── confidence_interval()
│       ├── cumulative_statistics()
│       ├── effective_sample_size()
│       ├── effective_sample_size_below()
│       ├── find_steady_state_rolling_variance()
│       ├── find_steady_state_std()
│       ├── find_steady_state_threshold()
│       ├── head()
│       ├── is_stationary()
│       ├── mean()
│       ├── mean_uncertainty()
│       ├── normalize_data()
│       ├── optimal_window_size()
│       ├── trim()
│       └── variables()
│   ├── Ensemble.py
│   └── Ensemble methods:
│       ├── __init__()
│       ├── common_variables()
│       ├── compute_statistics()
│       ├── confidence_interval()
│       ├── effective_sample_size()
│       ├── get_member()
│       ├── head()
│       ├── is_stationary()
│       ├── mean()
│       ├── mean_uncertainty()
│       ├── members()
│       ├── summary()
│       └── trim()
│
├── postprocessing/
│   ├── Exporter.py
│   └── Exporter methods:
│       ├── __init__()
│       ├── display_dataframe()
│       ├── display_dictionary()
│       ├── display_json()
│       ├── display_numpy()
│       ├── save_dataframe()
│       ├── save_dictionary()
│       ├── save_json()
│       ├── save_numpy()
│       ├── to_dataframe()
│       ├── to_dictionary()
│       ├── to_json()
│       └── to_numpy()
│   ├── Plotter.py
│   └── Plotter methods:
│       ├── __init__()
│       ├── ensemble_steady_state_automatic_plot()
│       ├── ensemble_steady_state_plot()
│       ├── ensemble_trace_plot()
│       ├── format_dataset_name()
│       ├── plot_acf()
│       ├── plot_acf_ensemble()
│       ├── steady_state_automatic_plot()
│       ├── steady_state_plot()
│       └── trace_plot()
│
└── preprocessing/
    ├── from_csv()
    ├── from_dict()
    ├── from_gx()
    ├── from_json()
    ├── from_netcdf()
    └── from_numpy()

```

## Installation

1. **Clone the repository**:
    - Using SSH:
    ```bash
    git clone git@github.com:sandialabs/quends.git
    cd quends
    ```
    - Using HTTPS:
    ```bash
    git clone https://github.com/sandialabs/quends.git
    cd quends
    ```

2. **Install the package and dependencies**:
    You can install the package along with its dependencies using pip:
    ```bash
    pip install .
    ```

3. **Verify the installation**:
    To ensure that the installation was successful, you can run a simple test:
    ```bash
    python -c "import quends"
    ```


## Usage

Examples are shown in the `examples/notebooks` directories.
- `cgyro`: Contains all CGYRO data
- `gx`: Contains all gx data
- `gx/ensemble`: Contains all ensemble data
- `DataStream_Guide-CGRYO.ipynb`: DataStream guide for CGYRO data
- `DataStream_Guide-GX.ipynb`: DataStream guide for GX data
- `DataStream_Guide-Ensemble.ipynb`: DataStream guide for Ensembles
- `DataStream_Guide.ipynb`: DataStream guide

## For Developers

1. **Clone the repository**:
    - Using SSH:
    ```bash
    git clone git@github.com:sandialabs/quends.git
    cd quends
    ```
    - Using HTTPS:
    ```bash
    git clone https://github.com/sandialabs/quends.git
    cd quends
    ```

2. **Install the package and dependencies**:
    You can install the package along with its dependencies using pip:
    ```bash
    pip install -e .\[dev\]
    ```

3. **Install pre-commit hooks**
    To ensure code quality and consistency, install:
    ```bash
    pre-commit install
    ```

4. **Run Ruff**:
    For linting and fixing issues:
    ```bash
    ruff check --fix
    ```

5. **Run Black**:
    To format your code with Black:
    ```bash
    black .
    ```

6. **Run isort**:
    To format your code with Black:
    ```bash
    isort .
    ```


### Module Descriptions

### `base/`
The `base/` directory contains core classes and methods that form the foundation of the uncertainty quantification toolkit. These classes are designed to handle and analyze time series data, providing essential functionalities for data manipulation, statistical analysis, and ensemble management.

#### Key Modules:
- **`DataStream.py`**: This module provides the `DataStream` class, which is designed to handle and analyze time series data. It includes methods for computing various statistics, assessing data quality, and performing data normalization.

    **Key Methods:**
    - `__init__()`: Initializes the DataStream object with data.
    - `additional_data()`: Adds additional data to the stream.
    - `compute_statistics()`: Computes basic statistics for the data.
    - `cumulative_statistics()`: Computes cumulative statistics for the data.
    - `confidence_interval()`: Calculates confidence intervals for the data.
    - `effective_sample_size()`: Estimates the effective sample size.
    - `mean()`: Compute the mean of the short-term averages.
    - `mean_uncertainty()`: Compute the uncertainty (standard error) of the mean of the short-term averages.
    - `optimal_window_size()`: Returns the optimal window size that results in the lowest uncertainty (minimum std) in the mean prediction.
    - `trim()`: Trims the data to start from the steady state and retains only the specified column.

- **`Ensemble.py`**: This module provides the `Ensemble` class, which is used to manage and analyze ensembles of data. Represents an ensemble of DataStream objects.

    **Key Methods:**
    - `__init__()`: Initializes the Ensemble object with a set of members.
    - `confidence_interval()`: Calculates confidence intervals for the ensemble.
    - `effective_sample_size()`: Estimates the effective sample size for each sepcified columns in each member.
    - `mean()`: Compute the mean for the ensemble.
    - `mean_uncertainty()`: Compute the mean uncertainty for the ensemble.
    - `summary()`: Print and return a summary of the ensemble: number of members, common variables, and a brief summary (number of samples, columns, head) for each member.

### `postprocessing/`
The `postprocessing/` directory contains modules responsible for exporting and visualizing the results of the uncertainty quantification analyses. These modules facilitate the presentation of data in various formats and provide tools for creating informative plots to help users interpret the results.

#### Key Modules:
- **`Exporter.py`**: This module provides the `Exporter` class, which is responsible for encapsulates plotting functionality for time series data.. It includes methods for saving dataframes, dictionaries, and other data structures to files.

    **Key Methods:**
    - `__init__()`: Initializes the Exporter object.
    - `save_dataframe()`: Saves a DataFrame to a specified file format.
    - `to_json()`: Converts data to JSON format.

- **`Plotter.py`**: This module provides the `Plotter` class, which encapsulates plotting functionality for time series data.

    **Key Methods:**
    - `__init__()`: Initializes the Plotter object.
    - `ensemble_steady_state_automatic_plot()`: Plot steady state detection automatically for each ensemble member on a grid.
    - `ensemble_steady_state_plot()`: Plot steady state detection for each ensemble member using a user-supplied steady state start.
    - `ensemble_trace_plot()`: Plot ensemble time series data, with traces from each ensemble member plotted on the same axes.

### `preprocessing/`
This module provides functions for loading and preprocessing data from various sources. It includes methods for reading data from CSV, JSON, NetCDF, and other formats.

**Key Functions:**
- `from_csv()`: Load a data stream from a CSV file.
- `from_dictionary()`: Load a data stream from a dictionary.
- `from_gx()`: Load a data stream from a GX outputs.
- `from_json()`: Loads data from a JSON file.
- `from_netcdf()`: Load specified variables from a NetCDF4 file into a pandas DataFrame, ensuring all variables have the same length, and extracting only variables that end with ‘_t’ or ‘_st’ from the Diagnostics group.
- `from_numpy()`: Load a data stream from a NumPy array.


## Summary
Key functionalities include:
- **Data Handling**: Seamlessly load and preprocess data from various formats, including CSV, JSON, and NetCDF.
- **Statistical Analysis**: Compute essential statistics and assess data quality with built-in methods for effective sample size estimation and confidence interval calculations.
- **Visualization**: Create informative plots to visualize trends, correlations, and patterns in time series data.

## Contributing
Feel free to submit issues and merge requests. For major changes, please open an issue first to discuss what you would like to change.

## License
BSD 3-Clause License

Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

