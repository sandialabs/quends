# Quantification of Uncertainty in ENsembles of Data Streams (QUENDS)

#### Evans Etrue Howard, Abeyah Calpatura, Pieterjan Robbe, Bert Debusschere

[![Deploy to GitHub Pages](https://github.com/sandialabs/quends/actions/workflows/deployment.yml/badge.svg)](https://github.com/sandialabs/quends/actions/workflows/deployment.yml)
[![Run Tests](https://github.com/sandialabs/quends/actions/workflows/python-tests.yml/badge.svg)](https://github.com/sandialabs/quends/actions/workflows/python-tests.yml)
[![pages-build-deployment](https://github.com/sandialabs/quends/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/sandialabs/quends/actions/workflows/pages/pages-build-deployment)

## Overview
This project focuses on uncertainty quantification in plasma turbulent simulations. It includes modules for loading and processing NetCDF and CSV datasets, estimating steady states, computing effective sample sizes, and running uncertainty quantification analyses. The project is structured into multiple Python scripts, each handling different aspects of the analysis. The following Python scripts are included:

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [For Developers](#for-developers)
- [Documentation](#documentation)
- [Examples](#examples)
- [Summary](#summary)
- [Contributing](#contributing)
- [License](#license)

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

5. **Run isort**:
    To format your code with Black:
    ```bash
    isort .
    ```

6. **Run Black**:
    To format your code with Black:
    ```bash
    black .
    ```

## Documentation
For comprehensive information on how to use the QUENDS package, please refer to our [official documentation](https://sandialabs.github.io/quends/documentation/index.html). 

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

