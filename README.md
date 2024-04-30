# distributed quantile regression feature screening
This repository contains the code to reproduce the experimental results
from the paper *"Communication-Efficient Feature Screening for Ultrahigh-dimensional Data under Quantile Regression"*.

## Prerequisites
- python==3.11.8
- numpy==1.26.4
- scipy==1.12.0
- pandas==2.2.0
- numba==0.59.0
- openpyxl==3.1.2


## Usage
- `simulator.py` contains the code to simulate the data.
- `fun.py` contains all the functions used in the paper.
- `conquer.py` contains the code to reproduce the initial values.
- `screening.py` contains the code to implement the proposed method.
- `U_fun.py` contains the code to implement the U-statistics method.
- `main.py` contains the code to reproduce the results in the paper.




