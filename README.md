# Description

Official implementation for paper "Robust estimation of heterogeneous treatment effects in randomized trials leveraging external data"

- Our methods are found n src/randomization_aware and baselines are found in src/baselines

## Directory Structure

- **`src/randomization_aware/`**: Contains our proposed methods.
- **`src/baselines/`**: Contains baseline methods used for comparison.
- **`src/data/`**: Contains script to simulate data or sample from real-world STAR dataset.

## Reproduction Instructions

To reproduce the results from the paper:

1. **Install Dependencies**  
   Install required packages using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

2. The script run.py was used to produce the results in Table 1 of the paper:
3. Jupyter notebooks in the notebooks/ directory were used to generate the results shown in the subfigures of Figure 1.
