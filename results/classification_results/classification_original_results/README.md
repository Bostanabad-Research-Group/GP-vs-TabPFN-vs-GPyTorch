Baseline classification run, kept because the paper's classification tables were not available. The layout matches `../classification_new_results/`.

- `electrical_grid/`, `truss_6d/`, `stellar/`, and `steel_plates/` hold each dataset's raw rows and summary
- `onedim/onedim_example.csv` and `onedim_example.png` are the 1D Dirichlet example
- `summary/` holds the combined figure and table

A later rerun writes to `../classification_new_results/` and does not change these files. `python compare_to_paper.py` compares the two.
