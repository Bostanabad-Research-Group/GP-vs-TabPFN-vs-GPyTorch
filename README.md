# CURRENT README AND OTHER EXPERIMENTAL SCRIPTS MAY BE INNACURATE AND INCOMPLETE AS OF 8/17/26. 
# FUTURE UPDATES WILL CORRECT THIS, AND EXPLAIN HOW TO USE THIS REPOSITORY MORE CLEARLY. 
# Currently, the repository is unorganized. If you are currently reading this and would like an accurate explanation on how to use the repository, reach out to kbenjaco@uci.edu for help. This repo should be updated and finalized within a few weeks

# Gaussian Processes vs Tabular Prior Fitted Network
Public repository containing scripts for validating experiments and additional results for the paper "On the Uncertainty Quantification Ability of Tabular Foundation Models" - part of the IEEE Computing in Science and Engineering Special Issue on "Controversies on the Usage of AI/ML for Science and Engineering".

# What is currently in this repo: 
- A requirements.txt file containing a list of necessary package and their versions to run these experiments [folder: 'GPvsPFN'].
- Current snapshot of our GP+ library (5-17-2026) [folder: 'gpplus'].
- Scripts to generate data and run comparison studies between GP+ and TabPFN v2.5 [folder: 'experiments'].
- Detailed results from comparison studies (including additonal experiments conducted on real-world datasets and their results) [folder: 'experiments/results_paper'].

# Setup: 
1. Clone repository to a new environment (use python 3.11): https://github.com/Bostanabad-Research-Group/GP-vs-TabPFN-vs-GPyTorch
2. Navigate to the repo's foler: cd GPvsPFN -
3. Install all required packages to environment: pip install -r requirements.txt
<!-- 2. Install compatible version of pytorch: https://pytorch.org/ -->
<!-- 5. Install GP+ package: pip install . -->


# Running experiments:
- All material for running experiments will be in the 'experiments' folder
- Run all comparison problems produced in the paper using the '00_run_all_experiments' script.
- Run comparisons on specific problems individually in the 'A#' scripts within the 'experiments' folder, or edit the '00'/'B#' scripts.
   - The setups in the problems can be easily changed in the function calls for each problem (train_size, num_test, dimensions, etc.).
- Defaults for all problems are in the 'defaults.py' file.
   - Control which model to run with 'run_models'. If 'run_models = None' both models will be ran. If 'run_models = 'gp'', only GP+ model will be used. If 'run_models = 'pfn'', only PFN model will be used.
   - Detailed trainer analysis for GP+ can be enabled by setting 'TRAINER_INFO = TRUE' (GP+ model must be running)
   - GPvsPFN RRMSE and NIS result plotting can be enabled with 'PLOT_METRICS = TRUE'
   - Results will be saved to the 'experiments/results' folder, unless the save-path is edited.


# TabPFN usage:
Use of PriorLab's TabPFN v2.5 model requires token authentification through HuggingFace website. Instructions will automatically be provided in the console's output. The instructions will advise you to sign in and create a token (https://huggingface.co/settings/tokens). In the token's permissions, under 'Repositories', check the box that says "Read access to contents of all public gated repos you can access". Then save the token, then copy and paste it into the console when prompted.
