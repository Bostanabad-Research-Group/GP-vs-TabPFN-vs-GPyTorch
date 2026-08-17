import torch
import numpy as np

import argparse

from _GITBO import *
import warnings
warnings.filterwarnings("ignore")
import time

from TestProblems_Utils import *



if __name__ == "__main__":
    value = 10
    rank_r = value
    parser = argparse.ArgumentParser(description='Process experiment parameters Nature_PFN')
    parser.add_argument('--ITER', type=int, default=420, help='Number of iterations')
    parser.add_argument('--DIM', type=int, default=value, help='Number of dim')
    parser.add_argument('--INITIAL_DIR', type=str, default=None, help='For loading initial data')
    parser.add_argument('--SAVE_DIR', type=str, default=None, help='For saving data')
    parser.add_argument('--FUNC_NAME', type=str, default='Ackley', help='Function')
    parser.add_argument('--ACQ', type=str, default='ThompsonSampling', help='ACQ')
    parser.add_argument('--N_SAMPLE', type=int, default=1, help='N_SAMPLE')
    parser.add_argument('--N_PENDING', type=int, default=5000, help='N_PENDING')
    parser.add_argument('--CONSTRAINED', action="store_true", help='CONSTRAINED')
    parser.add_argument('--TRIAL', type=int, default=0, help='TRIAL')
    parser.add_argument('--DEVICE', type=str, default="cpu", help='DEVICE')
    parser.add_argument('--GPU_DEVICE', type=str, default="cuda:0", help='GPU_DEVICE')
    parser.add_argument('--RANK_R', type=int, default=rank_r, help='RANK_R')
    parser.add_argument('--SAMPLE_SCALE', type=float, default=0.2, help='SAMPLE_SCALE')
    parser.add_argument('--GI_SUBSPACE', type=bool, default=True, help='GI_SUBSPACE')
    # t00 = time.time()
    
    
    args = parser.parse_args()

    ITER = args.ITER
    DIM = args.DIM
    INITIAL_DIR = args.INITIAL_DIR
    SAVE_DIR = args.SAVE_DIR
    FUNC_NAME = args.FUNC_NAME
    N_SAMPLE = args.N_SAMPLE
    N_PENDING = args.N_PENDING
    
    ACQ_TYPE = args.ACQ
    TRIAL = args.TRIAL
    CONSTRAINED = args.CONSTRAINED
    DEVICE = args.DEVICE
    GPU_DEVICE = args.GPU_DEVICE
    RANK_R = args.RANK_R
    SAMPLE_SCALE = args.SAMPLE_SCALE
    
    GI_SUBSPACE = args.GI_SUBSPACE
    
    
    # print(f'INITIAL_DIR: {INITIAL_DIR}, SAVE_DIR: {SAVE_DIR}')

    #################################################
    # Assigned Function
    Function = Function_selector(FUNC_NAME, DIM, CONSTRAINED = CONSTRAINED)
    #################################################

    # for TRIAL in range(2):
    # Generate a random seed (several ways to do this)
    opts = []
    t0 = time.time()
    for i in range(3):
        seed = 42 + i # Or your favorite integer
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # If using CUDA:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f'Seed: {seed}')
        print("Function Name:", FUNC_NAME)
        print(f"X Dimensions: {DIM}, Rank: {rank_r}")
    
        # print(f'Seed: {seed}, noise: {noise}')
        # random_seed = np.random.randint(0, 2**31)
        # print(f'random_seed: {random_seed}')
    
        xx_result, maxx = GITBO(Function,  
                                     seed,
                                       Trail_N = TRIAL,
                                       N_iterations = ITER, 
                                       Acquisition = ACQ_TYPE, 
                                       INITIAL_DIR = INITIAL_DIR,
                                       SAVE_DIR = SAVE_DIR,           # None for testing
                                       N_PENDING = N_PENDING,          # Xpen # of samples
                                       N_CANDIDATES = N_SAMPLE,          # Pend 1 for each iteration
                                       DEVICE=DEVICE,
                                       GPU_DEVICE=GPU_DEVICE,
                                        GI_SUBSPACE = GI_SUBSPACE,
                                        rank_r = RANK_R,
                                        scale = SAMPLE_SCALE,
                                      )
        opts.append(maxx[-1].item())
        print(f'\n {FUNC_NAME} GITBO Done \n')
    t1 = time.time()  
    average_opt = np.mean(opts)
    average_time = (t1-t0) / (i+1)
    print(f"Average optimium: {average_opt:.3f} \nAverage time: {average_time:.2f}s ({average_time/60:.2f} min)")
        
    
    
    
    



