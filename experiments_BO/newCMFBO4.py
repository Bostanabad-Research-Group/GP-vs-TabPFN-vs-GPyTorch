#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:09:01 2025

@author: kian
"""
import numpy as np
import torch
from gpplus.models import GP_Plus as GP
from gpplus.optim import fit_model_scipy
from joblib import Parallel, delayed, parallel_backend
from joblib.externals.loky import set_loky_pickler
from gpplus.bayesian_optimizations.AFs import CAF_HF, CAF_LF
import random
from scipy.optimize import minimize
from gpplus.preprocessing.normalizeX import standard
from botorch.utils.sampling import draw_sobol_samples
from torch.distributions import Normal
from tqdm import tqdm
import time
import sys
import threading
import pickle

def CMFBO(Xtrain=None,
        ytrain=None,
        costs=None,
        l_bound=None,
        u_bound=None,
        bounds=None,
        xmean=None,
        xstd=None,
        qual_index=None,
        quant_kernel='Rough_RBF',
        optimization_technique='L-BFGS-B',
        data_gen_func=None, # Objective function 
        data_gen_const=None, # Constraint function
        n_restarts=4, # Number of restarts when fitting model (fit_mll_scipy)
        n_samples=8, # Number of samples to evaluate in parallel (run_scipy)
        n_samples_secondary=30, # Number of initial samples to start with for first run of L-BFGS-B
        n_converge=10, # Number of most recently best sampled outputs to be considered for convergence
        n_converge2=10, # Number of most recently best sampled outputs to be considered for convergence in secondary_optimization 
        tol_constraint=1e-3, # Constraint tolerance {g(x) < 0} ----> {g(x) < tol}
        tol_convergence=1e-3,
        tol_convergence2=None, #tol_convergence2 = tol_convergence by default
        maximize_flag=False,
        one_iter=False,
        max_cost=4000000,
        min_iter=1,
        max_iter=None,
        MF=True, # Specify number of initial data: [num_samples_HF, num_samples_LF1, ....]
        AF_hf=CAF_HF,
        AF_lf=CAF_LF,
        IS=True,
        secondary_optimization=False,
        seed=None,
        test_data=False,
        device=None,
        DEBUG=False, # Prints debugging information
        Timer=False, # Prints timer statements
        ):  
      
# %% Function Setup
    
    if DEBUG:
        print("Debug mode is on!")
        first_iteration = True
    
    if Timer: 
        print("Timer is on!")
        time_init0 = time.time()
    
    # if device is None:
    #     tkwargs = {"dtype": torch.double,
    #         "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),}
    
    tkwargs = {"dtype": torch.double,
            "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),}
    
    if DEBUG: print(torch.cuda.is_available())     # ??? ADD CUDA COMPATABILITY
    
    # if bounds: bounds_condition = True
    
    if (min_iter and max_iter) is not None: 
        if min_iter > max_iter:
            raise ValueError("Invalid inputs: min_iter must be less than max_iter")
    
    if data_gen_func is None: raise ValueError("No objective function inputted")
    else: f_x = lambda x: data_gen_func(x).to(**tkwargs)
        
    if data_gen_const is None: raise ValueError("No constraint function inputted")
    
    elif isinstance(data_gen_const, list):
        if not all(callable(g) for g in data_gen_const):
            raise TypeError("All elements in constraint list must be functions")
        g_x=[]
        for g in data_gen_const:
            g_x.append(lambda x, g=g: g(x).to(**tkwargs))
        if DEBUG: print(f"{len(g_x)} constraints being used!")
    
    elif callable(data_gen_const): 
        g_x = lambda x: data_gen_const(x).to(**tkwargs) # Another constraint check ADD
        if DEBUG: print("One constraint is being used!")
    
    else: raise TypeError("Invalid type for constraint input — must be a function or list of functions")
    
    if bounds is None and l_bound and u_bound:
        bounds = torch.tensor([l_bound, u_bound], **tkwargs)
        if DEBUG: print("Bounds inputted as lower and upper...")
    
    bounds_tuple = tuple((l.item(), h.item()) for l, h in zip(bounds[0], bounds[1]))
    
    if tol_convergence2 is None:
        tol_convergence2 = tol_convergence
    
    def set_seed(seed):
        random.seed(int(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else: seed = int(seed)
    set_seed(seed)
    
    # stop_spinner = threading.Event()
    def spinner(stop_event):
        symbols = ['|', '/', '-', '\\']
        i = 0
        while not stop_event.is_set():
            sys.stderr.write(f'\b{symbols[i % len(symbols)]}')
            sys.stderr.flush()
            i += 1
            time.sleep(0.1)
        sys.stderr.write('\b \b')  # clean up spinner
        sys.stderr.flush()
        print(file=sys.stderr)
    
    def generate_initial_data(num_samples, fidelity, seed):
        train_x = draw_sobol_samples(bounds[:,:],n=num_samples,q = 1,batch_shape=None,seed=seed).squeeze(1).to(**tkwargs)
        train_f = fidelity*torch.ones(num_samples,1)
        train_x_full = torch.cat((train_x, train_f), dim=1)
        train_obj = f_x(train_x_full).unsqueeze(-1) 
        if callable(g_x): g_x_list = [g_x]
        else: g_x_list = g_x
        train_constraints = torch.cat([g(train_x_full.clone().detach()).unsqueeze(-1) for g in g_x_list], dim=1)
        return train_x_full, train_obj, train_constraints  
    
    def calculate_cost(Xtrain, costs=costs):
        import torch
        if torch.is_tensor(Xtrain):
            if Xtrain.ndim == 1:
                Xtrain = Xtrain.unsqueeze(0)      
            fidelity = Xtrain[:,-1]
            
        elif not torch.is_tensor(Xtrain):
            fidelity = torch.tensor([Xtrain])
            
        else: raise TypeError("Wrong data input for calculating costs. Input must have associated fidelity!")
        
        cost = 0
        if costs:
            for i in range(len(fidelity)):
                if fidelity[i]==0:
                    cost += costs[0]
                elif fidelity[i]==1:
                    cost += costs[1]
                elif fidelity[i]==2:
                    cost += costs[2]
                elif fidelity[i]==3:
                    cost += costs[3]
                elif fidelity[i] % 1 != 0:
                    raise ValueError("Inputted fidelity source should be a whole number!")
                else:
                    raise ValueError("No cost associated with this fidelity source!")

        return cost

    def run_scipy(EI, best_f, bounds, model, fidelity, n_restarts=n_samples):
        import torch
        import numpy as np
        from scipy.optimize import minimize
        from botorch.utils.sampling import draw_sobol_samples
        bound_np = np.array(bounds, dtype=np.float64)  # shape (11, 2)
        bounds_torch = torch.tensor(bound_np, dtype=torch.float64)  # shape (10, 2)
        bounds_torch = bounds_torch.T # (2, 10)
        fidelity_bound = torch.tensor([[fidelity], [fidelity]], dtype=torch.float64)  # (2, 1)
        bounds_torch = torch.cat((bounds_torch, fidelity_bound), dim=1)               # (2, 11)
        bound_np = np.vstack((bound_np, fidelity_bound.numpy().T))
        sobol_samples = draw_sobol_samples(bounds_torch, n=n_restarts, q=1).squeeze(1).double()
        sobol_samples[:, -1] = torch.round(sobol_samples[:, -1])
        sobol_samples = sobol_samples.numpy()
        all_starts = list(sobol_samples)
        # 4. Run parallel optimization
        def run(x0):
            res = minimize(EI, x0, args=(best_f, model, model_constraints, xmean, xstd, calculate_cost, tol_constraint), jac=True, bounds=bound_np)
            return res.fun, res.x
        set_loky_pickler("dill")
        with parallel_backend('loky'):
            out = Parallel(n_jobs=-1)(delayed(run)(x0) for x0 in all_starts)
        set_loky_pickler("pickle")
        # 5. Pick best result
        temp = [o[0] for o in out]
        tempx = [o[1] for o in out]
        min_index = np.argmin(temp)
        return temp[min_index], tempx[min_index]  # also return all results
    
    def testAF(EI, best_f, bounds, model, fidelity, n_restarts=n_samples): # THIS IS ONLY FOR TESTING PURPOSES. IGNORE OR DELETE LATER
        import torch
        import numpy as np
        from botorch.utils.sampling import draw_sobol_samples
        bound_np = np.array(bounds, dtype=np.float64)  # shape (11, 2)
        bounds_torch = torch.tensor(bound_np, dtype=torch.float64)  # shape (10, 2)
        bounds_torch = bounds_torch.T # (2, 10)
        fidelity_bound = torch.tensor([[fidelity], [fidelity]], dtype=torch.float64)  # (2, 1)
        bounds_torch = torch.cat((bounds_torch, fidelity_bound), dim=1)               # (2, 11)
        bound_np = np.vstack((bound_np, fidelity_bound.numpy().T))
        sobol_samples = draw_sobol_samples(bounds_torch, n=n_restarts, q=1).squeeze(1).double()
        sobol_samples[:, -1] = torch.round(sobol_samples[:, -1])
        sobol_samples = sobol_samples.numpy()
        all_starts = list(sobol_samples)
        temp = []
        tempx = []
        for x in all_starts:
            temp.append(EI(x, best_f, model, model_constraints, xmean, xstd, calculate_cost, tol_constraint))
            tempx.append(x)
        print("testing done")
        
    if secondary_optimization:
        def objective_function(x):
            x = torch.tensor(x).to(torch.float32)
            sample_val = torch.cat(((x.reshape(1, -1) - xmean) / xstd, torch.zeros(1, 1)), dim=1)
            constraint_vals = []
            with torch.no_grad():
                for model_c in model_constraints:
                    constraint_val = model_c.predict(sample_val, return_std=False, include_noise=True)
                    constraint_vals.append(constraint_val)
            constraint_vals = torch.tensor(constraint_vals)
            if (constraint_vals > tol_constraint).any():
                return 1e6
            else:
                with torch.no_grad():
                    prediction = model.predict(sample_val, return_std=False, include_noise=True)
                return prediction.item()
                
    if Timer: print(f"CMFBO initialized in {time.time() - time_init0:.1f} s!")
    
# %% Generate data and evaluate cost
    if DEBUG: print("Generating data and evaluating costs...")
    if Timer: time_data0 = time.time()
    
    if Xtrain is None and ytrain is None:
        Xtrain_true = []
        ytrain = []
        constraints = []
        
        for i in range(len(MF)):
            Xtrain_temp, ytrain_temp, constraints_temp = generate_initial_data(MF[i], i, seed)
            
            Xtrain_true.append(Xtrain_temp)
            ytrain.append(ytrain_temp)
            constraints.append(constraints_temp)
        
        Xtrain_true = torch.cat(Xtrain_true)
        ytrain = torch.cat(ytrain).reshape(-1)
        
        if isinstance(data_gen_const, list):
            constraints = torch.cat(constraints)
        else: constraints = torch.cat(constraints).reshape(-1)
        
        Xtrain, xmean, xstd = standard(Xtrain_true, qual_index)
        
        X_high=[]
        
        y_high=[]
        for i in range(MF[0]):
            if (constraints[i]<=tol_constraint).all():
                X_high.append(list((Xtrain_true[i][0:-1])))
                y_high.append(ytrain[i].item())
        
        #######
        # More needs to be added to implement testing data! #
        if test_data: # Either generate extra test data, or split Xtrain into training and testing data
            print("...")
        #######
    
    cumulative_cost = []
    if costs is not None: 
        initial_cost = calculate_cost(Xtrain)
        cumulative_cost.append(initial_cost) # append intial cost
    
    else: 
        print("***** WARNING: Costs per samples were not inputted... Model will ignore costs! *****")
        cumulative_cost.append(float('-inf')) # Main loop will not end from cost considerations
        
    if Timer: print(f"Data generated and cost calculated in in {time.time()-time_data0:.1f} s!")
    
# %% Beginning of main Loop

    # Initialize variables below
    best_x=[]
    best_y_HF=[]
    xmin_list=[]
    ymin_list=[]
    Fidelity=[]
    improvement=False
    if secondary_optimization:
        initial_temp=[]
        # stop_MFBO=[]    
        secondary_optimal_x_scipy=[]
        secondary_optimal_y_scipy=[]
        # min_val=[]
        
    if DEBUG:
        iter_found=0
        failed=False
    
    # if cost is None:
    #     condition = lambda: m < max_iter
    # else:
    #     condition = lambda: cumulative_cost[-1] < cost_limit

    # while condition():
    #     # shared loop body
    #     m += 1
    
    if max_iter is None:
        max_iter = float('inf')
    
    iteration=[] # Probably can remove...
    m=0 # Loop counter
    if DEBUG: print("Beginning Loop...")
    time_loop0 = time.time()
    while True:
        if Timer: time_loop1 = time.time()
        if m != 0 and Timer or DEBUG:
            sys.stderr.write('\r' + ' ' * 80 + '\r') # Fixes spinner position if being used
            sys.stderr.flush()
        
        iteration.append(m)
        m+=1
        best_y_all = []
        for i in range(len(MF)):
            fidelity_mask = Xtrain[:, -1] == i
            if constraints.dim() == 1: constraint_mask = (constraints <= tol_constraint)
            else: constraint_mask = (constraints <= tol_constraint).all(dim=1)
            combined_mask = fidelity_mask & constraint_mask
            X_feasible = Xtrain[combined_mask]
            y_feasible = ytrain[combined_mask]
        
            if len(y_feasible) == 0:
                raise ValueError(f"No feasible points found for fidelity level {i}")
        
            min_idx = torch.argmin(y_feasible)
            best_y_temp = y_feasible[min_idx]
            best_y_all.append(best_y_temp)
        
            if i == 0:  # Only append best_x from high-fidelity source
                best_x_feasible = X_feasible[min_idx]
                best_x.append(best_x_feasible)
        
        best_y_HF.append(best_y_all[0])
        # iteration.append(m)
        
        if DEBUG and m % 10 == 0: # The print statement is to be used as a breakpoint during debugging if desired. Otherwise, should be commented out
            print("Checkpoint")
            
        # CONVERGENCE CHECKS... Potentially can add 'Improvement' variable to confirm a new optimum has been found before completing...
        if (cumulative_cost[-1] >= max_cost):
            if DEBUG: print("Maximum cost reached. Ending optimization loop!")
            break
        if (m > max_iter):
            if DEBUG: print("Maximum iterations complete. Ending optimization loop!")
            break
        if m>min_iter: # Convergence check
            if np.var(best_y_HF[-n_converge:])<tol_convergence:
                if DEBUG: print("Convergence criteria has been satisfied. Ending optimization loop!")
                if improvement: break

        if DEBUG: print(f"\n---------- New iteration: {m}, Cost: {cumulative_cost[-1]} ----------")

# %% Model setup and training
    
    ############ Model training #############
        if Timer: time_model = time.time()
        if DEBUG:
            print("Fitting models...  ", end="", flush=True)
            stop_event = threading.Event()
            spinner_thread = threading.Thread(target=spinner, args=(stop_event,), daemon=True)
            spinner_thread.start()

        # model = GPR(train_x = Xtrain, 
        #             train_y = ytrain,
        #             )
        # ??? Create models using GPR and GPTrainer!
        
        try: 
            model = GP(
                train_x=Xtrain,
                train_y=ytrain,
                qual_dict=qual_index,
                multiple_noise=True,
                # lb_noise = 1e-4,
                # fix_noise = False,
                m_gp='multiple_constant',
                quant_correlation_class=quant_kernel,
                interval_score=True
                )
            
            fit_model_scipy(
                model, 
                num_restarts = n_restarts, # number of restarts in the initial iteration
                add_prior=False, 
                n_jobs= -1,
                method = optimization_technique,
                bounds=True)
    
            if constraints.ndim == 1:
                constraints = constraints.unsqueeze(-1)  # (N,) -> (N,1)
    
            model_constraints=[]
            for i in range(constraints.shape[1]):  # iterate over each constraint dimension
                model_constraint_temp = GP(
                    train_x=Xtrain,
                    train_y=constraints[:, i].reshape(-1, 1),  # constraint i as column vector
                    m_gp='multiple_constant',
                    qual_dict=qual_index,
                    # multiple_noise=True,
                    # lb_noise=1e-5,
                    fix_noise=False,
                    quant_correlation_class=quant_kernel,
                    interval_score=True
                )
            
                fit_model_scipy(
                    model_constraint_temp,
                    num_restarts=n_restarts,
                    add_prior=False,
                    n_jobs=-1,
                    method=optimization_technique,
                    bounds=True
                )
            
                model_constraints.append(model_constraint_temp)
                
        except KeyboardInterrupt:
            stop_event.set()
            spinner_thread.join()
            raise KeyboardInterrupt()

        if DEBUG:
            stop_event.set()
            spinner_thread.join()

        if Timer:
            if m < 2: print(f"Models created and fit in {time.time() - time_model:.1f} s!")
            if m >= 2: print(f"Models updated and fit in {time.time() - time_model:.1f} s!")

        if DEBUG and first_iteration:
            print("===== DEBUG INFO =====")
            print("Seed:", torch.initial_seed())
            print("X shape:", Xtrain.shape, "Y shape:", ytrain.shape)
            # print("LR:", optimizer.param_groups[0]['lr'])
            print("Model:", model)
            # print("Kernel params:", model.covar_module.raw_lengthscale)
            print("CUDA:", next(model.parameters()).device)
            print("======================")
            first_iteration = False
            
# %% Aquisition function scoring

        if DEBUG: print("Running aquisition functions to determine next sampling point...")
        if Timer: time_AF = time.time() # Time for finding next best samples
        #################################################
        
        Y_list = []
        X_list = []
        # Ytemp, Xtemp = testAF(CAF_HF, best_y_all[0], bounds_tuple, model, fidelity = 1) # This line is for testing during development
        iterator1 = tqdm(range(len(MF)), desc="Aquisition Functions", unit="Fidelity source", unit_scale=True) if Timer or DEBUG else range(len(MF))
        for i in iterator1:
            if i == 0:
                Ytemp, Xtemp = run_scipy(AF_hf, best_y_all[i], bounds_tuple, model, fidelity = i)
            else:
                Ytemp, Xtemp = run_scipy(AF_lf, best_y_all[i], bounds_tuple, model, fidelity = i)
            Y_list.append(Ytemp)           
            X_list.append(Xtemp)
        
        # if Timer or DEBUG: sys.stdout.flush()
        
        Y_list_temp = Y_list.copy()       
        
        for i in range(len(MF)):
            sample_val = torch.cat([(torch.tensor(X_list[i][:-1]).reshape(1, -1) - xmean)/xstd,
                torch.tensor(X_list[i][-1]).reshape(1, -1)], axis=1)
            
            constraint_vals = []
            with torch.no_grad():
                for model_c in model_constraints:
                    constraint_val, _ = model_c.predict(sample_val, return_std=True, include_noise=True)
                    constraint_vals.append(constraint_val.item())
            constraint_vals = torch.tensor(constraint_vals)
            if (constraint_vals > tol_constraint).any():
                if DEBUG: failed = True
                with torch.no_grad():
                    mean, sigma = model.predict(sample_val, return_std=True, include_noise=True)
                    
                if sample_val[:,-1] == 0: 
                    print("HF sample is infeasible!")
                    ei_val = -(mean - best_y_all[i])
                    if costs is None: Y_list[i] = -ei_val.item()
                    else: Y_list[i] = -(ei_val / calculate_cost(sample_val)).item()
                else: 
                    print(f"LF{int(sample_val[:, -1])} sample is infeasible!")
                    u = -((mean - best_y_all[i]) / sigma)
                    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
                    updf = torch.exp(normal.log_prob(u))
                    ei_val = sigma * updf
                    if costs is None: Y_list[i] = -ei_val.item()
                    else: Y_list[i] = -(ei_val / calculate_cost(sample_val)).item()
              
        if DEBUG:
            if failed:
                print('############ EI/cost before checking constraints! ###############')
                print(np.array2string(np.array(Y_list_temp), formatter={'float_kind':lambda x: f"{x:.4e}"}))
                print('#################################################################')  
                failed = False
            print('############ Found EI/cost ###############')
            print(np.array2string(np.array(Y_list), formatter={'float_kind':lambda x: f"{x:.4e}"}))
            print('##########################################')         
            
        min_index = np.argmin(Y_list)
        if DEBUG:
            if min_index == 0:
                print("Using HF source samples!")
            else:
                print(f"Using LF{min_index} source samples")
                
        Xnew = X_list[min_index]
        ynew = f_x(torch.as_tensor(Xnew).unsqueeze(0))
       
        if constraints.shape[-1] == 1: const_new = g_x(torch.as_tensor(Xnew).unsqueeze(0)).unsqueeze(-1) 
        else: const_new = torch.stack([g(torch.as_tensor(Xnew).unsqueeze(0)) for g in g_x], dim=-1)
        
        if Xnew[-1]==0 and (const_new<=tol_constraint).all(): # Only add new sample value and output if constraints are satisfied, and it is from the high fidelity
            X_high.append(list(Xnew[0:-1]))
            y_high.append(ynew.item())
            
        constraints = torch.cat([constraints, const_new]) # USE APPEND INSTEAD? or is it impossible with torch?
        
        Xnew = np.concatenate([((Xnew[0:-1]-xmean.numpy())/xstd.numpy()).reshape(1,-1), Xnew[-1].reshape(-1,1)], axis=-1) # Normalizing new X sample

        if Timer: print(f"Aquisition functions completed in {time.time() - time_AF:.1f} s!")
        
# %% Secondary optimization
    
        if secondary_optimization:
            if DEBUG: print("\nPerforming secondary optimization...")
            if Timer: time_opt = time.time()
            
            if len(initial_temp)==0:
                x0=draw_sobol_samples(bounds, n=n_samples_secondary, q=1, batch_shape=None, seed=i).squeeze(1).to(**tkwargs)
            else: x0=torch.tensor(initial_temp)
            
            x0=torch.cat((x0,torch.tensor(X_high)),axis=0)
            
            optim_x=[]
            optim_y=[]
            
            iterator = tqdm(range(len(x0)), desc="L-BFGS optimization", unit="Restart", unit_scale=True) if Timer or DEBUG else range(len(x0))
            for j in iterator:
            # for j in range(len(x0)):
                result_lbfgs = minimize(objective_function, x0[j], bounds=bounds_tuple)
                optim_x.append(result_lbfgs.x)
                optim_y.append(result_lbfgs.fun)
            # if Timer or DEBUG: sys.stdout.flush()
            if Timer: print(f'L-BFGS with {len(x0)} restarts completed in {time.time() - time_opt:.1f} s!')
                      
            index=torch.argmin(torch.tensor(optim_y))
            optimal_x_scipy=torch.cat((torch.tensor(optim_x[index]).unsqueeze(0), torch.zeros(1,1)),dim=1).to(torch.float64)
            secondary_optimal_y_scipy.append(optim_y[index]) #### WHAT IS SECONDARY OPTIMAL VS BEST FEASIBLE VS BEST?
            secondary_optimal_x_scipy.append(optimal_x_scipy.clone())
            
            #### sort best_ys
            index=np.argsort(optim_y)
            initial_temp=np.array(optim_x)[index][0:n_converge2]
             
            if list(optimal_x_scipy[:,0:-1][0]) not in X_high:
                X_high.append(list((optimal_x_scipy[:,0:-1][0])))
                y_high.append(secondary_optimal_y_scipy[-1])
                
        
            if m > n_converge2:
                # for k in range(len(secondary_optimal_y_scipy)-n_converge2):
                y_mean=np.mean(secondary_optimal_y_scipy)
                y_std=np.std(secondary_optimal_y_scipy)
                
                if (np.var((secondary_optimal_y_scipy[-n_converge2:]-y_mean)/y_std))<=tol_convergence:
                    sec_cand=min(secondary_optimal_y_scipy[-n_converge2:])
                    best_y_HF[-1] = min(best_y_HF, sec_cand)
                    break
               
            ## best val we report is min(sec_cand, best HF sample)
            
# %% End of Loop
       
        Xtrain = torch.cat([Xtrain,torch.tensor(Xnew.reshape(1,-1))])
        ytrain = torch.cat([ytrain, ynew.reshape(-1,)])
        ymin_list.append(ynew.reshape(-1,))
        xmin_list.append(Xnew)
        Fidelity.append(Xnew[0][-1])
    
        if costs:
            new_cost = cumulative_cost[-1] + calculate_cost(Xnew[0][-1])
            cumulative_cost.append(new_cost)
    
        if DEBUG:
            print(f"\nNew X: [{', '.join(f'{x:.4f}' for x in Xnew.flatten())}]")
            print(f'New y: {float(ynew):.3f}')
            print(f'Current best y: {best_y_HF[-1].item():.3f} (iter: {iter_found})')
            if secondary_optimization: print(f"Last 10 secondary y optimums: {secondary_optimal_y_scipy[-10:]}")
            
            if maximize_flag: 
                if ynew>best_y_HF[-1] and Xnew[:,-1] == 0: 
                    print("New HF maximum found!")
                    iter_found = m
                    improvement = True
            else: 
                if ynew<best_y_HF[-1] and Xnew[:,-1] == 0:
                    print("New HF minimum found!")
                    iter_found = m
                    improvement = True
            
        if Timer: print(f'\nIteration #{m} time: {time.time() - time_loop1:.1f} s ({(time.time() - time_loop1)/60:.1f} mins)... Running time: {(time.time() - time_loop0)/60:.2f} min... [Seed: {int(seed)}]')
 
    total_time = time.time() - time_loop0
    if Timer: print(f'Total time for all {m-1} iterations: {total_time:.1f} s ({total_time/60:.2f} min)... [Seed: {int(seed)}]')

    return best_x, best_y_HF, cumulative_cost,xmin_list,ymin_list, Fidelity,secondary_optimal_x_scipy,secondary_optimal_y_scipy, total_time/60, iteration
# %% Test


if __name__ == '__main__':
    from gpplus.test_functions.wing_for_Kian import multi_fidelity_wing_value, const_MF, const_MF2
# Necessary inputs: cost function, MF=true, samples from each source
    costs = [1000, 100, 10, 1]
    MF = [30, 60, 60, 60] # How many samples to initialize for initial data for each data/fidelity source
    MF_test = [20,30,30,30]
    MF2 = [5,5,5,5]
    qual_index = {10:4}
    l_bound = [150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025]
    u_bound = [200, 300, 10, 10, 45, 1, 0.18, 6, 2500, 0.08]
    # bounds = ((150,200),(270,300),(6,10),(-10,10),(15,45),(0.5,1),(0.08,0.18),(2.5,6),(1700,2500),(0.025,0.08))
    constraint1 = const_MF
    constraint2 = const_MF2 
    constraint_list = [const_MF, const_MF2] # Constraints should be inputted as a list, items of lists should be functions
    # [best_x_feasible, best_y_HF, cumulative_cost, xmin_list, ymin_list, Fidelity,secondary_optimal_x_scipy,secondary_optimal_y_scipy, total_time, m] = CMFBO(costs=costs, 
    #            l_bound=l_bound, 
    #            u_bound=u_bound, 
    #            qual_index=qual_index, 
    #            data_gen_func=multi_fidelity_wing_value, 
    #            data_gen_const=constraint1, 
    #            MF=MF, 
    #            min_iter=50,
    #            n_restarts=32, # Number of restarts when fitting model (fit_mll_scipy)
    #            n_samples=8, # Number of samples to evaluate in parallel (run_scipy)
    #            secondary_optimization=True,
    #            DEBUG=True,
    #            Timer=True 
    #            ) 
    itr = 0
    t0 = time.time()
    np.random.seed(1) #4567
    random_seed0 = np.random.choice(range(0,1000), size=10, replace=False)
    random_seed = random_seed0[5:]
    if not isinstance(random_seed, (list, np.ndarray)): random_seed = [random_seed]
    for seed in random_seed:
        itr += 1
        output = {'best_x_feasible':[],'best_y_feasible':[],'cost':[],'xmin_list':[],'ymin_list':[],'Fidelity':[],'secondary_optimal_x_scipy':[],'secondary_optimal_y_scipy':[],'total_time':[],'num_iter':[]}
        print(f'***************** Random state: #{itr}, Seed: {int(seed)} *****************')
        # [best_feasible, cumulative_cost,xmin_list,ymin_list,Fidelity,secondary_optimal_x_scipy,secondary_optimal_y_scipy,best_predict,MSE_HF,MSE_l1,MSE_l2,MSE_l3] = run_MA2X_Bayesian(seed = seed, plotflag=False)
        [best_x, best_y_HF, cumulative_cost, xmin_list, ymin_list, Fidelity,secondary_optimal_x_scipy,secondary_optimal_y_scipy, total_time, iteration] = CMFBO(costs=costs, 
            l_bound=l_bound, 
            u_bound=u_bound, 
            qual_index=qual_index, 
            data_gen_func=multi_fidelity_wing_value, 
            data_gen_const=constraint1, 
            MF=MF, 
            n_samples_secondary=30,
            min_iter=50,
            n_restarts=32, # Number of restarts when fitting model (fit_mll_scipy)
            n_samples=16, # Number of samples to evaluate in parallel for aquisition functions (run_scipy)
            secondary_optimization=False,
            seed=seed,
            DEBUG=True,
            Timer=True) 
        output['cost'].append(cumulative_cost)
        output['best_x_feasible'].append(best_x)
        output['best_y_feasible'].append(best_y_HF) 
        output['xmin_list'].append(xmin_list) 
        output['ymin_list'].append(ymin_list) 
        output['Fidelity'].append(Fidelity)  
        output['secondary_optimal_x_scipy'].append(secondary_optimal_x_scipy)   
        output['secondary_optimal_y_scipy'].append(secondary_optimal_y_scipy)  
        output['total_time'].append(total_time)
        output['num_iter'].append(iteration)

        print(f'^^^^^^^^^^^^ End of Random state: #{itr}, Seed: {int(seed)} ^^^^^^^^^^^^\n\n')
        file = open(f"/home/kian/Repos/gp-private/examples/wingtestresults5_152_12n_{int(seed)}.pkl", "wb")
        pickle.dump(output,file)
        file.close()

    print(f"FINAL time for {itr} seeds: {time.time()-t0:.1f} s ({(time.time()-t0)/60:.2f} min)!")
    # file = open("D:/Research/codes/BO_Fourth_Paper/Results/Final/MFBO_Wing_1_ID_small_data.pkl", "wb")
    
