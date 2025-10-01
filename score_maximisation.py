import numpy as np
import pandas as p
import numba
import configparser


np.random.seed(1312)

## SCORE CALCULATION BASED ON SET VALUE METRICS ##
@numba.jit(fastmath=True, nogil=True, parallel = True)
def score(pred,target, func, args):
    VP = np.sum(np.logical_and(pred,target))
    FP = np.sum(pred) - VP
    FN = np.sum(target) - VP
    TN = len(pred) - VP - FP - FN
    return func(VP, FP, TN, FN, args)

## DEFINE OF UTILITY FUNCTIONS ##
@numba.njit(fastmath=True, nogil=True)
def f_b(TP, FP, TN, FN, args = [1]):
    if TP + FP + FN == 0:
        return 0
    b = args[0]
    return (1+b**2)*TP/((1+b**2)*TP + b*FN + FP)

@numba.njit(fastmath=True, nogil=True)
def jaccard(TP, FP, TN, FN, args = [1]):
    if TP + FP + FN == 0:
        return 0
    return TP/(TP + FP + FN)


@numba.njit(fastmath=True, nogil=True)
def deltaSR2(TP, FP, TN, FN, args = [1]):
    return -(FN-FP)**2

@numba.njit(fastmath=True, nogil=True)
def deltaSR1(TP, FP, TN, FN, args = [1]):
    delta = abs(FN-FP)
    if delta < args[0]:
        return -delta**2
    else:
        return -args[0]**2 - np.sqrt(delta - args[0])

@numba.njit(fastmath=True, nogil=True)
def tss(TP, FP, TN, FN, args = [1]):
    sensitivity = 0
    specificity = 0
    if (TP + FN) != 0:
        sensitivity = TP / (TP + FN)
    if (FP + TN) != 0:
        specificity = TN / (FP + TN)
    return sensitivity + specificity - 1


## MAXIMIZATION OF THE SCORE ##
@numba.njit(fastmath=True, nogil=True, parallel = True)
def max_func_n3(p, func, args):
    n = len(p)
    Umax, kmax, U  = -np.inf, 0, -np.inf
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    for i in range(n):
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros((n+1,n+1))
    S[0,0] = 1
    for i in range(n): 
        S[i+1,0] = (1 - p[n-i-1])*S[i,0]
        for j in range(i+1):
            S[i+1,j+1] = p[n-i-1]*S[i,j] + (1 - p[n-i-1])*S[i,j+1]     

    K = 1
    while K < n and U == Umax:
        U = 0.
        for i in range(K+1):
            for j in range(n - K + 1):
                U += C[K,i]*S[n-K,j]*func(i, K -i, n-K-j, j, args) #func(tp,fp,tn, fn)
        if U >= Umax :
            Umax, kmax  = U, K
        K += 1
    return kmax, Umax #return the number of species to keep and the score

@numba.njit(fastmath=True, nogil=True, parallel = True)
def max_f_b_n2(p, func, args):
    b = args[0]**(-2)
    p = np.sort(p)[::-1]
    n = len(p)
    Umax, kmax, U  = -np.inf, 0, -np.inf
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    for i in range(n): 
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros(2*n)      
    for i in range(2*n):
        S[i] = 1/(i+1)
    
    k = 0
    while k<n and U == Umax:
        U = 0
        K = n - k
        for i in range(1, K+1):
            U += 2*i*C[K,i]*S[i + K -1]
        for i in range(2*(K-1)):
            S[i] = p[K-1]*S[i+1] + (1 - p[K-1])*S[i]
            
        if U >= Umax :
            Umax, kmax  = U, K
        k += 1
    return kmax, Umax
    
@numba.njit(fastmath=True, nogil=True, parallel = True)
def max_jaccard_n2(p, func, args):
    pass
    
## BASELINE STRATEGIES ##
@numba.njit(fastmath=True, nogil=True)
def threshold(p, th = 0.5) :
    K = 0
    while  K < len(p) and p[K] >= th:
        K += 1
    return K

@numba.njit(fastmath=True, nogil=True, parallel = True)
def sum_th(p) :
    s = np.sum(p)
    return int(2*s - int(s))


## APPLY DECISION FUNCTIONS TO EACH SITES ##
@numba.njit(fastmath=True, nogil = True)
def iterate(SOL, PROBAS, P_CALIB, T_CALIB, func, args, max_func):
    S = len(SOL)
    N = len(SOL[0])
    ST = len(T_CALIB)


    SCORE = np.zeros((S, 7))
    KLIST = np.zeros((S, 8))

    OUTPUT = []

    # Calibrate strategies on the given data #

    print('BEGIN CALIBRATION...')

    # TopK
    Ktopk = 0
    Ut_topkmax = -np.inf
    Ktopkmax = 0

    upper_bound = 80

    upper_bound = min(upper_bound, N)
    if upper_bound > 100:
        print('WARNING: Topk calibration for many species may take a long time ...  \n' \
        'Consider changing the upper bound of the calibration')
    
    while Ktopk < upper_bound : #adapt upper bound if too many species
        Ut_topk = 0.
        for i in range(ST):
            probas = P_CALIB[i]
            tar = T_CALIB[i]
            sort = np.argsort(-probas)
            mask = np.where(probas > 0)[0]
            input = probas[sort][mask]
            topk = np.zeros(N)
            topk[sort[:Ktopk]] = 1
            Ut_topk += score(topk,tar,func, args)
        if Ut_topk >= Ut_topkmax:
            Ut_topkmax = Ut_topk
            Ktopkmax = Ktopk
        Ktopk += 1

    Ktopk = Ktopkmax
    print('Topk done')

    # Global threshold
    e = 0.01
    th = 1
    thmax = 1
    Uth_max = -np.inf
    while th >= 0:
        Ut_th = 0.
        for i in range(ST):
            probas = P_CALIB[i]
            tar = T_CALIB[i]
            sort = np.argsort(-probas)
            mask = np.where(probas > 0)[0]
            input = probas[sort][mask]
            Kth = threshold(input, th = th)
            nth = np.zeros(N)
            nth[sort[:Kth]] = 1
            Ut_th += score(nth, tar, func, args)
        if Ut_th >= Uth_max:
            Uth_max = Ut_th
            thmax = th
        th = th - e
    
    th = thmax
    print('Global threshold done')

    # Frequency threshold
    thf = np.ones(N)
    for i in range(N):
        per = 100 - np.mean(T_CALIB[:,i])*100
        if per == 100 :
            thf[i] = 1
        else:
            thf[i] = np.percentile(P_CALIB[:,i], per)
    print('Frequency threshold done')

    # Conformal threshold
    e = 0.01
    c = 1 -e
    thc_max = np.ones(N)
    Uc_max = -np.inf
    thc = np.ones(N)
    while c >= 0:
        Uc = 0.
        for i in range(N):
            occ = np.where(T_CALIB[:,i] == 1)[0]
            if len(occ) > 0:
                thc[i] = np.percentile(P_CALIB[occ,i], int(c*100))

        for i in range(ST):
            probas = P_CALIB[i]
            tar = T_CALIB[i]
            nc = probas > thc
            Uc += score(nc,tar,func, args)
        if Uc > Uc_max:
            Uc_max = Uc
            thc_max = thc.copy()
        c = c - e
    thc = thc_max

    print('Conformal calibration done')


    print('...CALIBRATION DONE')
    print('BEGIN BINARY PREDICTIONS...')


    # Iterate throught the dataset #
    for i in numba.prange(S):

        probas = PROBAS[i]
        tar = SOL[i]

        sort = np.argsort(-probas)
        input = probas[sort]
        mask = np.where(input > 0.0)[0] #clip species with null probability
        input = input[mask]

        K, _ = max_func(input, func, args)
        pred = np.zeros(N).astype(np.intp)
        pred[sort[:K]] = 1
        U = score(pred,tar,func, args)

        topk = np.zeros(N)
        topk[sort[:Ktopk]] = 1
        U_topk = score(topk,tar,func, args)
    
        K0_5 = threshold(input, th = 0.5)
        n0_5 = np.zeros(N)
        n0_5[sort[:K0_5]] = 1
        U_0_5 = score(n0_5,tar,func, args)

        Kt = threshold(input, th)
        nt = np.zeros(N)
        nt[sort[:Kt]] = 1
        U_t = score(nt,tar,func, args)

        nf = probas > thf
        Kf = np.sum(nf)
        U_f = score(nf,tar,func, args)

        nc = probas > thc
        C = np.sum(nc)
        U_c = score(nc,tar,func, args)

        Ksum = sum_th(input)
        nsum = np.zeros(N)
        nsum[sort[:Ksum]] = 1
        U_sum = score(nsum,tar,func, args)

        KLIST[i] = np.array([Ktopk, Kt, Kf,C, K0_5, Ksum , K, np.sum(tar)])
        SCORE[i] = np.array([U_topk, U_t, U_f, U_c, U_0_5, U_sum, U])
        OUTPUT.append(sort[:Kt])

    print("Top K        :" , np.mean(SCORE[:,0]))
    print("Th t         :" , np.mean(SCORE[:,1]))
    print("Th t_f       :" , np.mean(SCORE[:,2]))
    print("C_opti       :" , np.mean(SCORE[:,3]))
    print("Th t_0.5     :" , np.mean(SCORE[:,4]))
    print("Sum          :" , np.mean(SCORE[:,5]))
    print("MaxExp       :" , np.mean(SCORE[:,6]))

    return OUTPUT, KLIST,SCORE


## APPLY DECISION FUNCTIONS TO EACH SITES ##
@numba.njit(fastmath=True, nogil = True)
def iterate_maxexp(SOL, PROBAS, func, args, max_func):
    S = len(SOL)
    N = len(SOL[0])


    SCORE = np.zeros(S)
    KLIST = np.zeros((S,2))

    OUTPUT = []

    # Iterate throught the dataset #
    for i in numba.prange(S):

        probas = PROBAS[i]
        tar = SOL[i]
        
        sort = np.argsort(-probas)
        input = probas[sort]
        mask = np.where(input > 0.0)[0] #clip species with null probability
        input = input[mask]

        K, _ = max_func(input, func, args)
        pred = np.zeros(N).astype(np.intp)
        pred[sort[:K]] = 1
        U = score(pred,tar,func, args)


        KLIST[i] = [K, np.sum(tar)]
        SCORE[i] = U

        OUTPUT.append(sort[:K])

        K0 = KLIST[:,0]/np.mean(KLIST[:,0])
        K1 = KLIST[:,1]/np.mean(KLIST[:,1])

    print("MaxExp :" , 1 - np.mean((K0 - K1)**2)/np.var(K1))

    return OUTPUT, KLIST,SCORE


## APPLY DECISION FUNCTIONS TO EACH SITES ##
@numba.njit(fastmath=True, nogil = True)
def iterate_t(SOL, PROBAS, P_CALIB, T_CALIB, func, args, max_func):
    S = len(SOL)
    N = len(SOL[0])
    ST = len(T_CALIB)


    SCORE = np.zeros((N, 7))
    KLIST = np.zeros((N, 8))

    OUTPUT = []

    # Calibrate strategies on the given data #

    print('BEGIN CALIBRATION...')
    # Global threshold
    e = 0.01
    th = 1
    thmax = 1
    Uth_max = -np.inf
    while th >= 0:
        Ut_th = 0.
        for j in range(N):
            probas = P_CALIB[:,j]
            tar = T_CALIB[:,j]
            sort = np.argsort(-probas)
            mask = np.where(probas > 0)[0]
            input = probas[sort][mask]
            Kth = threshold(input, th = th)
            nth = np.zeros(ST)
            nth[sort[:Kth]] = 1
            Ut_th += score(nth,tar,func, args)
        if Ut_th >= Uth_max:
            Uth_max = Ut_th
            thmax = th
        th = th - e
    
    th = thmax
    print('Global threshold done')

    # Species threshold
    e = 0.01
    ths = np.ones(N)
    for j in numba.prange(N):
        thsmax = 1
        Uth_max = -np.inf
        while ths[j] >= 0:
            probas = P_CALIB[:,j]
            tar = T_CALIB[:,j]
            nth = probas > ths[j]
            Ut_th = score(nth,tar,func, args)
            if Ut_th >= Uth_max:
                Uth_max = Ut_th
                thsmax = ths[j]
            ths[j] = ths[j] - e
        
        ths[j] = thsmax
    print('Species threshold done')

    # Frequency threshold
    thf = np.ones(N)
    for j in range(N):
        per = 100 - np.mean(T_CALIB[:,j])*100
        if per == 100 :
            thf[j] = 1
        else:
            thf[j] = np.percentile(P_CALIB[:,j], per)
    print('Frequency threshold done')

    # Conformal threshold
    e = 0.01
    c = 1 -e
    thc_max = np.ones(N)
    Uc_max = -np.inf
    thc = np.ones(N)
    while c >= 0:
        Uc = 0.
        for j in range(N):
            occ = np.where(T_CALIB[:,j] == 1)[0]
            if len(occ) > 0:
                thc[j] = np.percentile(P_CALIB[occ,j], int(c*100))

        for j in range(N):
            probas = P_CALIB[:,j]
            tar = T_CALIB[:,j]
            nc = probas > thc[j]
            Uc += score(nc,tar,func, args)
        if Uc > Uc_max:
            Uc_max = Uc
            thc_max = thc.copy()
        c = c - e
    thc = thc_max

    print('Conformal calibration done')


    print('...CALIBRATION DONE')
    print('BEGIN BINARY PREDICTIONS...')


    # Iterate throught the dataset #
    for j in numba.prange(N):

        probas = PROBAS[:,j]
        tar = SOL[:,j]

        sort = np.argsort(-probas)
        input = probas[sort]
        mask = np.where(input > 0.0)[0] #clip species with null probability
        input = input[mask]

        K, _ = max_func(input, func, args)
        pred = np.zeros(S).astype(np.intp)
        pred[sort[:K]] = 1
        U = score(pred,tar,func, args)


        Ks = threshold(input, th = ths[j])
        ns = np.zeros(S)
        ns[sort[:Ks]] = 1
        U_s = score(ns,tar,func, args)


        K0_5 = threshold(input, th = 0.5)
        n0_5 = np.zeros(S)
        n0_5[sort[:K0_5]] = 1
        U_0_5 = score(n0_5,tar,func, args)

        Kt = threshold(input, th)
        nt = np.zeros(S)
        nt[sort[:Kt]] = 1
        U_t = score(nt,tar,func, args)

        nf = probas > thf[j]
        Kf = np.sum(nf)
        U_f = score(nf,tar,func, args)

        nc = probas > thc[j]
        C = np.sum(nc)
        U_c = score(nc,tar,func, args)

        Ksum = sum_th(input)
        nsum = np.zeros(S)
        nsum[sort[:Ksum]] = 1
        U_sum = score(nsum,tar,func, args)

        KLIST[j] = np.array([Ks, Kt, Kf,C, K0_5, Ksum , K, np.sum(tar)])
        SCORE[j] = np.array([U_s, U_t, U_f, U_c, U_0_5, U_sum, U])
        OUTPUT.append(sort[:Ks])

    print("Th t_s       :" , np.mean(SCORE[:,0]))
    print("Th t         :" , np.mean(SCORE[:,1]))
    print("Th t_f       :" , np.mean(SCORE[:,2]))
    print("C_opti       :" , np.mean(SCORE[:,3]))
    print("Th t_0.5     :" , np.mean(SCORE[:,4]))
    print("Sum          :" , np.mean(SCORE[:,5]))
    print("MaxExp       :" , np.mean(SCORE[:,6]))

    return OUTPUT, KLIST,SCORE

def main():

    ## CONFIGURATION FILE ##
    config = configparser.ConfigParser()
    config.read('parameters.ini')

    ## DEFINE FILE PATH ##
    sol_file =  config['File paths']['sol_file']
    pred_file = config['File paths']['pred_file']
    tcalib_file = config['File paths']['tcalib_file']
    pcalib_file = config['File paths']['pcalib_file'] 

    ## DEFINE PARAMETERS OF EXPERIMENT ##
    pval = float(config['Experiment']['prob_val'])

    if config['Experiment']['train_calib'].lower() == 'true' :
        train_calib = True
    elif config['Experiment']['train_calib'].lower() == 'false' :
        train_calib = False
    else : 
        raise ValueError("train_calib must be True or False")
    
    if config['Experiment']['predict_cal'].lower() == 'false' :
        predict_cal = False
    elif config['Experiment']['predict_cal'].lower() == 'true' :
        predict_cal = True
    else :
        raise ValueError("predict_cal must be True or False")

    if config['Experiment']['run_quad'].lower() == 'true' :
        quad = True
    elif config['Experiment']['run_quad'].lower() == 'false' :
        quad = False
    else :
        raise ValueError("run_quad must be True or False")
    
    if config['Experiment']['only_maxexp'].lower() == 'true' :
        only_maxexp = True
    elif config['Experiment']['only_maxexp'].lower() == 'false' :
        only_maxexp = False
    else :
        raise ValueError("only_maxexp must be True or False")
    
    if config['Experiment']['transpose'].lower() == 'true' :
        transpose = True
    elif config['Experiment']['transpose'].lower() == 'false' :
        transpose = False
    else :
        raise ValueError("transpose must be True or False")

    ## DEFINE UTILITY FUNCTION ##
    func = config['Utility function']['metric']
    param = config['Utility function']['param']

    if pval < 0 or pval > 1:
        raise ValueError("prob_val must be between 0 and 1")
    try :
        args = list(map(float, param.split(',')))
        if func == 'f_b':
            func = f_b
        elif func == 'jaccard':
            func = jaccard
        elif func == 'deltaSR2':
            func = deltaSR2
        elif func == 'deltaSR1':
            func = deltaSR1
        elif func == 'tss':
            func = tss
        else:
            raise ValueError(f"{func} metric must be first implemented")
        _ = func(1, 1, 1, 1, args) 
    except ValueError:
         raise ValueError(f"{func} metric must be first implemented")
    
    ## DEFINE MAXIMIZATION FUNCTION ##

    if quad:
        if func == jaccard:
            max_func = max_jaccard_n2
        elif func == f_b and args[0] == int(args[0]):
            max_func = max_f_b_n2
        else:
            raise ValueError("Quadratic resolution impossible or not implemented")
    else:
        max_func = max_func_n3


    ## LOAD DATA ##
    sol = p.read_csv(sol_file)
    probas = p.read_csv(pred_file, delimiter = ',')
    probas = probas.join(sol.set_index('surveyId'), on='surveyId')
    surveys = probas['surveyId'].to_numpy(dtype=str)
    sol = probas['speciesId'].to_numpy(dtype=str)
    PROBAS = probas.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)
    del probas
    
    pick = np.random.randint(0, len(sol), size = int(len(sol)*pval))

    if train_calib :
        tcalib  = p.read_csv(tcalib_file)
        pcalib  = p.read_csv(pcalib_file, delimiter = ',')
        pcalib = pcalib.join(tcalib.set_index('surveyId'), on='surveyId')
        tcalib = pcalib['speciesId'].to_numpy(dtype=str)
        P_CALIB = pcalib.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)
        s_calib = pcalib['surveyId']
        del pcalib

    else:
        tcalib = sol[pick]
        P_CALIB = PROBAS[pick]
        s_calib = surveys[pick]
    

    if predict_cal :
        sol = tcalib
        PROBAS = P_CALIB
        surveys = s_calib

    else :
        PROBAS = np.delete(PROBAS, pick, axis = 0)
        sol = np.delete(sol, pick, axis = 0)
        surveys = np.delete(surveys, pick, axis = 0)

    S = len(sol)
    ST = len(tcalib)
    N = len(PROBAS[0])

    SOL = np.zeros((S,N), dtype = np.intp)
    for i in range(S) :
        r_sol = sol[i].split(' ')
        if r_sol != ['nan']:
            for id in r_sol:
                SOL[i,int(id)] = 1

    T_CALIB = np.zeros((ST,N), dtype = np.intp)
    for i in range(ST) :
        t_p = tcalib[i].split(' ')
        if t_p != ['nan']:
            for id in t_p:
                T_CALIB[i,int(id)] = 1
    del tcalib


    if transpose :
        if only_maxexp:
            output, nb_species, score = iterate_maxexp(SOL.T, PROBAS.T, func, args, max_func)
        output, nb_species, score = iterate_t(SOL, PROBAS, P_CALIB, T_CALIB, func, args, max_func)
    elif only_maxexp:
        output, nb_species, score = iterate_maxexp(SOL, PROBAS, func, args, max_func)
    else:
        output, nb_species , score = iterate(SOL, PROBAS, P_CALIB, T_CALIB, func, args, max_func)




    p.DataFrame(
        score, 
        columns = ['MaxExp'] if only_maxexp else ['TopK', 'Th t', 'Th t_f', 'C_opti', 'Th t_0.5', 'Sum', 'MaxExp']
        ).to_csv("submissions/score_distrib.csv", index = False)

    if transpose:
        data_concatenated = ['']*S
        for id_spec in range(len(output)):
            spec = output[id_spec]
            for id_site in spec:
                if data_concatenated[id_site] == '':
                    data_concatenated[id_site] = str(id_spec)
                else:
                    data_concatenated[id_site] += ' ' + str(id_spec)

        p.DataFrame(
            nb_species, 
            columns = ['Th_t_s', 'Th t', 'Th t_f', 'C_opti', 'Th t_0.5', 'Sum', 'MaxExp', 'True'] 
            ).to_csv("submissions/nb_sites.csv", index = False)
    
    else :
        data_concatenated = [' '.join(map(str, row)) for row in output]
    

        p.DataFrame(
            nb_species, 
            columns = ["MaxExp", "True"] if only_maxexp else ['TopK', 'Th t', 'Th t_f', 'C_opti', 'Th t_0.5', 'Sum', 'MaxExp', 'True']
            ).to_csv("submissions/nb_species.csv", index = False)


    p.DataFrame(
        {'surveyId': surveys,
        'speciesId': data_concatenated,
        }).to_csv("submissions/binary_predictions.csv", index = False)
        
if __name__ == "__main__":
    main()