import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as p
import seaborn as sns
from scipy import stats
import os
from sklearn.metrics import r2_score


## \/ FUNCTION CALLS AT THE END \/ ##
 
colors = ['#11999E', '#40514E', '#FFB22C']

def statistic(x, y, axis):
    return np.mean(x, axis) - np.mean(y, axis)


def run_permutation_test():
    score = p.read_csv("predictions/score_distrib.csv")
    names = score.columns
    for i in range(0, len(score.columns)-1):
        res = stats.permutation_test((score.iloc[:,-1], score.iloc[:,i]), statistic = statistic, vectorized=True, alternative='greater')
        print(f"{names[-1]} vs {names[i]}: S = {res.statistic}, p-value = {res.pvalue}")


def plot_calibration_curve(case_study, subset = 'train', path = 'data/', ):
    title="Calibration Curve"
    xlabel="Predicted Probability"
    ylabel="Fraction of Positives"

    sol_file = f'{path}CS{case_study}_{subset}_species.csv'
    pred_file = f'{path}CS{case_study}_{subset}_probas.csv'

    sol = p.read_csv(sol_file)
    probas = p.read_csv(pred_file)
    probas = probas.join(sol.set_index('surveyId'), on='surveyId')
    probas = probas.dropna()
    sol = probas['speciesId'].to_numpy(dtype=str)

    PROBAS = probas.drop(columns = ['surveyId', 'speciesId']).to_numpy(dtype=np.float32)
    
    S,N = PROBAS.shape
    SOL = np.zeros((S,N), dtype = np.intp)
    for i in range(S) :
        r_sol = sol[i].split(' ')
        for id in r_sol:
            SOL[i,int(id)] = 1

    dp = 0.05
    bins = np.arange(0,1,dp)
    Y = []
    X = []
    sY = []
    for bin in bins:
        idx = np.where((PROBAS >= bin)*(PROBAS < bin+dp)) 
        if len(idx[0]!= 0) :
            Y.append(np.mean(SOL[idx]))
            sY.append(np.std(SOL[idx])/np.sqrt(len(idx[0])))
            X.append(np.mean(PROBAS[idx]))
    X, Y, sY = np.array(X), np.array(Y), np.array(sY)

    plt.figure(figsize=(10, 6))
    plt.scatter(X,Y, c = colors[0], s = 50 )
    plt.plot(X,Y, c = colors[0] , linewidth = 1)
    plt.plot(X,Y + 3*sY,'--',c = colors[0] , linewidth = 0.5)
    plt.plot(X,Y - 3*sY,'--',c = colors[0] , linewidth = 0.5)

    plt.plot((0,1),(0,1), c = 'black', linewidth = 0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.fill_between(X, Y + 3*sY, Y - 3*sY, color= colors[0], alpha=0.1)
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    if not os.path.exists('figures/'):
        os.makedirs('figures/')
    plt.savefig(f'figures/calibration_curve_CS{case_study}_{subset}.svg')


def plot_prev(case_study =1, data_path = 'data/'):

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 24,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    train_file = f'{data_path}CS{case_study}_train_species.csv'
    calib_file = f'{data_path}CS{case_study}_train_probas.csv'
    
    pred_tr_file = f'predictions/pred_CS{case_study}_F1.csv'
    pred_tr_file2 = f'predictions/pred_CS{case_study}_F2.csv'
    pred_tr_fileJ = f'predictions/pred_CS{case_study}_J.csv'
    pred_tr_fileTSS = f'predictions/pred_CS{case_study}_TSS.csv'

    tr_species = p.read_csv(train_file)
    tr_surveys = tr_species['surveyId']
    tr_species = tr_species['speciesId'].to_numpy(dtype=str)
    tr_probas = p.read_csv(calib_file).merge(tr_surveys, how = 'right', on='surveyId')
    CALIB = tr_probas.drop(columns = ['surveyId']).to_numpy(dtype=np.float32)


    Str,N = CALIB.shape
    T = np.zeros(N)

    for i in range(Str) :
        r_sol = tr_species[i].split(' ')
        if r_sol != ['nan']:
            for id in r_sol:
                T[int(id)] += 1

    Ptr = (np.sum(CALIB, axis = 0)+1)/Str

    T = (T+1)/Str

    
    Hue = 0.01/(T+0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    pred_train = p.read_csv(pred_tr_file).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr1 = np.zeros(N)

    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        if r_sol != ['nan']:
            for id in r_sol:
                Ytr1[int(id)] += 1


    Ytr1 = (Ytr1+1) / Str

    Hue2 = 0.01/(Ytr1+0.01)

    plt.figure(figsize=(10, 10))
    B = max(max(T), max(Ytr1))
    sns.scatterplot(x=T, y=Ytr1, s=20, color = "#39366b", legend=False, linewidth=0, alpha = 0.5)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.xlabel("Prevalence (log scale)",)
    plt.ylabel("Predicted prevalence (log scale)")
    plt.gca().set_aspect('equal')
    plt.xlim(1/Str, B)
    plt.ylim(1/Str, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)
    plt.tight_layout()
    if not os.path.exists('figures/'):
        os.makedirs('figures/')
    plt.savefig('figures/prev_f1.svg')



    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,10), sharex=True)
    pred_train = p.read_csv(pred_tr_file2).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr = np.zeros(N)

    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        if r_sol != ['nan']:
            for id in r_sol:
                Ytr[int(id)] += 1

    Ytr = (Ytr+1) / Str



    B = max(max(Ytr1), max(Ytr))
    ax1.scatter(x= T, y= Ytr/Ytr1, s=20, c= Hue2, cmap='mako_r', linewidth=0, alpha = 0.5)
    ax1.plot([1, 0], [1, B],color= colors[2])
    ax1.set_xlim(1/Str/1.1, B*1.1)
    ax1.set_ylabel("F2", fontsize = 24)
    vals = ax1.get_yticks()
    ax1.set_yticks(vals)
    ax1.set_yticklabels([r"+{:.0f}\%".format((x-1)*100) for x in vals])
    ax1.set_ylim(0.5, 8)

    ax1.set_xscale('log')
    ax1.grid(linewidth = 1)


    pred_train = p.read_csv(pred_tr_fileJ).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr = np.zeros(N)

    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        if r_sol != ['nan']:
            for id in r_sol:
                Ytr[int(id)] += 1

    Ytr = (Ytr+1) / Str

    B = max(max(Ytr1), max(Ytr))
    ax2.scatter(T, Ytr/Ytr1, s=20, c= Hue2, cmap='mako_r', linewidth=0, alpha = 0.5)
    ax2.plot([1, 0], [1, B],color= colors[2])

    ax2.set_xlim(1/Str/1.1, B*1.1)
    ax2.set_ylabel("J", fontsize = 24)
    ax2.set_ylim(0.5, 1.1)
    vals = ax2.get_yticks()
    ax2.set_yticks(vals)
    ax2.set_yticklabels([r'-{:.0f}\%'.format((1-x)*100) for x in vals])

    ax2.set_xscale('log')
    ax2.grid(linewidth = 1)

    pred_train = p.read_csv(pred_tr_fileTSS).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr = np.zeros(N)

    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        if r_sol != ['nan']:
            for id in r_sol:
                Ytr[int(id)] += 1

    Ytr = (Ytr+1) / Str

    B = max(max(Ytr1), max(Ytr))
    ax3.scatter(x= T, y= Ytr/Ytr1, s=20, c= Hue2, cmap='mako_r', alpha = 0.5, linewidth=0)
    ax3.plot([1, 0], [1, B],color= colors[2])

    ax3.set_xlim(1/Str/1.1, B*1.1)
    ax3.set_ylabel("TSS", fontsize = 24)    
    ax3.set_ylim(1, 1000)

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(linewidth = 1)
    ax3.set_xlabel("Prevalence (log scale)", fontsize = 24)
    vals = ax3.get_yticks()
    ax3.set_yticks(vals)
    ax3.set_yticklabels([r'x{:}'.format(x) for x in vals])
    plt.tight_layout()
    plt.savefig('figures/prev_comp.svg')
    plt.show()



## RUN PERMUTATION TEST OF PREVIOUS RUN ##

#run_permutation_test()


## PLOT CALIBRATION CURVE ##

subsets = ['train', 'test']
case_studies = [1, 2, 3]
for cs in case_studies:
    for subset in subsets:
        plot_calibration_curve(case_study=cs, subset = subset)
plt.show()



## PLOT PREVALENCE FIGURES ##
'''Need to have predictions files of the desired study in the predictions/ folder in proper format for all metrics studied.
Case study 1 with MaxExp on train dataset already included in the repo (Figures 2 in the paper)'''
plot_prev(case_study=1)