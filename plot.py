import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as p
import seaborn as sns
from scipy import stats
from sklearn.metrics import f1_score, r2_score,auc, precision_recall_curve, roc_curve

from score_maximisation import score

colors = ['#11999E', '#40514E', '#FFB22C']

def run_wmw_test():
    score = p.read_csv("submissions/score_distrib.csv")
    names = score.columns
    for i in range(1, len(score.columns)):
        wmw, pvalue = stats.mannwhitneyu(score.iloc[:,0], score.iloc[:,i], alternative='greater')
        print(f"{names[0]} vs {names[i]}: WMW = {wmw}, p-value = {pvalue}")


def statistic(x, y, axis):
    return np.mean(x, axis) - np.mean(y, axis)


def run_permutation_test():
    score = p.read_csv("submissions/score_distrib.csv")
    names = score.columns
    for i in range(0, len(score.columns)-1):
        res = stats.permutation_test((score.iloc[:,-1], score.iloc[:,i]), statistic = statistic, vectorized=True, alternative='greater')
        print(f"{names[-1]} vs {names[i]}: S = {res.statistic}, p-value = {res.pvalue}")


def plot_aucs():
    sol_file = 'rls_train_species.csv'
    pred_file = 'rls_train_probas_ww.csv'

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


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    rec_grid = np.linspace(0.0, 1.0, 1000)
    mean_pr = np.zeros_like(rec_grid)

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(S):
        if np.sum(SOL[i]) != 0 :
            pr, rec, _ = precision_recall_curve(SOL[i], PROBAS[i])
            mean_pr += np.interp(rec_grid, rec[::-1], pr[::-1])

            fpr,tpr,  _ = roc_curve(SOL[i], PROBAS[i])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)

    mean_pr /= S
    mean_tpr /= S

    print("mean PR-AUC:", auc(rec_grid, mean_pr))
    print("mean ROC-AUC:", auc(fpr_grid, mean_tpr))

    ax1.plot(fpr_grid, mean_tpr, c = colors[0], linewidth = 2, label='Train, ROC-AUC = {:.2f}'.format(auc(fpr_grid, mean_tpr)))
    ax2.plot(rec_grid, mean_pr, c = colors[0], linewidth = 2, label='Train, PR-AUC = {:.2f}'.format(auc(rec_grid, mean_pr)))
    ax1.plot((0,1),(0,1), c = 'black', linewidth = 0.5)


    sol_file = 'rls_test_species.csv'
    pred_file = 'rls_test_probas_ww.csv'

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
    

    rec_grid = np.linspace(0.0, 1.0, 1000)
    mean_pr = np.zeros_like(rec_grid)

    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)


    for i in range(S):
        if np.sum(SOL[i]) != 0 :
            pr, rec, _ = precision_recall_curve(SOL[i], PROBAS[i])
            mean_pr += np.interp(rec_grid, rec[::-1], pr[::-1])

            fpr,tpr,  _ = roc_curve(SOL[i], PROBAS[i])
            mean_tpr += np.interp(fpr_grid, fpr, tpr)

    mean_pr /= S
    mean_tpr /= S

    print("mean PR-AUC:", auc(rec_grid, mean_pr))
    print("mean ROC-AUC:", auc(fpr_grid, mean_tpr))

    ax1.plot(fpr_grid, mean_tpr, c = colors[1], linewidth = 2, label='Test, ROC-AUC = {:.2f}'.format(auc(fpr_grid, mean_tpr)))
    ax2.plot(rec_grid, mean_pr, c = colors[1], linewidth = 2, label='Test, PR-AUC = {:.2f}'.format(auc(rec_grid, mean_pr)))

    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid()
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.set_title("PR Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid()
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def plot_score_distribution():
    title="Score Distribution"
    xlabel="Decision functions"
    ylabel="Score"
    score  = p.read_csv("submissions/score_distrib.csv")

    plt.figure(figsize=(10, 6))
    sns.violinplot(data= score, palette="muted")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def plot_calibration_curve():
    title="Calibration Curve"
    xlabel="Predicted Probability"
    ylabel="Fraction of Positives"

    sol_file = 'data/rls_train_species.csv'
    pred_file = 'data/rls_train_probas.csv'

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
    plt.show()


def plot_species_richness():
    title="Species richness"
    xlabel="True species richness"
    ylabel="Predicted species richness"

    sol_file = 'data/rls_test_species.csv'
    probas_file = 'data/rls_test_probas.csv'

    sol_file = 'data/cleaned_GeoLifeCLEF_species.csv'
    probas_file = 'data/cleaned_GeoLifeCLEF_probas.csv'

    # sol_file = 'data/birds_study_species.csv'
    # probas_file = 'data/birds_study_probas.csv'

    sol = p.read_csv(sol_file)
    probas = p.read_csv(probas_file)
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

    X = np.sum(SOL, axis = 1)
    Y = np.sum(PROBAS, axis = 1)
    Var = Y - np.sum(PROBAS**2, axis = 1)

    M = max(max(X), max(Y))
    slope, intercept, r_value, _, _ = stats.linregress(X, Y)
    regression_line = slope * np.array([0,M]) + intercept
    print("R2 :", r_value**2)
    plt.figure(figsize=(10, 10))
    plt.plot((0,M),(0,M), c = 'gray', linewidth = 1)
    plt.plot([0,M], regression_line, c= colors[0], alpha = 0.5, linewidth=2, label=f"Regression line (r={r_value:.2f})")
    # for  i in range(S):
    #     plt.plot([X[i], X[i]], [Y[i] - np.sqrt(20*Var[i]), Y[i] + np.sqrt(20*Var[i])], c = colors[0], alpha = 0.3, linewidth = 0.5)
    plt.scatter(X,Y, c = colors[0], s = 10, alpha = 0.5 )
    plt.grid(linewidth = 1)
    plt.xlim(0, M)
    plt.ylim(0, M)
    plt.show()


def plot_pred_sr():

    pred_file = 'submissions/predictions_rls_F1.csv'
    probas_file = 'data/rls_test_probas.csv'
    spec_file = 'data/rls_test_species.csv'


    pred_file = 'submissions/predictions_GLC_F1.csv'
    probas_file = 'data/GeoLifeCLEF_probas.csv'
    spec_file = 'data/GeoLifeCLEF_species.csv'

    # pred_file = 'submissions/predictions_birds_TSS.csv'
    # probas_file = 'data/birds_study_probas.csv'
    # spec_file = 'data/birds_study_species.csv'


    pred_dt = p.read_csv(pred_file)
    probas = p.read_csv(probas_file)

    species = p.read_csv(spec_file)
    surveys = pred_dt['surveyId']
    species = species.merge(surveys, how = 'right', on='surveyId')['speciesId'].to_numpy(dtype=str)
    probas = p.read_csv(probas_file)
    probas = probas.merge(surveys, how = 'right', on='surveyId')
    pred_dt = pred_dt['speciesId'].to_numpy(dtype=str)

    PROBAS = probas.drop(columns = ['surveyId']).to_numpy(dtype=np.float32)
    Y = []
    Xt = []
    S,N = PROBAS.shape
    for i in range(S) :
        Y.append(len(pred_dt[i].split(' ')))
        Xt.append(len(species[i].split(' ')))

    X = np.sum(PROBAS, axis = 1)

    
    plt.figure(figsize=(10, 10))
    print(r2_score(X, Y))
    print(r2_score(Xt, Y))

    X = np.log(X)/np.log(10)
    Y = np.log(np.round(Y,0))/np.log(10)
    B = max(max(X), max(Y))
    sns.scatterplot(x=X, y=Y, s=5, color=".15")
    sns.histplot(x=X, y=Y, bins=40, pthresh=.1, cmap="mako")
    sns.kdeplot(x=X, y=Y, levels=5, color="w", linewidths=1)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted set size versus expected richness", fontsize = 18)
    plt.xlabel("Expected species richness (log scale)", fontsize = 14)
    plt.ylabel("Predicted set size (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(0, B)
    plt.ylim(0, B)
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    X = np.log(Xt)/np.log(10)
    B = max(max(X), max(Y))
    sns.scatterplot(x=X, y=Y, s=5, color=".15")
    sns.histplot(x=X, y=Y, bins=40, pthresh=.1, cmap="mako")
    sns.kdeplot(x=X, y=Y, levels=5, color="w", linewidths=1)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])
    plt.title("Predited set size versus true richness", fontsize = 18)
    plt.xlabel("True species richness (log scale)", fontsize = 14)
    plt.ylabel("Predicted set size (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(0, B)
    plt.ylim(0, B)
    plt.grid(linewidth = 1)

    plt.show()



def plot_prev():


    # pred_file = 'submissions/predictions_rls_F1.csv'
    # pred_file2 = 'submissions/predictions_rls_F2.csv'
    # probas_file = 'data/rls_test_probas.csv'
    # spec_file = 'data/rls_test_species.csv'
    # train_file = 'data/rls_train_species.csv'
    # calib_file = 'data/rls_train_probas.csv'
    # pred_tr_file = 'submissions/predictions_rls_train_F1.csv'
    # pred_tr_file2 = 'submissions/predictions_rls_train_F2.csv'


    # pred_file = 'submissions/predictions_GLC_F1.csv'
    # pred_file2 = 'submissions/predictions_GLC_F2.csv'
    # probas_file = 'data/cleaned_GeoLifeCLEF_probas.csv'
    # spec_file = 'data/cleaned_GeoLifeCLEF_species.csv'
    # train_file = 'data/cleaned_GeoLifeCLEF_tcalib.csv'
    # calib_file = 'data/cleaned_GeoLifeCLEF_pcalib.csv'
    # pred_tr_file = 'submissions/pred_GLC_train_F1.csv'
    # pred_tr_file2 = 'submissions/pred_GLC_train_F2.csv'

    pred_file = 'submissions/predictions_birds_F1.csv'
    pred_file2 = 'submissions/predictions_birds_F2.csv'
    probas_file = 'data/birds_study_probas.csv'
    spec_file = 'data/birds_study_species.csv'
    train_file = 'data/birds_study_species.csv'
    calib_file = 'data/birds_study_calib.csv'
    pred_tr_file = 'submissions/pred_birds_train_F1.csv'
    pred_tr_file2 = 'submissions/pred_birds_train_F2.csv'

    pred_dt = p.read_csv(pred_file)
    pred_dt2 = p.read_csv(pred_file2)['speciesId'].to_numpy(dtype=str)
    probas = p.read_csv(probas_file)

    species = p.read_csv(spec_file)
    surveys = pred_dt['surveyId']
    species = species.merge(surveys, how = 'right', on='surveyId')['speciesId'].to_numpy(dtype=str)
    probas = p.read_csv(probas_file)
    probas = probas.merge(surveys, how = 'right', on='surveyId')

    pred_dt = pred_dt['speciesId'].to_numpy(dtype=str)
    PROBAS = probas.drop(columns = ['surveyId']).to_numpy(dtype=np.float32)

    tr_species = p.read_csv(train_file)
    tr_surveys = tr_species['surveyId']
    tr_species = tr_species['speciesId'].to_numpy(dtype=str)
    tr_probas = p.read_csv(calib_file).merge(tr_surveys, how = 'right', on='surveyId')
    CALIB = tr_probas.drop(columns = ['surveyId']).to_numpy(dtype=np.float32)



    S,N = PROBAS.shape
    Str,_ = CALIB.shape
    X = np.zeros(N)
    Y = np.zeros(N)
    Y2 = np.zeros(N)
    T = np.zeros(N)
    for i in range(S) :
        r_sol = species[i].split(' ')
        for id in r_sol:
            if r_sol != ['nan']:
                X[int(id)] += 1
        r_sol = pred_dt[i].split(' ')
        for id in r_sol:
            Y[int(id)] += 1
        r_sol = pred_dt2[i].split(' ')
        for id in r_sol:
            Y2[int(id)] += 1

    for i in range(Str) :
        r_sol = tr_species[i].split(' ')
        if r_sol != ['nan']:
            for id in r_sol:
                T[int(id)] += 1

    P = (np.sum(PROBAS, axis = 0)+1)/S
    Ptr = (np.sum(CALIB, axis = 0)+1)/Str

    X = (X+1)/S
    Y = (Y+1)/S
    Y2 = (Y2+1)/S
    T = (T+1)/Str

    
    Hue = 0.01/(T+0.01)
    Hue2 = np.log(X) - np.log(T)
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


    pred_train = p.read_csv(pred_tr_file).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr = np.zeros(N)

    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        for id in r_sol:
            Ytr[int(id)] += 1

    Ytr = (Ytr+1) / Str

    plt.figure(figsize=(10, 10))
    B = max(max(T), max(Ytr))
    sns.scatterplot(x=T, y=Ytr, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted train prevalence F1 versus Train prevalence \n R2log : " + str(round(r2_score(np.log(T), np.log(Ytr)),2)) \
              + ", R2 : " + str(round(r2_score(T, Ytr),2)), fontsize = 18)
    plt.xlabel("Train prevalence (log scale)", fontsize = 14)
    plt.ylabel("Predicted prevalence (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)


    pred_train = p.read_csv(pred_tr_file2).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr2 = np.zeros(N)
    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        for id in r_sol:
            Ytr2[int(id)] += 1
    Ytr2 = (Ytr2+1) / Str

    plt.figure(figsize=(10, 10))
    B = max(max(T), max(Ytr2))
    sns.scatterplot(x=T, y=Ytr2, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted train prevalence F2 versus Train prevalence \n R2log : " + str(round(r2_score(np.log(T), np.log(Ytr2)),2)) \
              + ", R2 : " + str(round(r2_score(T, Ytr2),2)), fontsize = 18)
    plt.xlabel("Train prevalence (log scale)", fontsize = 14)
    plt.ylabel("Predicted prevalence (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)


    plt.figure(figsize=(10, 10))
    B = max(max(Ytr), max(Ytr2))
    sns.scatterplot(x=Ytr, y=Ytr2, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted train prevalence F2 versus Predicted train prevalence F1", fontsize = 18)
    plt.xlabel("Predicted train prevalence F1 (log scale)", fontsize = 14)
    plt.ylabel("Predicted train prevalence F2 (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)


    plt.figure(figsize=(10, 10))
    B = max(max(Y), max(Y2))
    sns.scatterplot(x=Y, y=Y2, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted prevalence F2 versus Predicted prevalence F1", fontsize = 18)
    plt.xlabel("Predicted prevalence F1 (log scale)", fontsize = 14)
    plt.ylabel("Predicted prevalence F2 (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(X), max(Y))
    sns.scatterplot(x=X, y=Y, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted prevalence F1 versus True prevalence \n R2log : " + str(round(r2_score(np.log(X), np.log(Y)),2)) \
              + ", R2 : " + str(round(r2_score(X, Y),2)), fontsize = 18)
    plt.xlabel("True prevalence (log scale)", fontsize = 14)
    plt.ylabel("Predicted prevalence (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(X), max(Y2))
    sns.scatterplot(x=X, y=Y2, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted prevalence F2 versus True prevalence \n R2log : " + str(round(r2_score(np.log(X), np.log(Y2)),2)) \
              + ", R2 : " + str(round(r2_score(X, Y2),2)), fontsize = 18)
    plt.xlabel("True prevalence (log scale)", fontsize = 14)
    plt.ylabel("Predicted prevalence (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(X), max(P))
    sns.scatterplot(x=X, y=P, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Expected prevalence versus True prevalence " \
              + ", R2log : " + str(round(r2_score(np.log(X), np.log(P)),2)) \
              + ", R2 : " + str(round(r2_score(X, P),2)), fontsize = 18)
    plt.xlabel("True prevalence (log scale)", fontsize = 14)
    plt.ylabel("Expected prevalence (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(T), max(Ptr))
    sns.scatterplot(x=T, y=Ptr, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Expected train prevalence versus Train prevalence \n R2log : " + str(round(r2_score(np.log(T), np.log(Ptr)),2)) \
              + ", R2 : " + str(round(r2_score(T, Ptr),2)), fontsize = 18)
    plt.xlabel("Train prevalence (log scale)", fontsize = 14)
    plt.ylabel("Expected prevalence (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(X), max(T))
    sns.scatterplot(x=X, y=T, s=10, hue= Hue2, palette='icefire', legend=False, hue_norm= norm)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "True prevalence versus Train prevalence \n R2log : " + str(round(r2_score(np.log(X), np.log(T)),2)) \
              + ", R2 : " + str(round(r2_score(X, T),2)), fontsize = 18)
    plt.xlabel("True prevalence (log scale)", fontsize = 14)
    plt.ylabel("Train prevalence (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1/S, B)
    plt.ylim(1/S, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.show()


def plot_rich():

    # pred_file = 'submissions/predictions_rls_F1.csv'
    # pred_file2 = 'submissions/predictions_rls_F2.csv'
    # probas_file = 'data/rls_test_probas.csv'
    # spec_file = 'data/rls_test_species.csv'
    # train_file = 'data/rls_train_species.csv'
    # calib_file = 'data/rls_train_probas.csv'
    # pred_tr_file = 'submissions/predictions_rls_train_F1.csv'
    # pred_tr_file2 = 'submissions/predictions_rls_train_F2.csv'


    # pred_file = 'submissions/predictions_GLC_F1.csv'
    # pred_file2 = 'submissions/predictions_GLC_F2.csv'
    # probas_file = 'data/cleaned_GeoLifeCLEF_probas.csv'
    # spec_file = 'data/cleaned_GeoLifeCLEF_species.csv'
    # train_file = 'data/cleaned_GeoLifeCLEF_tcalib.csv'
    # calib_file = 'data/cleaned_GeoLifeCLEF_pcalib.csv'
    # pred_tr_file = 'submissions/pred_GLC_train_F1.csv'
    # pred_tr_file2 = 'submissions/pred_GLC_train_F2.csv'

    pred_file = 'submissions/predictions_birds_F1.csv'
    pred_file2 = 'submissions/predictions_birds_F2.csv'
    probas_file = 'data/birds_study_probas.csv'
    spec_file = 'data/birds_study_species.csv'
    train_file = 'data/birds_study_species.csv'
    calib_file = 'data/birds_study_calib.csv'
    pred_tr_file = 'submissions/pred_birds_train_F1.csv'
    pred_tr_file2 = 'submissions/pred_birds_train_F2.csv'

    pred_dt = p.read_csv(pred_file)
    pred_dt2 = p.read_csv(pred_file2)['speciesId'].to_numpy(dtype=str)
    probas = p.read_csv(probas_file)

    species = p.read_csv(spec_file)
    surveys = pred_dt['surveyId']
    species = species.merge(surveys, how = 'right', on='surveyId')['speciesId'].to_numpy(dtype=str)
    probas = p.read_csv(probas_file)
    probas = probas.merge(surveys, how = 'right', on='surveyId')

    pred_dt = pred_dt['speciesId'].to_numpy(dtype=str)
    PROBAS = probas.drop(columns = ['surveyId']).to_numpy(dtype=np.float32)

    tr_species = p.read_csv(train_file)
    tr_surveys = tr_species['surveyId']
    tr_species = tr_species['speciesId'].to_numpy(dtype=str)
    tr_probas = p.read_csv(calib_file).merge(tr_surveys, how = 'right', on='surveyId')
    CALIB = tr_probas.drop(columns = ['surveyId']).to_numpy(dtype=np.float32)


    S,N = PROBAS.shape
    Str,_ = CALIB.shape
    X = np.zeros(S)
    Y = np.zeros(S)
    Y2 = np.zeros(S)
    T = np.zeros(Str)

    for i in range(S) :
        r_sol = species[i].split(' ')
        if r_sol != ['nan']:
            X[i] = len(r_sol)

        r_sol = pred_dt[i].split(' ')
        Y[i] = len(r_sol)

        r_sol = pred_dt2[i].split(' ')
        Y2[i] = len(r_sol)

    for i in range(Str) :
        r_sol = tr_species[i].split(' ')
        if r_sol != ['nan']:
            T[i] = len(r_sol)

    P = (np.sum(PROBAS, axis = 1)+1)
    Ptr = (np.sum(CALIB, axis = 1)+1)



    X = (X+1)
    
    Y = (Y+1)
    Y2 = (Y2+1)
    T = (T+1)



    Hue = 0.01/(X/N+0.01)
    Hue2 = 0.01/(T/N+0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    pred_train = p.read_csv(pred_tr_file).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr = np.zeros(Str)
    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        Ytr[i] = len(r_sol)
    Ytr = (Ytr+1)



    plt.scatter(T, Ytr)
    plt.plot((0,max(T)),(0,max(T)), c = 'black', linewidth = 0.5)
    plt.gca().set_aspect('equal')
    plt.show()

    plt.figure(figsize=(10, 10))
    B = max(max(T), max(Ytr))
    sns.scatterplot(x=T, y=Ytr, s=10, hue= Hue2, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted train richness F1 versus Train richness \n R2log : " + str(round(r2_score(np.log(T), np.log(Ytr)),2)) \
              + ", R2 : " + str(round(r2_score(T, Ytr),2)), fontsize = 18)
    plt.xlabel("Train richness (log scale)", fontsize = 14)
    plt.ylabel("Predicted richness (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    pred_train = p.read_csv(pred_tr_file2).merge(tr_surveys, how = 'right', on='surveyId')["speciesId"].to_numpy(dtype=str)
    Ytr2 = np.zeros(Str)
    for i in range(Str) :
        r_sol = pred_train[i].split(' ')
        Ytr2[i] = len(r_sol)
    Ytr2 = (Ytr2+1)

    plt.figure(figsize=(10, 10))
    B = max(max(T), max(Ytr2))
    sns.scatterplot(x=T, y=Ytr2, s=10, hue= Hue2, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted train richness F2 versus Train richness \n R2log : " + str(round(r2_score(np.log(T), np.log(Ytr2)),2)) \
              + ", R2 : " + str(round(r2_score(T, Ytr2),2)), fontsize = 18)
    plt.xlabel("Train richness (log scale)", fontsize = 14)
    plt.ylabel("Predicted richness (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)



    plt.figure(figsize=(10, 10))
    B = max(max(Ytr), max(Ytr2))
    sns.scatterplot(x=Ytr, y=Ytr2, s=10, hue= Hue2, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted train richness F2 versus Predicted train richness F1", fontsize = 18)
    plt.xlabel("Predicted train richness F1 (log scale)", fontsize = 14)
    plt.ylabel("Predicted train richness F2 (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(Y), max(Y2))
    sns.scatterplot(x=Y, y=Y2, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted richness F2 versus Predicted richness F1", fontsize = 18)
    plt.xlabel("Predicted richness F1 (log scale)", fontsize = 14)
    plt.ylabel("Predicted richness F2 (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

 

    plt.figure(figsize=(10, 10))
    B = max(max(X), max(Y))
    sns.scatterplot(x=X, y=Y, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted richness F1 versus True richness \n R2log : " + str(round(r2_score(np.log(X), np.log(Y)),2)) \
              + ", R2 : " + str(round(r2_score(X, Y),2)), fontsize = 18)
    plt.xlabel("True richness (log scale)", fontsize = 14)
    plt.ylabel("Predicted richness (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(X), max(Y2))
    sns.scatterplot(x=X, y=Y2, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Predicted richness F2 versus True richness \n R2log : " + str(round(r2_score(np.log(X), np.log(Y2)),2)) \
              + ", R2 : " + str(round(r2_score(X, Y2),2)), fontsize = 18)
    plt.xlabel("True richness (log scale)", fontsize = 14)
    plt.ylabel("Predicted richness (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(X), max(P))
    sns.scatterplot(x=X, y=P, s=10, hue= Hue, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Expected richness versus True richness " \
              + ", R2log : " + str(round(r2_score(np.log(X), np.log(P)),2)) \
              + ", R2 : " + str(round(r2_score(X, P),2)), fontsize = 18)
    plt.xlabel("True richness (log scale)", fontsize = 14)
    plt.ylabel("Expected richness (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)

    plt.figure(figsize=(10, 10))
    B = max(max(T), max(Ptr))
    sns.scatterplot(x=T, y=Ptr, s=10, hue= Hue2, palette='mako', legend=False)
    sns.lineplot(x=[0, B], y=[0, B],color= colors[2])

    plt.title( "Expected train richness versus Train richness \n R2log : " + str(round(r2_score(np.log(T), np.log(Ptr)),2)) \
              + ", R2 : " + str(round(r2_score(T, Ptr),2)), fontsize = 18)
    plt.xlabel("Train richness (log scale)", fontsize = 14)
    plt.ylabel("Expected richness (log scale)", fontsize = 14)
    plt.gca().set_aspect('equal')
    plt.xlim(1, B)
    plt.ylim(1, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)
    plt.tight_layout()
    plt.show()



def map_species():
    sol_path = 'data/cleaned_GeoLifeCLEF_tcalib.csv'
    probas_path = 'data/cleaned_GeoLifeCLEF_pcalib.csv'
    path_base = 'submissions/pred_GLC_train_'
    loc_path = 'data/GeoLifeCLEF_metadata_train.csv'
    metrics = ['F1','F2']

    # sol_path = "data/rls_train_species.csv"
    # probas_path = "data/rls_train_probas.csv"
    # path_base = 'submissions/predictions_rls_train_'
    # loc_path = 'data/rls_metadata.csv'
    # metrics = ['F1','F2']

    list_pred = []
    for metric in metrics:
        fname = f"{path_base}{metric}.csv"
        list_pred.append(fname)
    
    
    probas = p.read_csv(probas_path)

    # species_name = probas.columns[species_idx+1]
    # print(species_name)

    species_name = "Carcharhinus amblyrhynchos"
    species_name = "Urolophus kapalensis"
    species_name = "Urolophus paucimaculatus"
    # species_name = "Trygonoptera imitata" 
    species_name = "Achoerodus viridis"

    # species_idx = probas.columns.to_list().index(species_name) -1
    species_idx = 302

    print(species_idx)

    surveys = probas['surveyId'].to_numpy(dtype=str)
    location_df = p.read_csv(loc_path)
    location_df = location_df.groupby('surveyId').first().reset_index()
    location_df = probas.merge(location_df,how = "left", on='surveyId')[['lon', 'lat']]
    probas = probas.iloc[:, species_idx+1].to_numpy(dtype=np.float32)

    lon  = location_df['lon'].to_numpy(dtype=np.float32)
    lat  = location_df['lat'].to_numpy(dtype=np.float32)

    m = Basemap(projection='merc',llcrnrlat= int(lat.min()),urcrnrlat= int(lat.max() +1 ),\
                llcrnrlon= int(lon.min()),urcrnrlon= int(lon.max() +1),lat_ts= 20, resolution='i')

    x, y = m(lon, lat)
    ## map species probabilities
    plt.figure()
    m.drawcoastlines(color='black', linewidth=0.5)
    m.drawmapboundary(fill_color='white') 
    m.fillcontinents(color='lightgray',lake_color='white', alpha=0.2)
    # m.fillcontinents(color='#13090f',lake_color='white')

    plt.title(f"Species: {species_name}", fontsize=16)
    order = np.argsort(probas)
    plt.scatter(x=x[order], y=y[order], c=np.log(probas[order]/(1- probas[order])), cmap='mako_r', s= 20)
    plt.colorbar(label='Probability logits')
    plt.tight_layout()
    plt.savefig(f'figures/map_probas.svg')


    grad = mpl.colormaps['mako_r'](np.linspace(0.4, 0.8, len(metrics)))

    ## map true species presence
    sol_df = p.read_csv(sol_path)
    presence = np.zeros(len(surveys), dtype=bool)
    for i in range(len(surveys)):
        spec = str(sol_df.iloc[i,1]).split(' ')
        if spec[0] != 'nan':
            if str(species_idx) in spec:
                presence[i] = True
    print(np.sum(presence))
    plt.figure()
    m = Basemap(projection='merc',llcrnrlat= int(lat.min()),urcrnrlat= int(lat.max() +1 ),\
                llcrnrlon= int(lon.min()),urcrnrlon= int(lon.max() +1),lat_ts= 20, resolution='i')
    m.drawcoastlines(color='black', linewidth=0.5)
    m.drawmapboundary(fill_color='white') 
    m.fillcontinents(color='lightgray',lake_color='white', alpha=0.2)
    plt.scatter(x=x[~presence], y=y[~presence], color = '#e2e2e2' , s=20)
    plt.scatter(x=x[presence], y=y[presence], color = grad[-1], s=20)
    plt.title(f"True Presence of {species_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/map_true.svg')



    ## map predictions

    plt.figure()
    m = Basemap(projection='merc',llcrnrlat= int(lat.min()),urcrnrlat= int(lat.max() +1 ),\
                llcrnrlon= int(lon.min()),urcrnrlon= int(lon.max() +1),lat_ts= 20, resolution='i')
    m.drawcoastlines(color='black', linewidth=0.5)
    m.drawmapboundary(fill_color='white') 
    m.fillcontinents(color='lightgray',lake_color='white', alpha=0.2)

    for idx in range(len(list_pred)):
        pred_file = list_pred[- idx - 1]
        pred_df = p.read_csv(pred_file)
        pred = np.zeros(len(surveys), dtype=bool)
        for i in range(len(surveys)):
            if surveys[i]== str(pred_df.iloc[i,0]):
                if str(species_idx) in pred_df.iloc[i,1].split(' '):
                    pred[i] = True


        tp = pred * presence
        fp = pred * ~presence
        fn = ~pred * presence
        print(np.sum(tp), np.sum(fp), np.sum(fn))
        if idx == 0:
            plt.scatter(x = x[~pred], y = y[~pred], color = '#e2e2e2', s= 20, label='Absence')

        if idx == 0:
            plt.scatter(x = x[pred], y = y[pred], color = colors[2], s=20, label=f'{metrics[-idx-1]} Predicted Presence')
        else:
            plt.scatter(x = x[pred], y = y[pred], color = grad[idx], s=20, label=f'{metrics[-idx-1]} Predicted Presence')
    plt.tight_layout()
    plt.savefig(f'figures/map_pred.svg')

    plt.show()



def map_iucn():

    sol_path = "data/rls_train_species.csv"
    iucn_file = "data/rls_traits.csv"
    probas_path = "data/rls_train_probas.csv"
    path_base = 'submissions/predictions_rls_train_'
    loc_path = 'data/rls_metadata.csv'
    metrics = ['F1','F5']

    list_pred = []
    for metric in metrics:
        fname = f"{path_base}{metric}.csv"
        list_pred.append(fname)


    iucn = p.read_csv(iucn_file)[['rls_species_name','IUCN_category']].to_numpy(dtype=str)
    status = {"LC":0, "NT":1, "VU":2, "EN":3, "CR":4, "EW":5, "EX":6}
    probas = p.read_csv(probas_path)
    spec_name = probas.columns[1:]    

    spec_iucn = np.ones_like(spec_name, dtype=int) * -1
    for i in range(len(spec_name)):
        index = np.where(iucn[:,0] == spec_name[i])[0]
        if len(index) == 0 :
            print(f"Species {spec_name[i]} not found")
        else:
            if iucn[index[0],1] in status.keys():
                spec_iucn[i] = status[iucn[index[0],1]]

    surveys = probas['surveyId'].to_numpy(dtype=str)
    location_df = p.read_csv(loc_path)
    location_df = location_df.groupby('surveyId').first().reset_index()
    location_df = probas.merge(location_df,how = "left", on='surveyId')[['lon', 'lat','site_code']]
    probas = probas.iloc[:, 1:].to_numpy(dtype=np.float32)


    # Group by site_code
    site_groups = location_df.groupby('site_code')
    site_lon = site_groups['lon'].mean().to_numpy(dtype=np.float32)
    site_lat = site_groups['lat'].mean().to_numpy(dtype=np.float32)
    site_codes = site_groups.groups.keys()
    print(len(site_codes))
    m = Basemap(projection='merc',llcrnrlat= int(site_lat.min()),urcrnrlat= int(site_lat.max() +1 ),\
                llcrnrlon= int(site_lon.min()),urcrnrlon= int(site_lon.max() +1),lat_ts= 20, resolution='i')

    x, y = m(site_lon, site_lat)

    grad = mpl.colormaps['mako_r'](np.linspace(0, 1, 5))

    ## map true species presence
    sol_df = p.read_csv(sol_path)["speciesId"]
    site_iucn = np.ones(len(site_codes), dtype=int) * -1
    for i, site in enumerate(site_codes):
        for spec in sol_df[location_df['site_code'] == site] :
            if str(spec) != 'nan':
                spec  = spec.split(' ')
                for s in spec:
                    site_iucn[i] = max(site_iucn[i], spec_iucn[int(s)])


    plt.figure()


    m.drawcoastlines(color='black', linewidth=0.5)
    m.drawmapboundary(fill_color='white') 
    m.fillcontinents(color='lightgray',lake_color='white', alpha=0.2)
    plt.scatter(x[site_iucn == -1], y[site_iucn == -1], color = grad[0], s=30)
    for i in range(5):
        plt.scatter(x[site_iucn == i], y[site_iucn == i], color = grad[i], s=30, label=list(status.keys())[i])
    # plt.title(f"Worst IUCN status observed", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/icun_true.svg')



    ## map predictions
    pred_iuc  = []
    for idx in range(len(list_pred)):

        plt.figure()
        m = Basemap(projection='merc',llcrnrlat= int(site_lat.min()),urcrnrlat= int(site_lat.max() +1 ),\
                    llcrnrlon= int(site_lon.min()),urcrnrlon= int(site_lon.max() +1),lat_ts= 20, resolution='i')

        m.drawcoastlines(color='black', linewidth=0.5)
        m.drawmapboundary(fill_color='white') 
        m.fillcontinents(color='lightgray',lake_color='white', alpha=0.2)

        pred_file = list_pred[idx]
        pred_df = p.read_csv(pred_file)["speciesId"]
        site_iucn = np.ones(len(site_codes), dtype=int) * -1
        for i, site in enumerate(site_codes):
            for spec in pred_df[location_df['site_code'] == site] :
                if str(spec) != 'nan':
                    spec  = spec.split(' ')
                    for s in spec:
                        site_iucn[i] = max(site_iucn[i], spec_iucn[int(s)])


        plt.scatter(x[site_iucn == -1], y[site_iucn == -1], color = grad[0], s=30)
        for i in range(5):
            plt.scatter(x[site_iucn == i], y[site_iucn == i], color = grad[i], s=30, label=list(status.keys())[i])
        # plt.title(f"Worst IUCN status predicted with {metrics[idx]}", fontsize=16)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'figures/icun_{metrics[idx]}.svg')
        pred_iuc.append(site_iucn)

    ## difference between predictions
    plt.figure()
    m = Basemap(projection='merc',llcrnrlat= int(site_lat.min()),urcrnrlat= int(site_lat.max() +1 ),\
                llcrnrlon= int(site_lon.min()),urcrnrlon= int(site_lon.max() +1),lat_ts= 20, resolution='i')
    m.drawcoastlines(color='black', linewidth=0.5)
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='lightgray',lake_color='white', alpha=0.2)
    site_iucn = pred_iuc[1] - pred_iuc[0]
    plt.scatter(x[site_iucn == 0], y[site_iucn == 0], color = '#e2e2e2', s=30, label='No status change')
    plt.scatter(x[site_iucn > 0], y[site_iucn > 0], color = colors[0], s=30, label='More threatened status with F2')
    plt.legend()

    # plt.title(f"Difference between predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/icun_diff.svg')

    plt.show()


def score_comp():

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 24,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    spec_file = 'data/cleaned_GeoLifeCLEF_species.csv'
    pred_file = 'submissions/pred_GLC_F1_t.csv'
    pred_file2 = 'submissions/predictions_GLC_F1.csv'
    probas_file = 'data/cleaned_GeoLifeCLEF_probas.csv'

    # spec_file = 'data/rls_test_species.csv'
    # pred_file = 'submissions/predictions_rls_F1_t.csv'
    # pred_file2 = 'submissions/predictions_rls_F1.csv'
    # probas_file = 'data/rls_test_probas.csv'

    pred_df = p.read_csv(pred_file)
    pred_df2 = p.read_csv(pred_file2)["speciesId"]

    surveys = pred_df['surveyId']
    pred_df = pred_df['speciesId']

    probas = p.read_csv(probas_file)
    probas = probas.merge(surveys, how = 'right', on='surveyId')
    probas = probas.iloc[:, 1:].to_numpy(dtype=np.float32)
    S,N = probas.shape


    species = p.read_csv(spec_file)
    species = species.merge(surveys, how = 'right', on='surveyId')
    species = species['speciesId'].to_numpy(dtype=str)

    T = np.zeros((S,N))

    for i in range(S) :
        r_sol = species[i]
        if str(r_sol) != 'nan':
            r_sol = r_sol.split(' ')
            for id in r_sol:
                T[i, int(id)] = 1

    T = T.T #
    S, N = N, S #

    Prev = (np.sum(T, axis = 0)+1)/S


    Hue = 0.01/(Prev+0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    S, N = N, S #
    Y1 = np.zeros((S,N))
    Y2 = np.zeros((S,N))
    for i in range(S) :
        r_sol = pred_df[i]
        if str(r_sol) != 'nan':
            r_sol = r_sol.split(' ')
            for id in r_sol:
                Y1[i, int(id)] = 1

        r_sol = pred_df2[i]
        if str(r_sol) != 'nan':
            r_sol = r_sol.split(' ')
            for id in r_sol:
                Y2[i, int(id)] = 1
    
    Y1 , Y2 = Y1.T, Y2.T #
    S, N = N, S #

    func = lambda y_true, y_pred : f1_score(y_true, y_pred, zero_division=0)

    S1 = np.zeros(N)
    S2 = np.zeros(N)
    for j in range(N):

        S1[j] = func(T[:, j], Y1[:, j])
        S2[j] = func(T[:, j], Y2[:, j])

    print(np.mean(S1), np.mean(S2))
    plt.figure(figsize=(10, 10))
    plt.scatter(x= Prev, y= S2 - S1, s=20, c= Hue, linewidth=0, alpha = 0.5, cmap ='mako_r')
    X = np.arange(1/S, max(Prev)+1/S, 1/S)
    Y = np.zeros_like(X)
    for i in range(len(X)):
        bin_mask = (Prev >= X[i]) * (Prev < X[i+1]) if i < len(X)-1 else (Prev >= X[i])
        if np.sum(bin_mask) > 0:
            Y[i] = np.mean(S2[bin_mask]- S1[bin_mask])
    

    plt.plot(X, Y, color = colors[0], linewidth=1)
    plt.axhline(0, color= colors[2])
    plt.xlim(1/S, max(Prev))
    plt.xscale('log')
    plt.ylim(-0.5, 0.5)
    plt.grid(linewidth = 1)
    plt.tight_layout()
    plt.show()

def plot_prev():

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 24,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    train_file = 'data/rls_train_species.csv'
    calib_file = 'data/rls_train_probas.csv'
    pred_tr_file = 'submissions/predictions_rls_train_F1.csv'
    pred_tr_file2 = 'submissions/predictions_rls_train_F2.csv'
    pred_tr_fileJ = 'submissions/predictions_rls_train_J.csv'
    pred_tr_fileTSS = 'submissions/predictions_rls_train_TSS.csv'

    train_file = 'data/cleaned_GeoLifeCLEF_tcalib.csv'
    calib_file = 'data/cleaned_GeoLifeCLEF_pcalib.csv'
    pred_tr_file = 'submissions/pred_GLC_train_F1.csv'
    pred_tr_file2 = 'submissions/pred_GLC_train_F2.csv'
    pred_tr_fileJ = 'submissions/pred_GLC_train_J.csv'
    pred_tr_fileTSS = 'submissions/pred_GLC_train_TSS.csv'


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

    # plt.title( "Predicted train prevalence F1 versus Train prevalence \n R2log : " + str(round(r2_score(np.log(T), np.log(Ytr1)),2)) \
            #   + ", R2 : " + str(round(r2_score(T, Ytr1),2)), fontsize = 18)
    plt.xlabel("Prevalence (log scale)",)
    plt.ylabel("Predicted prevalence (log scale)")
    plt.gca().set_aspect('equal')
    plt.xlim(1/Str, B)
    plt.ylim(1/Str, B)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(linewidth = 1)
    plt.tight_layout()
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
    # ax1.set_yscale('log')
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
    # ax2.set_yscale('log')
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
    # fig.suptitle("Change in predicted prevalence for different metric \n compared to F1 in function of species prevalence", fontsize = 18)
    plt.show()


#hill_numbers()
# map_iucn()
# score_comp()
map_species()
#plot_calibration_curve()
#plot_species_richness_all()
#plot_aucs()
#plot_species_richness()
#plot_pred_sr()
#plot_sr_fb()
# plot_score_distribution()
# plot_prev()
#plot_rich()
#run_permutation_test()