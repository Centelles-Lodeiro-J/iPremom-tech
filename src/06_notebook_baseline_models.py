"""Notebook 06 — Baseline Models (final, M2 lifelines-only)
Outputs to outputs/notebook_06/
"""
import argparse, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
plt.rcParams.update({"figure.dpi":150,"figure.facecolor":"white","axes.facecolor":"#f8f8f8","axes.spines.top":False,"axes.spines.right":False})
CLASSIFICATION_MODELS=[('M1a_overall_survival','M1a overall survival','binary','AUC-ROC'),('M1b_cancer_specific_survival','M1b cancer-specific survival','binary','AUC-ROC'),('M3_pam50_subtype','M3 PAM50 subtype','multi','Macro-F1'),('M4_histologic_grade','M4 histologic grade','ordinal','QW-Kappa')]
M2_MODELS=[('M2a_overall_survival_cox','M2a overall survival Cox'),('M2b_cancer_specific_cox','M2b cancer-specific Cox')]

def save(fig,path): path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path,bbox_inches='tight'); plt.close(fig); print(f'  Saved: {path.name}')

def load_split(splits_dir, key):
    d=splits_dir/key
    return pd.read_csv(d/'X_train.csv'), pd.read_csv(d/'X_test.csv'), pd.read_csv(d/'y_train.csv'), pd.read_csv(d/'y_test.csv')

def make_lr(): return LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,solver='lbfgs',random_state=42)

def score_task(clf,X,y,task):
    yp=clf.predict(X)
    if task=='binary': return roc_auc_score(y, clf.predict_proba(X)[:,1])
    if task=='multi': return f1_score(y, yp, average='macro', zero_division=0)
    return cohen_kappa_score(y, yp, weights='quadratic')

def fig_accuracy_illusion(splits_dir,out):
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    models=[('M1a_overall_survival','M1a'),('M1b_cancer_specific_survival','M1b'),('M3_pam50_subtype','M3'),('M4_histologic_grade','M4')]
    mins=[]; accs=[]; labs=[]
    for k,l in models:
        y=load_split(splits_dir,k)[2].iloc[:,0]
        vc=y.value_counts(normalize=True)
        mins.append(float(vc.min())); accs.append(float((y==y.mode()[0]).mean())); labs.append(l)
    x=np.arange(len(labs)); w=0.35
    axes[0].bar(x-w/2, accs, width=w, color='#e74c3c', alpha=0.8, label='Naive accuracy')
    axes[0].bar(x+w/2, mins, width=w, color='#3498db', alpha=0.8, label='Minority rate')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labs); axes[0].legend(fontsize=8); axes[0].set_title('Accuracy illusion')
    ytr=load_split(splits_dir,'M1b_cancer_specific_survival')[2].iloc[:,0]; yte=load_split(splits_dir,'M1b_cancer_specific_survival')[3].iloc[:,0]
    maj=int(ytr.mode()[0]); ypred=np.full(len(yte), maj); cm=confusion_matrix(yte, ypred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[1]); axes[1].set_title('M1b majority-class confusion matrix')
    yprob=np.full(len(yte), float(ytr.mean()))
    metrics={'Accuracy':accuracy_score(yte, ypred), 'AUC-ROC':roc_auc_score(yte, yprob), 'PR-AUC':average_precision_score(yte, yprob)}
    axes[2].axis('off'); axes[2].text(0.0,0.95,'Naive predictor metrics (M1b)',fontsize=11,fontweight='bold')
    y=0.75
    for k,v in metrics.items(): axes[2].text(0.0,y,f'{k}: {v:.3f}',fontsize=10); y-=0.12
    fig.tight_layout(); save(fig,out/'23_accuracy_illusion.png')

def fig_classification_baselines(splits_dir,out):
    fig,axes=plt.subplots(2,2,figsize=(16,12)); axes=axes.flatten()
    for ax,(key,label,task,metric) in zip(axes, CLASSIFICATION_MODELS):
        Xtr,Xte,ytr_raw,yte_raw=load_split(splits_dir,key)
        ytr=ytr_raw.iloc[:,0]; yte=yte_raw.iloc[:,0]
        clin=[c for c in Xtr.columns if not c.startswith('gene_programme')]
        lr=make_lr(); lr.fit(Xtr[clin], ytr)
        score=score_task(lr, Xte[clin], yte, task)
        ax.axis('off'); ax.text(0.02,0.9,label,fontweight='bold',fontsize=11); ax.text(0.02,0.75,f'Clinical-only baseline {metric}: {score:.3f}',fontsize=10); ax.text(0.02,0.60,f'n train={len(Xtr):,}  n test={len(Xte):,}',fontsize=10); ax.text(0.02,0.45,f'Clinical features used: {len(clin)}',fontsize=10)
    fig.tight_layout(); save(fig,out/'24_26_classification_baselines.png')

def _prepare_cox_features(Xtr, Xte):
    Xtr=Xtr.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    Xte=Xte.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    med=Xtr.median().fillna(0.0)
    Xtr=Xtr.fillna(med); Xte=Xte.fillna(med)
    nunique=Xtr.nunique(dropna=False); keep=nunique[nunique>1].index.tolist(); Xtr=Xtr[keep]; Xte=Xte[keep]
    if Xtr.shape[1]==0: raise ValueError('No usable Cox features remain.')
    scaler=StandardScaler(); Xtr=pd.DataFrame(scaler.fit_transform(Xtr),columns=Xtr.columns,index=Xtr.index); Xte=pd.DataFrame(scaler.transform(Xte),columns=Xte.columns,index=Xte.index)
    return Xtr, Xte

def _fit_cox_stable(dftr):
    from lifelines import CoxPHFitter
    last=None
    for pen in [0.1,1.0,3.0,10.0]:
        try:
            cph=CoxPHFitter(penalizer=pen,l1_ratio=0.0); cph.fit(dftr, duration_col='time', event_col='event', show_progress=False); return cph, pen
        except Exception as e:
            last=e
    raise last

def fig_m2_lifelines(splits_dir,out):
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,(key,label) in zip(axes,M2_MODELS):
        Xtr,Xte,ytr,yte=load_split(splits_dir,key)
        clin=[c for c in Xtr.columns if not c.startswith('gene_programme')]
        Xtr2,Xte2=_prepare_cox_features(Xtr[clin], Xte[clin])
        dftr=Xtr2.copy(); dftr['time']=ytr['time'].values; dftr['event']=ytr['event'].values
        dfte=Xte2.copy(); dfte['time']=yte['time'].values; dfte['event']=yte['event'].values
        cph,pen=_fit_cox_stable(dftr)
        cidx=float(cph.score(dfte, scoring_method='concordance_index'))
        top=cph.summary.assign(abscoef=lambda d:d['coef'].abs()).sort_values('abscoef',ascending=False).head(12)
        ax.barh(top.index[::-1], top['coef'][::-1], color=['#e74c3c' if v>0 else '#3498db' for v in top['coef'][::-1]], alpha=0.85)
        ax.axvline(0,color='black',lw=0.8); ax.set_title(f'{label}\nClinical-only Cox baseline C-index={cidx:.3f}\npenalizer={pen}')
    fig.tight_layout(); save(fig,out/'27b_28b_m2_lifelines.png')

def main():
    p=argparse.ArgumentParser(description='Notebook 06 — Baseline models')
    p.add_argument('--splits-dir', type=Path, default=Path('outputs')/'splits')
    p.add_argument('--output-dir', type=Path, default=Path('outputs')/'notebook_06')
    args=p.parse_args()
    fig_accuracy_illusion(args.splits_dir,args.output_dir)
    fig_classification_baselines(args.splits_dir,args.output_dir)
    fig_m2_lifelines(args.splits_dir,args.output_dir)
if __name__=='__main__': main()
