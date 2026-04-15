"""Notebook 07 — Models and Algorithm Comparison (final, M2 lifelines-only)
Outputs to outputs/notebook_07/
"""
import argparse, warnings
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
plt.rcParams.update({"figure.dpi":150,"figure.facecolor":"white","axes.facecolor":"#f8f8f8","axes.spines.top":False,"axes.spines.right":False})
ALGO_COLORS={"LR (L2)":"#3498db","EN (L1+L2)":"#9b59b6","RF":"#2ecc71"}
SET_COLORS={"A":"#95a5a6","B":"#3498db","C":"#e67e22","D":"#9b59b6"}
N_TOP_GENES=15
CLASS_MODELS=[('M1a_overall_survival','M1a overall survival','binary','AUC-ROC'),('M1b_cancer_specific_survival','M1b cancer-specific survival','binary','AUC-ROC'),('M3_pam50_subtype','M3 PAM50 subtype','multi','Macro-F1'),('M4_histologic_grade','M4 histologic grade','ordinal','QW-Kappa')]
M2_KEYS=[('M2a_overall_survival_cox','M2a overall survival Cox'),('M2b_cancer_specific_cox','M2b cancer-specific Cox')]

def save(fig,path): path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path,bbox_inches='tight'); plt.close(fig); print(f'  Saved: {path.name}')
def load_split(splits_dir,key):
    d=splits_dir/key; return pd.read_csv(d/'X_train.csv'), pd.read_csv(d/'X_test.csv'), pd.read_csv(d/'y_train.csv'), pd.read_csv(d/'y_test.csv')
def make_lr(): return LogisticRegression(C=1.0,class_weight='balanced',penalty='l2',solver='lbfgs',max_iter=2000,random_state=42)
def make_en(): return LogisticRegression(C=1.0,class_weight='balanced',penalty='elasticnet',solver='saga',l1_ratio=0.5,max_iter=3000,random_state=42)
def make_rf(): return RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1)

def compute_metric(clf, X, y, task):
    yp=clf.predict(X)
    if task=='binary': return roc_auc_score(y, clf.predict_proba(X)[:,1])
    if task=='multi': return f1_score(y, yp, average='macro', zero_division=0)
    return cohen_kappa_score(y, yp, weights='quadratic')

def bootstrap_ci(clf, X, y, task, n=100, seed=42):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y)); Xb=X.iloc[idx]; yb=y.iloc[idx]
        try: vals.append(compute_metric(clf,Xb,yb,task))
        except Exception: pass
    return np.percentile(vals,[2.5,97.5]) if vals else (np.nan,np.nan)

def build_feature_sets(X_tr, X_te, y_tr_series, feats):
    clin=[f for f in feats if not f.startswith('gene_programme')]
    gp=[f for f in feats if f.startswith('gene_programme')]
    gene_cols=[c for c in X_tr.columns if c.startswith('g_')]
    notes=[]
    X_tr_r=X_tr[clin].reset_index(drop=True); X_te_r=X_te[clin].reset_index(drop=True)
    gp_tr=X_tr[gp].reset_index(drop=True); gp_te=X_te[gp].reset_index(drop=True)
    if gene_cols:
        mi=mutual_info_classif(X_tr[gene_cols].values, y_tr_series.values, random_state=42)
        top_genes=pd.Series(mi,index=gene_cols).nlargest(min(N_TOP_GENES,len(gene_cols))).index.tolist()
        scaler=StandardScaler()
        gtr=pd.DataFrame(scaler.fit_transform(X_tr[top_genes]),columns=top_genes)
        gte=pd.DataFrame(scaler.transform(X_te[top_genes]),columns=top_genes)
    else:
        top_genes=[]; gtr=pd.DataFrame(index=X_tr.index); gte=pd.DataFrame(index=X_te.index)
        notes.append('Raw gene columns not present in split files: C==A and D==B.')
    sets_tr={'A':X_tr_r,'B':pd.concat([X_tr_r,gp_tr],axis=1),'C':pd.concat([X_tr_r,gtr.reset_index(drop=True)],axis=1),'D':pd.concat([X_tr_r,gp_tr,gtr.reset_index(drop=True)],axis=1)}
    sets_te={'A':X_te_r,'B':pd.concat([X_te_r,gp_te],axis=1),'C':pd.concat([X_te_r,gte.reset_index(drop=True)],axis=1),'D':pd.concat([X_te_r,gp_te,gte.reset_index(drop=True)],axis=1)}
    return sets_tr,sets_te,notes

def fig_feature_representation(splits_dir,models_dir,out):
    fig,axes=plt.subplots(2,2,figsize=(16,12)); axes=axes.flatten(); note_lines=[]
    for ax,(key,label,task,metric) in zip(axes, CLASS_MODELS):
        Xtr,Xte,ytr_raw,yte_raw=load_split(splits_dir,key); ytr=ytr_raw.iloc[:,0]; yte=yte_raw.iloc[:,0]
        feats=joblib.load(models_dir/f'{key}_features.joblib') if (models_dir/f'{key}_features.joblib').exists() else Xtr.columns.tolist()
        sets_tr,sets_te,notes=build_feature_sets(Xtr,Xte,ytr,feats)
        note_lines.extend([f'{label}: {n}' for n in notes])
        scores=[]; labels=[]
        for s in ['A','B','C','D']:
            lr=make_lr(); lr.fit(sets_tr[s], ytr); scores.append(compute_metric(lr, sets_te[s], yte, task)); labels.append(f'{s}\n({sets_tr[s].shape[1]}f)')
        ax.bar(labels,scores,color=[SET_COLORS[s] for s in ['A','B','C','D']],alpha=0.85)
        ax.set_title(label); ax.set_ylabel(metric)
    if note_lines:
        fig.text(0.02,0.01,' | '.join(dict.fromkeys(note_lines)),fontsize=8)
    fig.tight_layout(); save(fig,out/'29_feature_representation.png')

def fit_and_save_models(splits_dir, models_dir, cv_dir):
    models_dir.mkdir(parents=True, exist_ok=True); cv_dir.mkdir(parents=True, exist_ok=True)
    summary=[]
    # classification models
    for key,label,task,metric in CLASS_MODELS:
        Xtr,Xte,ytr_raw,yte_raw=load_split(splits_dir,key); ytr=ytr_raw.iloc[:,0]; yte=yte_raw.iloc[:,0]
        feats=(pd.read_csv(splits_dir/key/'feature_selection'/'selected_feature_list.csv')['feature'].tolist() if (splits_dir/key/'feature_selection'/'selected_feature_list.csv').exists() else Xtr.columns.tolist())
        feats=[f for f in feats if f in Xtr.columns]
        algos={'LR (L2)':make_lr(),'EN (L1+L2)':make_en(),'RF':make_rf()}
        rows=[]; best=None
        for name,clf in algos.items():
            clf.fit(Xtr[feats], ytr)
            sc=compute_metric(clf, Xte[feats], yte, task); ci=bootstrap_ci(clf,Xte[feats],yte,task)
            joblib.dump(clf, models_dir/f'{key}_{"rf" if name=="RF" else "lr" if name=="LR (L2)" else "en"}.joblib')
            rows.append({'algorithm':name,'score':sc,'ci_low':ci[0],'ci_high':ci[1]})
            if best is None or sc>best[1]: best=(name,sc,clf)
        joblib.dump(feats, models_dir/f'{key}_features.joblib')
        joblib.dump(best[0], models_dir/f'{key}_best_name.joblib')
        pd.DataFrame(rows).to_csv(cv_dir/key/'cv_results.csv', index=False) if (cv_dir/key).mkdir(parents=True, exist_ok=True) is None else None
        summary.append({'Model':label,'Metric':metric,'Best algorithm':best[0],'Test score':best[1]})
    # M2 proper training CV for feature-set selection
    from lifelines import CoxPHFitter
    def prep(X, scaler=None, fit=True):
        X=X.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
        med=X.median(numeric_only=True); X=X.fillna(med).fillna(0.0)
        keep=X.var(numeric_only=True); keep=keep[keep>1e-12].index.tolist()
        X=X[keep]
        # Pipeline outputs are already scaled for key continuous features and gene programmes.
        # Here we only clean and drop constant columns; no additional scaling is applied.
        return X.reset_index(drop=True), keep, None
    for m2key, m2label in M2_KEYS:
        Xtr,Xte,ytr,yte=load_split(splits_dir,m2key)
        feats=(pd.read_csv(splits_dir/m2key/'feature_selection'/'selected_feature_list.csv')['feature'].tolist() if (splits_dir/m2key/'feature_selection'/'selected_feature_list.csv').exists() else Xtr.columns.tolist())
        feats=[f for f in feats if f in Xtr.columns]
        clin=[f for f in feats if not f.startswith('gene_programme')]
        full=[f for f in feats]
        cv_rows=[]
        skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        strat=ytr['event'].astype(int)
        best_label=None; best_score=-np.inf; best_feats=None
        for feat_label, feat_list in [('A: Clinical',clin),('B: Clin+NMF',full)]:
            fold_scores=[]
            for tr_idx, va_idx in skf.split(Xtr, strat):
                Xtr_f, Xva_f = Xtr.iloc[tr_idx][feat_list], Xtr.iloc[va_idx][feat_list]
                ytr_f, yva_f = ytr.iloc[tr_idx], ytr.iloc[va_idx]
                Xtr_p, keep, sc = prep(Xtr_f)
                Xva_p = Xva_f.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
                Xva_p = Xva_p.fillna(Xtr_f.median(numeric_only=True)).fillna(0.0)
                Xva_p = Xva_p[keep].reset_index(drop=True)
                dtr=Xtr_p.copy(); dtr['time']=ytr_f['time'].values; dtr['event']=ytr_f['event'].values
                dva=Xva_p.copy(); dva['time']=yva_f['time'].values; dva['event']=yva_f['event'].values
                cph=CoxPHFitter(penalizer=1.0,l1_ratio=0.0)
                cph.fit(dtr,duration_col='time',event_col='event',show_progress=False)
                fold_scores.append(float(cph.score(dva, scoring_method='concordance_index')))
            mean_sc=float(np.mean(fold_scores))
            cv_rows.append({'feature_set':feat_label,'c_index_cv_mean':mean_sc})
            if mean_sc>best_score:
                best_score=mean_sc; best_label=feat_label; best_feats=feat_list
        pd.DataFrame(cv_rows).to_csv(cv_dir/m2key/'cv_results.csv', index=False) if (cv_dir/m2key).mkdir(parents=True, exist_ok=True) is None else None
        # fit once on full train, evaluate once on test
        Xtr_p, keep, sc = prep(Xtr[best_feats])
        Xte_p = Xte[best_feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
        Xte_p = Xte_p.fillna(Xtr[best_feats].median(numeric_only=True)).fillna(0.0)
        Xte_p = Xte_p[keep].reset_index(drop=True)
        dtr=Xtr_p.copy(); dtr['time']=ytr['time'].values; dtr['event']=ytr['event'].values
        dte=Xte_p.copy(); dte['time']=yte['time'].values; dte['event']=yte['event'].values
        cph=CoxPHFitter(penalizer=1.0,l1_ratio=0.0); cph.fit(dtr,duration_col='time',event_col='event',show_progress=False)
        cidx=float(cph.score(dte,scoring_method='concordance_index'))
        joblib.dump(cph, models_dir/f'{m2key}_cox.joblib'); joblib.dump(keep, models_dir/f'{m2key}_features.joblib'); joblib.dump(sc, models_dir/f'{m2key}_scaler.joblib'); joblib.dump(best_label, models_dir/f'{m2key}_best_name.joblib')
        summary.append({'Model':m2label,'Metric':'C-index','Best algorithm':'CoxPH','Test score':cidx})
    return pd.DataFrame(summary)

def fig_summary(summary_df,out):
    fig,ax=plt.subplots(figsize=(12,4+0.6*len(summary_df))); ax.axis('off'); ax.text(0.5,0.98,'Final model summary',ha='center',va='top',transform=ax.transAxes,fontsize=12,fontweight='bold')
    y=0.88
    for _,r in summary_df.iterrows():
        ax.text(0.02,y,f"{r['Model']}: {r['Best algorithm']} — {r['Metric']}={r['Test score']:.3f}",fontsize=10); y-=0.09
    fig.tight_layout(); save(fig,out/'30_model_summary.png')

def main():
    p=argparse.ArgumentParser(description='Notebook 07 — Models and algorithm comparison')
    p.add_argument('--splits-dir', type=Path, default=Path('outputs')/'splits')
    p.add_argument('--models-dir', type=Path, default=Path('outputs')/'models')
    p.add_argument('--cv-dir', type=Path, default=Path('outputs')/'cv_results')
    p.add_argument('--output-dir', type=Path, default=Path('outputs')/'notebook_07')
    args=p.parse_args()
    fig_feature_representation(args.splits_dir,args.models_dir,args.output_dir)
    summary_df=fit_and_save_models(args.splits_dir,args.models_dir,args.cv_dir)
    fig_summary(summary_df,args.output_dir)
if __name__=='__main__': main()
