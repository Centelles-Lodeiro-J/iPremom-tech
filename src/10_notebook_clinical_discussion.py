"""Notebook 10 — Clinical Discussion (final, includes M2 Cox)
Outputs to outputs/notebook_10/
"""
import argparse, warnings
from pathlib import Path
import joblib, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
warnings.filterwarnings('ignore')
plt.rcParams.update({"figure.dpi":150,"figure.facecolor":"white","axes.facecolor":"#f8f8f8","axes.spines.top":False,"axes.spines.right":False})
HEAD_BG="#2c3e50"; ROW_A="#f0f4f8"; ROW_B='white'

def save(fig,path): path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path,bbox_inches='tight'); plt.close(fig); print(f'  Saved: {path.name}')

def load_scores(splits_dir, models_dir, cv_dir):
    items=[]
    defs=[('M1a_overall_survival','M1a overall survival','binary'),('M1b_cancer_specific_survival','M1b cancer-specific survival','binary'),('M3_pam50_subtype','M3 PAM50 subtype','multi'),('M4_histologic_grade','M4 histologic grade','ordinal')]
    for key,label,task in defs:
        feats=joblib.load(models_dir/f'{key}_features.joblib')
        best_name=None
        cv_path=cv_dir/key/'cv_results.csv'
        if cv_path.exists():
            cv=pd.read_csv(cv_path)
            if not cv.empty and 'score' in cv.columns:
                best_name=str(cv.sort_values('score', ascending=False).iloc[0]['algorithm'])
        if best_name is None and (models_dir/f'{key}_best_name.joblib').exists():
            best_name=joblib.load(models_dir/f'{key}_best_name.joblib')
        suffix = 'rf' if best_name == 'RF' else ('en' if isinstance(best_name, str) and 'EN' in best_name else 'lr')
        model_path=models_dir/f'{key}_{suffix}.joblib'
        if not model_path.exists():
            model_path=models_dir/f'{key}_rf.joblib'
            if not model_path.exists(): model_path=models_dir/f'{key}_lr.joblib'
        clf=joblib.load(model_path)
        X=pd.read_csv(splits_dir/key/'X_test.csv')[feats]; y=pd.read_csv(splits_dir/key/'y_test.csv').iloc[:,0]
        yp=clf.predict(X)
        sc=roc_auc_score(y, clf.predict_proba(X)[:,1]) if task=='binary' else (f1_score(y, yp, average='macro', zero_division=0) if task=='multi' else cohen_kappa_score(y, yp, weights='quadratic'))
        items.append({'Model':label,'Metric':'AUC-ROC' if task=='binary' else ('Macro-F1' if task=='multi' else 'QW-Kappa'),'Score':sc,'Can do':'Risk stratification' if task=='binary' else ('Subtype prediction' if task=='multi' else 'Grade support'),'Cannot do':'External validation still needed'})
    for m2key, m2label in [('M2a_overall_survival_cox','M2a overall survival Cox'),('M2b_cancer_specific_cox','M2b cancer-specific Cox')]:
        cph=joblib.load(models_dir/f'{m2key}_cox.joblib')
        feats=joblib.load(models_dir/f'{m2key}_features.joblib')
        sc=joblib.load(models_dir/f'{m2key}_scaler.joblib') if (models_dir/f'{m2key}_scaler.joblib').exists() else None
        Xte=pd.read_csv(splits_dir/m2key/'X_test.csv')[feats]
        Xte=Xte.apply(pd.to_numeric, errors='coerce').replace([float('inf'),float('-inf')], pd.NA)
        Xte=Xte.fillna(Xte.median(numeric_only=True)).fillna(0.0)
        if sc is not None:
            # scaler retained for compatibility; current pipeline stores None because no extra scaling is applied here
            Xte=pd.DataFrame(sc.transform(Xte), columns=feats)
        yte=pd.read_csv(splits_dir/m2key/'y_test.csv')
        df=Xte.copy(); df['time']=yte['time'].values; df['event']=yte['event'].values
        cidx=float(cph.score(df, scoring_method='concordance_index'))
        items.append({'Model':m2label,'Metric':'C-index','Score':cidx,'Can do':'Ranks patients by survival risk with censoring handled','Cannot do':'Provide exact individual survival times'})
    return pd.DataFrame(items)

def fig_capability(scores,out):
    fig,ax=plt.subplots(figsize=(16,4+0.7*len(scores))); ax.axis('off'); ax.text(0.5,0.99,'Capability assessment — final models',ha='center',va='top',transform=ax.transAxes,fontsize=12,fontweight='bold')
    y=0.9
    for i,(_,r) in enumerate(scores.iterrows()):
        bg=ROW_A if i%2==0 else ROW_B
        ax.add_patch(plt.Rectangle((0.0,y-0.08),0.98,0.08,facecolor=bg,transform=ax.transAxes))
        ax.text(0.01,y-0.04,f"{r['Model']} | {r['Metric']}={r['Score']:.3f}",transform=ax.transAxes,fontsize=10,fontweight='bold')
        ax.text(0.36,y-0.04,f"Can do: {r['Can do']}",transform=ax.transAxes,fontsize=9)
        ax.text(0.68,y-0.04,f"Cannot do: {r['Cannot do']}",transform=ax.transAxes,fontsize=9)
        y-=0.09
    fig.tight_layout(); save(fig,out/'42_capability_assessment.png')

def fig_findings(scores,out):
    fig,ax=plt.subplots(figsize=(14,6)); ax.axis('off'); ax.text(0.5,0.98,'Key findings and limitations',ha='center',va='top',transform=ax.transAxes,fontsize=12,fontweight='bold')
    lines=[
        '1. M1b restores cancer-specific risk modelling rather than only all-cause mortality.',
        '2. M2 now uses proper lifelines Cox modelling and C-index, not proxy AUC.',
        '3. M3 and M4 remain classification tasks and are reported in Macro-F1 / QW-Kappa.',
        '4. Permutation importance for M4 now uses quadratic weighted kappa.',
        '5. Raw-gene feature sets are explicitly flagged when absent from split files.',
        'Limitations: no external validation yet; survival calibration remains limited.'
    ]
    y=0.85
    for line in lines:
        ax.text(0.03,y,line,transform=ax.transAxes,fontsize=10); y-=0.11
    fig.tight_layout(); save(fig,out/'43_key_findings_and_limitations.png')

def main():
    p=argparse.ArgumentParser(description='Notebook 10 — Clinical discussion')
    p.add_argument('--splits-dir', type=Path, default=Path('outputs')/'splits')
    p.add_argument('--models-dir', type=Path, default=Path('outputs')/'models')
    p.add_argument('--cv-dir', type=Path, default=Path('outputs')/'cv_results')
    p.add_argument('--output-dir', type=Path, default=Path('outputs')/'notebook_10')
    args=p.parse_args()
    scores=load_scores(args.splits_dir,args.models_dir,args.cv_dir)
    fig_capability(scores,args.output_dir)
    fig_findings(scores,args.output_dir)
if __name__=='__main__': main()
