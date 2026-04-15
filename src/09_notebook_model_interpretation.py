"""Notebook 09 — Model Interpretation (final)
Outputs to outputs/notebook_09/
"""
import argparse, warnings
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
plt.rcParams.update({"figure.dpi":150,"figure.facecolor":"white","axes.facecolor":"#f8f8f8","axes.spines.top":False,"axes.spines.right":False})
MODEL_DEFS=[('M1a_overall_survival','M1a overall survival'),('M1b_cancer_specific_survival','M1b cancer-specific survival'),('M3_pam50_subtype','M3 PAM50 subtype'),('M4_histologic_grade','M4 histologic grade')]

def save(fig,path): path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path,bbox_inches='tight'); plt.close(fig); print(f'  Saved: {path.name}')
def load_perm(perm_dir,key): return pd.read_csv(perm_dir/f'{key}_perm_imp.csv')
def feat_color(f): return '#2ecc71' if f.startswith('gene_programme') else ('#9b59b6' if f.startswith('ohe_') else '#3498db')

def fig_global(perm_dir, models_dir, out):
    fig,axes=plt.subplots(3,2,figsize=(16,16)); axes=axes.flatten()
    for ax,(key,label) in zip(axes[:4], MODEL_DEFS):
        pi=load_perm(perm_dir,key).head(20)
        ax.barh(pi['feature'][::-1], pi['importance_mean'][::-1], xerr=pi['importance_std'][::-1], color=[feat_color(f) for f in pi['feature'][::-1]], alpha=0.85)
        ax.set_title(label); ax.axvline(0,color='black',lw=0.8)
    for ax, key, title in [(axes[4],'M2a_overall_survival_cox','M2a overall survival — Cox coefficients'),(axes[5],'M2b_cancer_specific_cox','M2b cancer-specific — Cox coefficients')]:
        cph=joblib.load(models_dir/f'{key}_cox.joblib')
        top=cph.summary.assign(abs_coef=lambda d:d['coef'].abs()).sort_values('abs_coef', ascending=False).head(20)
        ax.barh(top.index[::-1], top['coef'][::-1], color=['#e74c3c' if v>0 else '#3498db' for v in top['coef'][::-1]], alpha=0.85)
        ax.set_title(title); ax.axvline(0,color='black',lw=0.8)
    fig.tight_layout(); save(fig,out/'35_global_importance.png')

def fig_lr(models_dir,out):
    fig,axes=plt.subplots(2,2,figsize=(16,12)); axes=axes.flatten()
    for ax,(key,label) in zip(axes, MODEL_DEFS):
        model_path=models_dir/f'{key}_lr.joblib'
        if not model_path.exists():
            ax.axis('off'); ax.text(0.02,0.8,f'{label}: LR model not saved',fontsize=10); continue
        lr=joblib.load(model_path); feats=joblib.load(models_dir/f'{key}_features.joblib')
        coef=np.abs(lr.coef_).mean(axis=0) if getattr(lr.coef_,'ndim',1)==2 and lr.coef_.shape[0]>1 else np.abs(lr.coef_[0])
        ser=pd.Series(coef,index=feats).sort_values(ascending=False).head(20)
        ax.barh(ser.index[::-1], ser.values[::-1], color=[feat_color(f) for f in ser.index[::-1]], alpha=0.85)
        ax.set_title(label)
    fig.tight_layout(); save(fig,out/'36_lr_coefficients.png')

def main():
    p=argparse.ArgumentParser(description='Notebook 09 — Model interpretation')
    p.add_argument('--perm-dir', type=Path, default=Path('outputs')/'permutation_importance')
    p.add_argument('--models-dir', type=Path, default=Path('outputs')/'models')
    p.add_argument('--output-dir', type=Path, default=Path('outputs')/'notebook_09')
    args=p.parse_args()
    fig_global(args.perm_dir,args.models_dir,args.output_dir)
    fig_lr(args.models_dir,args.output_dir)
if __name__=='__main__': main()
