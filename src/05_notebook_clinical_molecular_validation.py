"""
Notebook 03 — Clinical and Molecular Insights
==============================================
Answers the biological validation question before any model is trained:
does the data tell a coherent clinical and molecular story?

All statistics in captions are computed within each function.
Manually curated content (biological pathway labels, clinical interpretations)
is explicitly labelled as such.

Sections
--------
  16  Kaplan–Meier survival by PAM50 subtype
  17  Kaplan–Meier survival by grade and receptor status
  18  Gene programme × PAM50 subtype heatmap (biological validation)
  19  Gene programme biological spotlight (GP05, GP10, GP14, GP01)
  20  Clinical variable correlation matrix
  21  Cohort confounding: before and after OHE mitigation
  22  Research questions and modelling rationale

Usage
-----
  python src/notebook_03_clinical_molecular_insights.py
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8", "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

CLINICAL = [
    "patient_id","age_at_diagnosis","geo_location_id","ethnicity",
    "type_of_breast_surgery","cancer_type","cancer_type_detailed","cellularity",
    "chemotherapy","pam50_+_claudin-low_subtype","cohort","er_status_measured_by_ihc",
    "er_status","neoplasm_histologic_grade","her2_status_measured_by_snp6","her2_status",
    "tumor_other_histologic_subtype","hormone_therapy","inferred_menopausal_state",
    "integrative_cluster","primary_tumor_laterality","lymph_nodes_examined_positive",
    "mutation_count","nottingham_prognostic_index","oncotree_code","overall_survival_months",
    "overall_survival","pr_status","radio_therapy","3-gene_classifier_subtype",
    "tumor_size","tumor_stage","death_from_cancer",
]

PAM50_ORDER  = ["Basal", "Her2", "LumB", "LumA", "claudin-low", "Normal"]
PAM50_COLORS = {"Basal":"#e74c3c","Her2":"#e67e22","LumB":"#f39c12",
                "LumA":"#3498db","claudin-low":"#9b59b6","Normal":"#2ecc71"}
HEAD_BG = "#2c3e50"; HEAD_FG = "white"
ROW_A   = "#f0f4f8"; ROW_B   = "white"; CELL_FG = "#2c3e50"
BORDER  = "#c8d0db"


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def km_curve(times, events):
    """Kaplan-Meier estimator. Returns (times, survival) as lists."""
    df = pd.DataFrame({"t": times, "e": events}).sort_values("t")
    S = 1.0; n = len(df); curve = [(0.0, 1.0)]
    for t in sorted(df["t"].unique()):
        d = int(((df["t"] == t) & (df["e"] == 1)).sum())
        if d > 0:
            S *= 1 - d / n
        n -= int((df["t"] == t).sum())
        curve.append((float(t), S))
    return list(zip(*curve))


def log_rank_p(t1, e1, t2, e2):
    """Log-rank test p-value between two groups."""
    all_t = sorted(set(t1) | set(t2))
    O1 = E1 = O2 = E2 = 0.0
    n1, n2 = len(t1), len(t2)
    t1s = pd.Series(t1); e1s = pd.Series(e1)
    t2s = pd.Series(t2); e2s = pd.Series(e2)
    for t in all_t:
        d1 = int(((t1s == t) & (e1s == 1)).sum())
        d2 = int(((t2s == t) & (e2s == 1)).sum())
        d  = d1 + d2
        n  = n1 + n2
        if n == 0:
            continue
        E1 += n1 * d / n if n > 0 else 0
        E2 += n2 * d / n if n > 0 else 0
        O1 += d1; O2 += d2
        n1 -= int((t1s == t).sum())
        n2 -= int((t2s == t).sum())
    chi2 = (O1 - E1)**2 / E1 if E1 > 0 else 0
    chi2 += (O2 - E2)**2 / E2 if E2 > 0 else 0
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return p


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — KM survival by PAM50
# ══════════════════════════════════════════════════════════════════════════════
def fig_km_pam50(df, out):
    """Kaplan–Meier curves computed here. Log-rank p vs LumA computed here."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Compute KM and summary stats per subtype
    summary_rows = []
    for sub in PAM50_ORDER:
        mask = df["pam50_+_claudin-low_subtype"] == sub
        if mask.sum() < 5:
            continue
        t   = df.loc[mask, "overall_survival_months"].values
        e   = df.loc[mask, "overall_survival"].values
        n   = int(mask.sum())
        n_e = int(e.sum())
        med_fu = float(np.median(t))
        km_t, km_s = km_curve(t, e)
        # 5-year survival (60 months)
        idx60 = max((i for i, tt in enumerate(km_t) if tt <= 60),
                    default=0)
        s5yr  = km_s[idx60] if km_s else np.nan
        # Plot
        axes[0].step(km_t, km_s, where="post",
                     color=PAM50_COLORS.get(sub, "grey"),
                     lw=2, label=f"{sub} (n={n})")
        summary_rows.append((sub, n, n_e, f"{n_e/n*100:.0f}%",
                              f"{med_fu:.0f}mo", f"{s5yr:.0%}"))

    # Log-rank p: each subtype vs LumA
    luma_mask = df["pam50_+_claudin-low_subtype"] == "LumA"
    t_luma = df.loc[luma_mask, "overall_survival_months"].values
    e_luma = df.loc[luma_mask, "overall_survival"].values
    pvals = {}
    for sub in PAM50_ORDER:
        if sub == "LumA": continue
        mask = df["pam50_+_claudin-low_subtype"] == sub
        if mask.sum() < 5: continue
        p = log_rank_p(df.loc[mask,"overall_survival_months"].values,
                       df.loc[mask,"overall_survival"].values,
                       t_luma, e_luma)
        pvals[sub] = p

    axes[0].axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.5,
                    label="50% survival")
    axes[0].set_xlabel("Time (months)")
    axes[0].set_ylabel("Survival probability")
    axes[0].set_title(
        "Kaplan–Meier survival by PAM50 subtype\n"
        "KM estimator computed here; log-rank p vs LumA computed here"
    )
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.05)
    # Add log-rank annotations
    for i, (sub, p) in enumerate(pvals.items()):
        ps = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        axes[0].text(0.98, 0.95 - i*0.07,
                     f"{sub} vs LumA: {ps}",
                     transform=axes[0].transAxes,
                     ha="right", va="top", fontsize=7.5,
                     color=PAM50_COLORS.get(sub,"grey"))

    # Right panel: summary table
    ax2 = axes[1]
    ax2.axis("off")
    ax2.text(0.5, 0.99,
             "PAM50 survival summary — all values computed here",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=10, fontweight="bold", color=HEAD_BG)

    headers = ["Subtype","n","Events","Event rate","Median f/u","5-yr survival"]
    col_x   = [0.0, 0.17, 0.27, 0.37, 0.52, 0.68]
    col_w   = [0.17, 0.10, 0.10, 0.15, 0.16, 0.17]
    head_h  = 0.055; row_h = 0.075
    y       = 0.92
    for h, x, w in zip(headers, col_x, col_w):
        ax2.add_patch(plt.Rectangle((x, y-head_h), w-0.004, head_h,
            facecolor=HEAD_BG, transform=ax2.transAxes, clip_on=False))
        ax2.text(x+0.008, y-head_h/2, h, transform=ax2.transAxes,
                 va="center", fontsize=8.5, fontweight="bold", color=HEAD_FG)
    y -= head_h
    for i, row in enumerate(summary_rows):
        y -= row_h
        bg = ROW_A if i % 2 == 0 else ROW_B
        for j, (txt, x, w) in enumerate(zip(row, col_x, col_w)):
            ax2.add_patch(plt.Rectangle((x, y), w-0.004, row_h,
                facecolor=bg, edgecolor=BORDER, lw=0.3,
                transform=ax2.transAxes, clip_on=False))
            fc = PAM50_COLORS.get(row[0], CELL_FG) if j == 0 else CELL_FG
            fw = "bold" if j == 0 else "normal"
            ax2.text(x+0.008, y+row_h/2, str(txt),
                     transform=ax2.transAxes, va="center",
                     fontsize=9, color=fc, fontweight=fw)
    ax2.text(0.5, 0.06,
             "Note: LumA has most patients but is not the best prognosis.\n"
             "claudin-low has highest event rate despite moderate n.\n"
             "[Clinical interpretation manually annotated]",
             transform=ax2.transAxes, ha="center", fontsize=8,
             style="italic", color="#555")

    fig.suptitle(
        "Section 1 — Survival Analysis by PAM50 Subtype\n"
        "Validates that molecular subtypes carry prognostic information "
        "consistent with published literature",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, out / "16_km_pam50.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — KM by grade and receptor status
# ══════════════════════════════════════════════════════════════════════════════
def fig_km_grade_receptor(df, out):
    """All KM curves and log-rank tests computed here."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    p_er = np.nan
    p_her2 = np.nan

    # Panel 1: by grade
    grade_colors = {1.0:"#2ecc71", 2.0:"#f39c12", 3.0:"#e74c3c"}
    grade_lr = {}
    g1_mask = df["neoplasm_histologic_grade"] == 1.0
    t1 = df.loc[g1_mask,"overall_survival_months"].values
    e1 = df.loc[g1_mask,"overall_survival"].values
    for g in [1.0, 2.0, 3.0]:
        mask = df["neoplasm_histologic_grade"] == g
        if mask.sum() < 5: continue
        t = df.loc[mask,"overall_survival_months"].values
        e = df.loc[mask,"overall_survival"].values
        n = int(mask.sum())
        km_t, km_s = km_curve(t, e)
        axes[0].step(km_t, km_s, where="post",
                     color=grade_colors[g], lw=2,
                     label=f"Grade {int(g)} (n={n})")
        if g != 1.0:
            p = log_rank_p(t, e, t1, e1)
            grade_lr[g] = p
    axes[0].set_xlabel("Months"); axes[0].set_ylabel("Survival probability")
    axes[0].set_title(
        "KM by histologic grade\n"
        + "\n".join([f"Grade {int(g)} vs Grade 1: "
                     f"{'p<0.001' if p<0.001 else f'p={p:.3f}'}"
                     for g, p in grade_lr.items()])
    )
    axes[0].legend(fontsize=9); axes[0].set_ylim(0, 1.05)

    # Panel 2: by ER status
    er_colors = {"Positive":"#3498db","Negative":"#e74c3c"}
    er_t = {}; er_e = {}
    for er in ["Positive","Negative"]:
        mask = df["er_status"] == er
        if mask.sum() < 5: continue
        t = df.loc[mask,"overall_survival_months"].values
        e = df.loc[mask,"overall_survival"].values
        n = int(mask.sum())
        km_t, km_s = km_curve(t, e)
        axes[1].step(km_t, km_s, where="post",
                     color=er_colors[er], lw=2,
                     label=f"ER {er} (n={n})")
        er_t[er] = t; er_e[er] = e
    if len(er_t) == 2:
        p_er = log_rank_p(er_t["Positive"], er_e["Positive"],
                          er_t["Negative"], er_e["Negative"])
    axes[1].set_xlabel("Months"); axes[1].set_ylabel("Survival probability")
    er_p_text = (
        "p=NA" if pd.isna(p_er)
        else ("p<0.001" if p_er < 0.001 else f"p={p_er:.3f}")
    )
    axes[1].set_title(
        f"KM by ER status\n"
        f"Log-rank: {er_p_text}\n"
        f"ER+ shows better prognosis [clinically expected]"
    )
    axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.05)

    # Panel 3: by HER2 status
    her2_colors = {"Positive":"#e74c3c","Negative":"#3498db"}
    her2_t = {}; her2_e = {}
    for h2 in ["Positive","Negative"]:
        mask = df["her2_status"] == h2
        if mask.sum() < 5: continue
        t = df.loc[mask,"overall_survival_months"].values
        e = df.loc[mask,"overall_survival"].values
        n = int(mask.sum())
        km_t, km_s = km_curve(t, e)
        axes[2].step(km_t, km_s, where="post",
                     color=her2_colors[h2], lw=2,
                     label=f"HER2 {h2} (n={n})")
        her2_t[h2] = t; her2_e[h2] = e
    if len(her2_t) == 2:
        p_her2 = log_rank_p(her2_t["Positive"], her2_e["Positive"],
                             her2_t["Negative"], her2_e["Negative"])
    axes[2].set_xlabel("Months"); axes[2].set_ylabel("Survival probability")
    her2_p_text = (
        "p=NA" if pd.isna(p_her2)
        else ("p<0.001" if p_her2 < 0.001 else f"p={p_her2:.3f}")
    )
    axes[2].set_title(
        f"KM by HER2 status\n"
        f"Log-rank: {her2_p_text}\n"
        f"Note: HER2+ in METABRIC era — targeted therapy limited"
    )
    axes[2].legend(fontsize=9); axes[2].set_ylim(0, 1.05)

    fig.suptitle(
        "Section 1 — Survival by Grade and Receptor Status\n"
        "All KM curves and log-rank tests computed here from raw data",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, out / "17_km_grade_receptor.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Gene programme × PAM50 heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig_gp_pam50_heatmap(df_train, gp_cols, out):
    """
    Computes mean GP score per PAM50 subtype and ANOVA F-stat per programme.
    Biological pathway labels are manually annotated — labelled as such.
    """
    # Compute mean GP per PAM50 (on training data)
    mean_mat = pd.DataFrame(index=PAM50_ORDER, columns=gp_cols, dtype=float)
    f_stats  = {}
    for gp in gp_cols:
        groups = [df_train.loc[df_train["pam50_+_claudin-low_subtype"]==sub, gp].dropna()
                  for sub in PAM50_ORDER]
        groups = [g for g in groups if len(g) > 2]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
        else:
            f, p = np.nan, np.nan
        f_stats[gp] = (f, p)
        for sub in PAM50_ORDER:
            vals = df_train.loc[df_train["pam50_+_claudin-low_subtype"]==sub, gp].dropna()
            mean_mat.loc[sub, gp] = vals.mean() if len(vals) > 0 else np.nan

    # Sort programmes by max absolute mean (most discriminative first)
    prog_order = mean_mat.abs().max(axis=0).sort_values(ascending=False).index

    # Biological pathway labels — manually curated
    bio_short = {
        "gene_programme_01": "GP01 Luminal/hormone",
        "gene_programme_02": "GP02 Invasion/stroma",
        "gene_programme_03": "GP03 PI3K/AKT",
        "gene_programme_04": "GP04 TGF-β/BCL2",
        "gene_programme_05": "GP05 Proliferation",
        "gene_programme_06": "GP06 TGF-β receptor",
        "gene_programme_07": "GP07 MMP/cell cycle",
        "gene_programme_08": "GP08 Notch/apoptosis",
        "gene_programme_09": "GP09 Luminal ER+",
        "gene_programme_10": "GP10 HER2/epithelial",
        "gene_programme_11": "GP11 TGF-β/ubiquitin",
        "gene_programme_12": "GP12 Angiogenesis",
        "gene_programme_13": "GP13 Mitochondria",
        "gene_programme_14": "GP14 Immune/IFN",
        "gene_programme_15": "GP15 Cell cycle arrest",
    }

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # Left: heatmap (mean scaled GP score)
    plot_mat = mean_mat[prog_order].T
    plot_mat.index = [bio_short.get(c, c) for c in plot_mat.index]
    non_na = plot_mat.values[~np.isnan(plot_mat.values)]
    vmax = float(np.max(np.abs(non_na))) if len(non_na) else 1.0
    sns.heatmap(plot_mat, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax, ax=axes[0],
                linewidths=0.5, square=False,
                cbar_kws={"shrink": 0.7, "label": "Mean GP score (training set)"},
                annot=True, fmt=".2f", annot_kws={"size": 7})
    axes[0].set_title(
        f"Mean gene programme score by PAM50 subtype\n"
        f"Training set only (n={len(df_train):,}). "
        f"Annotations = mean ± 0. "
        f"[Pathway labels manually annotated]"
    )
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30,
                             ha="right", fontsize=9)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=8)

    # Right: ANOVA F-stats (computed here)
    f_vals  = [f_stats[gp][0] for gp in prog_order]
    p_vals  = [f_stats[gp][1] for gp in prog_order]
    labels  = [bio_short.get(gp, gp) for gp in prog_order]
    colors  = ["#e74c3c" if f > 50 else "#f39c12" if f > 10
               else "#3498db" for f in f_vals]
    bars = axes[1].barh(labels[::-1], f_vals[::-1],
                         color=colors[::-1], alpha=0.85, height=0.6)
    axes[1].axvline(10,  color="#f39c12", ls="--", lw=1, alpha=0.7,
                    label="F=10")
    axes[1].axvline(50,  color="#e74c3c", ls="--", lw=1, alpha=0.7,
                    label="F=50")
    for bar, f, p in zip(bars, f_vals[::-1], p_vals[::-1]):
        ps = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        axes[1].text(f + 1, bar.get_y() + bar.get_height()/2,
                     f"F={f:.0f}  {ps}", va="center", fontsize=7.5)
    axes[1].set_xlabel("ANOVA F-statistic (GP score by PAM50 subtype)")
    axes[1].set_title(
        "Discriminative power per programme\n"
        "(all computed here from training data)\n"
        "Red=F>50 highly discriminative, amber=F>10"
    )
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Section 2 — Gene Programme Biological Validation\n"
        "Programmes with high F distinguish PAM50 subtypes — "
        "validates biological specificity",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, out / "18_gp_pam50_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Gene programme spotlight
# ══════════════════════════════════════════════════════════════════════════════
def fig_gp_spotlight(df_train, gp_cols, out):
    """
    Shows distribution of 4 key programmes across PAM50 and vs clinical vars.
    All correlations and distributions computed here.
    Pathway biological interpretations are manually annotated.
    """
    spotlight = [
        ("gene_programme_10", "GP10 — HER2/epithelial\n[ERBB2, CDH1, AR]",
         "Expected: highest in Her2-enriched subtype"),
        ("gene_programme_05", "GP05 — Proliferation\n[AURKA, CCNB1, BARD1]",
         "Expected: highest in LumB (high proliferation)"),
        ("gene_programme_14", "GP14 — Immune/interferon\n[STAT1, JAK2, CSF1R]",
         "Expected: highest in claudin-low (immune-rich)"),
        ("gene_programme_01", "GP01 — Luminal/hormone\n[GATA3, MAPT, RAB25]",
         "Expected: highest in LumA (hormone-driven)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col_i, (gp, label, expectation) in enumerate(spotlight):
        if gp not in df_train.columns:
            axes[0, col_i].axis("off")
            axes[1, col_i].axis("off")
            continue

        ax_vln = axes[0, col_i]
        pam_groups = [
            df_train.loc[df_train["pam50_+_claudin-low_subtype"] == sub, gp].dropna()
            for sub in PAM50_ORDER
        ]
        valid_groups = [g.values for g in pam_groups if len(g) > 2]
        valid_subs = [sub for sub, g in zip(PAM50_ORDER, pam_groups) if len(g) > 2]

        if valid_groups:
            parts = ax_vln.violinplot(
                valid_groups,
                positions=range(len(valid_groups)),
                showmedians=True,
            )
            for body, sub in zip(parts["bodies"], valid_subs):
                body.set_facecolor(PAM50_COLORS.get(sub, "grey"))
                body.set_alpha(0.7)

        ax_vln.set_xticks(range(len(valid_subs)))
        ax_vln.set_xticklabels(valid_subs, rotation=35, ha="right", fontsize=8)
        ax_vln.set_ylabel("GP score")

        if len(valid_groups) >= 2:
            f, p = stats.f_oneway(*valid_groups)
            anova_text = f"F={f:.1f}, {'p<0.001' if p < 0.001 else f'p={p:.3f}'}"
        else:
            anova_text = "F=NA, p=NA"

        ax_vln.set_title(
            f"{label}\n{anova_text}\n[{expectation}]",
            fontsize=8,
        )

        ax_cor = axes[1, col_i]
        grade_idx = df_train[["neoplasm_histologic_grade", gp]].dropna().index
        surv_idx = df_train[[gp, "overall_survival"]].dropna().index

        if len(grade_idx) >= 3:
            x_grade = df_train.loc[grade_idx, "neoplasm_histologic_grade"].astype(float)
            y_grade = df_train.loc[grade_idx, gp].astype(float)
            r_grade, p_grade = stats.pearsonr(x_grade, y_grade)

            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(grade_idx))
            ax_cor.scatter(
                x_grade.values + jitter,
                y_grade.values,
                alpha=0.12,
                s=5,
                c=[
                    PAM50_COLORS.get(sub, "grey")
                    for sub in df_train.loc[grade_idx, "pam50_+_claudin-low_subtype"].values
                ],
            )
            m, b, *_ = stats.linregress(x_grade.values, y_grade.values)
            xl = np.array([1, 3])
            ax_cor.plot(xl, m * xl + b, color="#2c3e50", lw=2)
        else:
            r_grade, p_grade = np.nan, np.nan

        if len(surv_idx) >= 3:
            x_surv = df_train.loc[surv_idx, gp].astype(float)
            y_surv = df_train.loc[surv_idx, "overall_survival"].astype(float)
            r_surv, p_surv = stats.pearsonr(x_surv, y_surv)
        else:
            r_surv, p_surv = np.nan, np.nan

        def _fmt_p(p):
            if pd.isna(p):
                return "p=NA"
            return "p<0.001" if p < 0.001 else f"p={p:.3f}"

        def _fmt_r(r):
            return "NA" if pd.isna(r) else f"{r:.3f}"

        ax_cor.set_xticks([1, 2, 3])
        ax_cor.set_xticklabels(["G1", "G2", "G3"])
        ax_cor.set_xlabel("Histologic grade")
        ax_cor.set_ylabel("GP score")
        ax_cor.set_title(
            f"vs grade: r={_fmt_r(r_grade)} ({_fmt_p(p_grade)})\n"
            f"vs survival: r={_fmt_r(r_surv)} ({_fmt_p(p_surv)})",
            fontsize=8,
        )

    fig.suptitle(
        "Section 2 — Gene Programme Spotlight\n"
        "4 key programmes validated against known biology. "
        "All statistics computed here. "
        "[Pathway labels manually annotated]",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, out / "19_gp_spotlight.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Clinical correlation matrix
# ══════════════════════════════════════════════════════════════════════════════
def fig_clinical_correlations(df, out):
    """Pearson correlations computed here. NPI shown but excluded from models."""
    clin_feats = {
        "age_at_diagnosis":               "Age",
        "tumor_size":                     "Tumour size",
        "lymph_nodes_examined_positive":  "Lymph nodes+",
        "neoplasm_histologic_grade":      "Grade",
        "tumor_stage":                    "Stage",
        "nottingham_prognostic_index":    "NPI ⚠ excl.",
        "overall_survival_months":        "Survival months",
        "mutation_count":                 "Mutation count",
        "overall_survival":               "Deceased (OS=0→excl)",
    }

    # Compute correlation matrix
    data = df[list(clin_feats.keys())].dropna()
    corr = data.corr(method="pearson")
    corr.index   = list(clin_feats.values())
    corr.columns = list(clin_feats.values())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: heatmap
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=axes[0],
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.5,
                cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    axes[0].set_title(
        f"Clinical variable correlations (n={len(data):,} complete rows)\n"
        f"All Pearson r values computed here. "
        f"NPI shown but excluded from models."
    )
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=8)

    # Right: key correlations annotated
    ax2 = axes[1]
    # Find top correlations (absolute r > 0.3)
    flat = corr.stack().reset_index()
    flat.columns = ["var1","var2","r"]
    flat = flat[flat["var1"] != flat["var2"]]
    flat = flat[flat["var1"] < flat["var2"]].sort_values("r", key=abs,
                                                          ascending=False)
    top = flat[flat["r"].abs() > 0.3].head(12)

    ax2.axis("off")
    ax2.text(0.5, 0.99,
             "Key correlations |r| > 0.3 — computed here",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=10, fontweight="bold", color=HEAD_BG)
    col_x = [0.0, 0.30, 0.60, 0.75]; col_w = [0.30, 0.30, 0.15, 0.25]
    headers = ["Variable 1","Variable 2","r","Implication"]
    head_h = 0.055; row_h = 0.070
    y = 0.92
    for h, x, w in zip(headers, col_x, col_w):
        ax2.add_patch(plt.Rectangle((x, y-head_h), w-0.004, head_h,
            facecolor=HEAD_BG, transform=ax2.transAxes, clip_on=False))
        ax2.text(x+0.008, y-head_h/2, h, transform=ax2.transAxes,
                 va="center", fontsize=8.5, fontweight="bold", color=HEAD_FG)
    y -= head_h

    # Manually annotate implications — labelled
    implications = {
        ("NPI ⚠ excl.", "Grade"):            "Direct: NPI contains grade",
        ("NPI ⚠ excl.", "Tumour size"):      "Direct: NPI contains size",
        ("NPI ⚠ excl.", "Lymph nodes+"):     "Direct: NPI contains nodes",
        ("Grade", "Tumour size"):             "Larger tumours → higher grade",
        ("Lymph nodes+", "Tumour size"):      "More nodes with larger tumours",
        ("Grade", "Lymph nodes+"):            "Aggressive tumours → nodal spread",
        ("Stage", "Lymph nodes+"):            "Staging includes nodal status",
        ("Stage", "Tumour size"):             "Size is key staging criterion",
    }

    for i, (_, row) in enumerate(top.iterrows()):
        y -= row_h
        bg = ROW_A if i % 2 == 0 else ROW_B
        impl = implications.get((row["var1"], row["var2"]),
                                 implications.get((row["var2"], row["var1"]),
                                                   "[manually annotated]"))
        r_color = "#c62828" if abs(row["r"]) > 0.7 else \
                  "#e65100" if abs(row["r"]) > 0.5 else CELL_FG
        vals = [row["var1"], row["var2"], f"{row['r']:.3f}", impl]
        for j, (txt, x, w) in enumerate(zip(vals, col_x, col_w)):
            ax2.add_patch(plt.Rectangle((x, y), w-0.004, row_h,
                facecolor=bg, edgecolor=BORDER, lw=0.3,
                transform=ax2.transAxes, clip_on=False))
            ax2.text(x+0.008, y+row_h/2, str(txt),
                     transform=ax2.transAxes, va="center",
                     fontsize=8, color=r_color if j==2 else CELL_FG)

    ax2.text(0.5, 0.04,
             "[Implication column manually annotated]",
             transform=ax2.transAxes, ha="center", fontsize=7.5,
             style="italic", color="#777")

    fig.suptitle(
        "Section 3 — Clinical Variable Correlations\n"
        "NPI shown for reference but excluded from all models "
        "(structural redundancy + leakage into M4)",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, out / "20_clinical_correlations.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Cohort confounding before and after OHE mitigation
# ══════════════════════════════════════════════════════════════════════════════
def fig_cohort_confounding_mitigation(df_train, gp_cols, out):
    """
    Computes F-stat for each GP vs cohort before and after residualising
    on cohort OHE. All F-stats computed here.
    """
    cohort_ohe = pd.get_dummies(df_train["cohort"],
                                 prefix="cohort", dtype=float)
    cohort_ohe = cohort_ohe.iloc[:, :-1]   # drop reference

    results = []
    for gp in gp_cols:
        gp_vals = df_train[gp].fillna(0)
        groups_before = [gp_vals[df_train["cohort"] == c].values
                         for c in [1,2,3,4,5]]
        f_b, p_b = stats.f_oneway(
            *[g for g in groups_before if len(g) > 0])

        # Residualise on cohort OHE
        lr = LinearRegression()
        lr.fit(cohort_ohe, gp_vals)
        resid = gp_vals - lr.predict(cohort_ohe)
        groups_after = [resid[df_train["cohort"] == c].values
                        for c in [1,2,3,4,5]]
        f_a, p_a = stats.f_oneway(
            *[g for g in groups_after if len(g) > 0])
        results.append({"gp": gp, "f_before": f_b, "p_before": p_b,
                         "f_after": f_a,  "p_after": p_a,
                         "reduction_pct": (f_b - f_a) / f_b * 100
                                          if f_b > 0 else 0})

    res_df = pd.DataFrame(results).sort_values("f_before", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: before vs after scatter
    axes[0].scatter(res_df["f_before"], res_df["f_after"],
                    c="#3498db", s=60, alpha=0.8, zorder=3)
    for _, row in res_df.iterrows():
        axes[0].annotate(row["gp"].replace("gene_programme_","GP"),
                         (row["f_before"], row["f_after"]),
                         fontsize=6.5, ha="left", va="bottom")
    maxf = res_df["f_before"].max()
    axes[0].plot([0, maxf], [0, maxf], "k--", lw=0.8, alpha=0.4,
                 label="No change")
    axes[0].axhline(10, color="#f39c12", ls="--", lw=0.8,
                    label="F=10")
    axes[0].set_xlabel("F-statistic before OHE correction")
    axes[0].set_ylabel("F-statistic after OHE correction")
    axes[0].set_title(
        "Cohort confounding: before vs after\n"
        "All F-stats computed here. "
        "Points below diagonal = confounding reduced."
    )
    axes[0].legend(fontsize=8)

    # Panel 2: reduction bar chart
    res_sorted = res_df.sort_values("reduction_pct", ascending=True)
    colors2 = ["#2ecc71" if r > 80 else "#f39c12" if r > 50 else "#e74c3c"
               for r in res_sorted["reduction_pct"]]
    bars = axes[1].barh(
        res_sorted["gp"].str.replace("gene_programme_","GP"),
        res_sorted["reduction_pct"],
        color=colors2, alpha=0.85, height=0.6)
    for bar, v in zip(bars, res_sorted["reduction_pct"]):
        axes[1].text(v + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{v:.0f}%", va="center", fontsize=8)
    axes[1].set_xlabel("F-statistic reduction (%)")
    axes[1].set_title(
        "% F-reduction after cohort OHE residualisation\n"
        "Green = >80% reduction (cohort effect largely removed)\n"
        "Computed from training data"
    )
    axes[1].axvline(80, color="#2ecc71", ls="--", lw=1)

    # Panel 3: example GP — before/after distribution by cohort
    # Use GP with highest original F-stat
    top_gp = res_df.iloc[0]["gp"]
    gp_vals = df_train[top_gp].fillna(0)
    lr2 = LinearRegression().fit(cohort_ohe, gp_vals)
    resid2 = gp_vals - lr2.predict(cohort_ohe)
    f_b2 = res_df.loc[res_df["gp"]==top_gp,"f_before"].values[0]
    f_a2 = res_df.loc[res_df["gp"]==top_gp,"f_after"].values[0]

    cohort_colors = sns.color_palette("tab10", 5)
    for ci, c in enumerate([1,2,3,4,5]):
        mask = df_train["cohort"] == c
        # Before: violin at position ci
        axes[2].scatter(
            np.full(mask.sum(), ci * 2),
            gp_vals[mask].values,
            alpha=0.15, s=4, color=cohort_colors[ci],
            label=f"Cohort {c}")
        # After: violin at position ci+0.8
        axes[2].scatter(
            np.full(mask.sum(), ci * 2 + 0.8),
            resid2[mask].values,
            alpha=0.15, s=4, color=cohort_colors[ci])
    # Add medians
    for ci, c in enumerate([1,2,3,4,5]):
        mask = df_train["cohort"] == c
        axes[2].plot(ci*2, gp_vals[mask].median(),
                     "s", color=cohort_colors[ci], ms=8, zorder=5)
        axes[2].plot(ci*2+0.8, resid2[mask].median(),
                     "D", color=cohort_colors[ci], ms=8, zorder=5)
    axes[2].set_xticks([0.4, 2.4, 4.4, 6.4, 8.4])
    axes[2].set_xticklabels(["Coh1","Coh2","Coh3","Coh4","Coh5"])
    axes[2].set_ylabel("GP score")
    axes[2].set_title(
        f"{top_gp.replace('gene_programme_','GP')}: "
        f"before (circle) vs after (diamond)\n"
        f"F before={f_b2:.1f} → F after={f_a2:.2f}\n"
        f"Mitigation: cohort OHE included as feature in all models"
    )
    axes[2].legend(fontsize=7, markerscale=1.5)

    fig.suptitle(
        "Section 4 — Cohort Confounding Mitigation\n"
        "Residualising on cohort OHE reduces GP-cohort F-stats. "
        "All statistics computed here from training data.",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    save(fig, out / "21_cohort_confounding_mitigation.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Research questions and modelling rationale
# ══════════════════════════════════════════════════════════════════════════════

def fig_research_questions(df, gp_cols, df_train, out):
    """
    Research questions and modelling rationale adapted to the final 6-task pipeline.
    Evidence values are computed here. Clinical-use wording is manually annotated.
    """

    def safe_numeric(s):
        s = pd.to_numeric(pd.Series(s), errors="coerce")
        return s.replace([np.inf, -np.inf], np.nan).dropna().astype(float)

    def safe_anova_from_groups(groups):
        clean = [safe_numeric(g).values for g in groups]
        clean = [g for g in clean if len(g) >= 2]
        if len(clean) < 2:
            return np.nan, np.nan
        try:
            f, p = stats.f_oneway(*clean)
            return float(f), float(p)
        except Exception:
            return np.nan, np.nan

    def safe_pearson(x, y):
        x = pd.to_numeric(pd.Series(x), errors="coerce")
        y = pd.to_numeric(pd.Series(y), errors="coerce")
        common = x.dropna().index.intersection(y.dropna().index)
        if len(common) < 3:
            return np.nan, np.nan
        try:
            r, p = stats.pearsonr(x.loc[common], y.loc[common])
            return float(r), float(p)
        except Exception:
            return np.nan, np.nan

    def fmt_p(p):
        if pd.isna(p):
            return "p=NA"
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"

    def fmt_f(f):
        return "NA" if pd.isna(f) else f"{f:.1f}"

    def fmt_r(r):
        return "NA" if pd.isna(r) else f"{r:.3f}"

    # Evidence for M1a / M2a (all-cause)
    grade_groups_os = [
        df.loc[df["neoplasm_histologic_grade"] == g, "overall_survival"]
        for g in [1.0, 2.0, 3.0]
    ]
    f_grade_os, p_grade_os = safe_anova_from_groups(grade_groups_os)

    mask_cl = df["pam50_+_claudin-low_subtype"] == "claudin-low"
    mask_lu = df["pam50_+_claudin-low_subtype"] == "LumA"
    p_km_os = log_rank_p(
        df.loc[mask_cl, "overall_survival_months"].values,
        df.loc[mask_cl, "overall_survival"].values,
        df.loc[mask_lu, "overall_survival_months"].values,
        df.loc[mask_lu, "overall_survival"].values,
    )

    # Evidence for M1b / M2b (cancer-specific)
    grade_groups_css = [
        df.loc[df["neoplasm_histologic_grade"] == g, "death_from_cancer"]
        for g in [1.0, 2.0, 3.0]
    ]
    f_grade_css, p_grade_css = safe_anova_from_groups(grade_groups_css)

    subtype_groups_css = [
        df.loc[df["pam50_+_claudin-low_subtype"] == sub, "death_from_cancer"]
        for sub in PAM50_ORDER
        if (df["pam50_+_claudin-low_subtype"] == sub).sum() >= 2
    ]
    f_sub_css, p_sub_css = safe_anova_from_groups(subtype_groups_css)

    # Evidence for M3
    gp_f_max = 0.0
    gp_best = None
    for gp in gp_cols:
        groups = [
            df_train.loc[df_train["pam50_+_claudin-low_subtype"] == sub, gp]
            for sub in PAM50_ORDER
            if (df_train["pam50_+_claudin-low_subtype"] == sub).sum() >= 2
        ]
        f, _ = safe_anova_from_groups(groups)
        if not pd.isna(f) and f > gp_f_max:
            gp_f_max = float(f)
            gp_best = gp

    # Evidence for M4
    r_gp05_grade, p_gp05_grade = safe_pearson(
        df_train["gene_programme_05"],
        df_train["neoplasm_histologic_grade"],
    )

    rows = [
        (
            "M1a",
            "Overall survival\n(binary classification)",
            f"Log-rank claudin-low vs LumA:\n{fmt_p(p_km_os)}\n"
            f"Grade ANOVA vs overall survival:\nF={fmt_f(f_grade_os)} ({fmt_p(p_grade_os)})",
            "All-cause risk stratification at diagnosis.\n"
            "Useful baseline, but mixes oncologic and non-oncologic mortality.\n"
            "[Clinical interpretation — manual]",
            "Target: overall_survival\nMetric: AUC-ROC\nShared frozen test set: 381 rows",
        ),
        (
            "M1b",
            "Cancer-specific survival\n(binary classification)",
            f"Grade ANOVA vs death_from_cancer:\nF={fmt_f(f_grade_css)} ({fmt_p(p_grade_css)})\n"
            f"PAM50 ANOVA vs death_from_cancer:\nF={fmt_f(f_sub_css)} ({fmt_p(p_sub_css)})",
            "Cancer-specific risk stratification.\n"
            "Best product-facing binary target.\n"
            "[Clinical interpretation — manual]",
            "Target: death_from_cancer\nMetric: AUC-ROC\nAligned to oncologic question",
        ),
        (
            "M2a",
            "Overall survival time\n(Cox, all-cause)",
            "Same disease-separation evidence as M1a.\n"
            "Time variable available for all rows.\n"
            "Censoring handled explicitly in Cox.",
            "All-cause time-to-event modelling.\n"
            "Useful comparator, less specific clinically.\n"
            "[Clinical interpretation — manual]",
            "Target: overall_survival_months + overall_survival\nMetric: C-index\nLifelines Cox path",
        ),
        (
            "M2b",
            "Cancer-specific survival time\n(Cox, cancer-specific)",
            "Same disease-separation evidence as M1b.\n"
            "Grade/PAM50 both relate to cancer-specific death.\n"
            "Censoring handled explicitly in Cox.",
            "Cancer-specific time-to-event modelling.\n"
            "Best survival-facing product target.\n"
            "[Clinical interpretation — manual]",
            "Target: overall_survival_months + death_from_cancer\nMetric: C-index\nLifelines Cox path",
        ),
        (
            "M3",
            "PAM50 molecular subtype\n(multiclass classification)",
            f"Strongest programme ANOVA:\n{gp_best or 'NA'} with F={gp_f_max:.1f}\n"
            "Gene programmes discriminate subtype structure.",
            "Subtype inference support.\n"
            "Useful when molecular subtype is unavailable or needs cross-checking.\n"
            "[Clinical interpretation — manual]",
            "Target: pam50_target (6 classes)\nMetric: Macro-F1\nRare classes dropped",
        ),
        (
            "M4",
            "Histologic grade\n(ordinal classification)",
            f"GP05 vs grade correlation:\nr={fmt_r(r_gp05_grade)} ({fmt_p(p_gp05_grade)})\n"
            "Supports a biological basis for grade prediction.",
            "Structured pathology support / second-opinion framing.\n"
            "Useful as a pathology companion task.\n"
            "[Clinical interpretation — manual]",
            "Target: neoplasm_histologic_grade_ord\nMetric: QW-kappa\nNPI excluded as leakage",
        ),
    ]

    model_colors = {
        "M1a": "#3498db",
        "M1b": "#1f77b4",
        "M2a": "#9b59b6",
        "M2b": "#7d3c98",
        "M3": "#e67e22",
        "M4": "#2ecc71",
    }

    fig_h = len(rows) * 1.28 + 2.2
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.axis("off")
    ax.text(
        0.5, 0.985,
        "Section 5 — Research Questions and Modelling Rationale",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=13, fontweight="bold", color=HEAD_BG,
    )
    ax.text(
        0.5, 0.962,
        "Evidence column: statistics computed here. Clinical-use wording: manually annotated. "
        "Updated to the final 6-task pipeline (M1a/M1b/M2a/M2b/M3/M4).",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=9, style="italic", color="#555",
    )

    headers = ["Model", "Target / task", "Data evidence (computed here)", "Clinical use [manual]", "Implementation notes"]
    col_x = [0.0, 0.07, 0.24, 0.48, 0.69]
    col_w = [0.07, 0.17, 0.24, 0.21, 0.30]
    head_h = 0.055
    row_h = 0.130
    sep = 0.010
    y = 0.935

    for h, x, w in zip(headers, col_x, col_w):
        ax.add_patch(plt.Rectangle((x, y - head_h), w - 0.004, head_h,
                                   facecolor=HEAD_BG, transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.008, y - head_h / 2, h, transform=ax.transAxes,
                va="center", fontsize=9, fontweight="bold", color=HEAD_FG)
    y -= head_h

    for i, (model, task, evidence, use, impl) in enumerate(rows):
        y -= (row_h + sep)
        mc = model_colors.get(model, "#888")

        ax.add_patch(plt.Rectangle((col_x[0], y), col_w[0] - 0.004, row_h,
                                   facecolor=mc, transform=ax.transAxes, clip_on=False))
        ax.text(col_x[0] + 0.018, y + row_h / 2, model, transform=ax.transAxes,
                va="center", fontsize=13, fontweight="bold", color="white")

        vals = [task, evidence, use, impl]
        xs = col_x[1:]
        ws = col_w[1:]
        bg = ROW_A if i % 2 == 0 else ROW_B
        for txt, x, w in zip(vals, xs, ws):
            ax.add_patch(plt.Rectangle((x, y), w - 0.004, row_h,
                                       facecolor=bg, edgecolor=BORDER, lw=0.3,
                                       transform=ax.transAxes, clip_on=False))
            ax.text(x + 0.008, y + row_h / 2, str(txt), transform=ax.transAxes,
                    va="center", fontsize=8.5, color=CELL_FG)

    save(fig, out / "22_research_questions.png")



def load_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="	")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(
        f"Unsupported file type: {path.suffix}. "
        "Use CSV, TSV, parquet, or pickle."
    )


def coerce_numeric_columns(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_text_lines(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def apply_deterministic_fixes(df):
    df = df.copy()

    if "patient_id" in df.columns and df["patient_id"].duplicated().any():
        df = (
            df.sort_values("patient_id")
              .drop_duplicates(subset=["patient_id"], keep="first")
              .reset_index(drop=True)
        )

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    if "cancer_type" in df.columns:
        df.loc[df["cancer_type"] == "Breast Sarcoma", "cancer_type"] = "Breast Cancer"

    if "er_status_measured_by_ihc" in df.columns:
        mask = df["er_status_measured_by_ihc"].isin(["Posyte", "Positve"])
        df.loc[mask, "er_status_measured_by_ihc"] = "Positive"

    if "cancer_type_detailed" in df.columns:
        df.loc[df["cancer_type_detailed"] == "Breast", "cancer_type_detailed"] = np.nan

    if "geo_location_id" in df.columns:
        df.loc[df["geo_location_id"] == 0, "geo_location_id"] = np.nan

    if "age_at_diagnosis" in df.columns:
        age = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
        df["age_at_diagnosis"] = age.where(age.between(18, 100))

    if "death_from_cancer" in df.columns:
        mapped = df["death_from_cancer"].map(
            {"Died of Disease": 1, "Died of Other Causes": 0, "Living": 0}
        )
        if mapped.notna().any():
            df["death_from_cancer"] = mapped
        df["death_from_cancer"] = pd.to_numeric(df["death_from_cancer"], errors="coerce")

    numeric_cols = [
        "overall_survival_months",
        "overall_survival",
        "neoplasm_histologic_grade",
        "tumor_stage",
        "tumor_size",
        "lymph_nodes_examined_positive",
        "mutation_count",
        "nottingham_prognostic_index",
        "cohort",
    ]
    df = coerce_numeric_columns(df, numeric_cols)

    missing_target_mask = (
        df["overall_survival"].isna() | df["overall_survival_months"].isna()
    )
    if missing_target_mask.any():
        df = df.loc[~missing_target_mask].reset_index(drop=True)

    return df


def apply_basic_analysis_imputation(df, train_idx):
    df = df.copy()
    if len(df) == 0:
        return df

    train_idx = [i for i in train_idx if 0 <= i < len(df)]
    if not train_idx:
        train_idx = list(range(len(df)))
    df_tr = df.iloc[train_idx]

    if "age_at_diagnosis" in df.columns:
        med = pd.to_numeric(df_tr["age_at_diagnosis"], errors="coerce").median()
        if pd.notna(med):
            df["age_at_diagnosis"] = pd.to_numeric(df["age_at_diagnosis"], errors="coerce").fillna(med)

    if "mutation_count" in df.columns:
        med = pd.to_numeric(df_tr["mutation_count"], errors="coerce").median()
        if pd.notna(med):
            df["mutation_count"] = pd.to_numeric(df["mutation_count"], errors="coerce").fillna(med)

    if "tumor_size" in df.columns:
        df["tumor_size"] = pd.to_numeric(df["tumor_size"], errors="coerce")
        stage_col = pd.to_numeric(df.get("tumor_stage"), errors="coerce") if "tumor_stage" in df.columns else None
        tr_stage = pd.to_numeric(df_tr.get("tumor_stage"), errors="coerce") if "tumor_stage" in df_tr.columns else None
        global_med = pd.to_numeric(df_tr["tumor_size"], errors="coerce").median()
        if stage_col is not None and tr_stage is not None:
            for stage in tr_stage.dropna().unique():
                stage_med = pd.to_numeric(df_tr.loc[tr_stage == stage, "tumor_size"], errors="coerce").median()
                if pd.notna(stage_med):
                    mask = stage_col.eq(stage) & df["tumor_size"].isna()
                    df.loc[mask, "tumor_size"] = stage_med
        if pd.notna(global_med):
            df["tumor_size"] = df["tumor_size"].fillna(global_med)

    for col in ["tumor_stage", "neoplasm_histologic_grade", "pam50_+_claudin-low_subtype", "er_status", "her2_status", "pr_status"]:
        if col in df.columns:
            series_tr = df_tr[col].dropna()
            if not series_tr.empty:
                df[col] = df[col].fillna(series_tr.mode().iloc[0])

    return df


def attach_gene_programmes(df, pipeline_outputs, train_idx):
    out = Path(pipeline_outputs)
    nmf_model_path = out / "nmf" / "nmf_model.joblib"
    nmf_scaler_path = out / "nmf" / "nmf_minmax_scaler.joblib"
    kept_genes_path = out / "metadata" / "kept_genes.txt"

    if not nmf_model_path.exists():
        raise FileNotFoundError(
            f"NMF model not found: {nmf_model_path}. Run the pipeline first."
        )
    if not nmf_scaler_path.exists():
        raise FileNotFoundError(
            f"NMF scaler not found: {nmf_scaler_path}. Run the pipeline first."
        )
    kept_genes = read_text_lines(kept_genes_path)

    import joblib

    nmf = joblib.load(nmf_model_path)
    nmf_scaler = joblib.load(nmf_scaler_path)

    missing_genes = [g for g in kept_genes if g not in df.columns]
    if missing_genes:
        raise ValueError(
            f"Input data are missing {len(missing_genes)} genes required by the saved NMF model. "
            f"First missing genes: {missing_genes[:10]}"
        )

    gene_df = df[kept_genes].apply(pd.to_numeric, errors="coerce").clip(lower=-10, upper=10)
    valid_train_idx = [i for i in train_idx if 0 <= i < len(gene_df)]
    if not valid_train_idx:
        valid_train_idx = list(range(len(gene_df)))
    gene_train = gene_df.iloc[valid_train_idx]
    gene_medians = gene_train.median(axis=0)
    gene_df = gene_df.fillna(gene_medians).fillna(0.0)

    full_scaled = np.clip(nmf_scaler.transform(gene_df.values), 0.0, 1.0)
    W_full = nmf.transform(full_scaled)
    prog_labels = [f"gene_programme_{i+1:02d}" for i in range(W_full.shape[1])]

    df = df.copy()
    for i, col in enumerate(prog_labels):
        df[col] = W_full[:, i]
    return df, prog_labels


def infer_train_df(
    df,
    train_df=None,
    split_col=None,
    train_value="train",
    pipeline_outputs=None,
):
    if train_df is not None:
        return train_df.copy()

    if split_col and split_col in df.columns:
        mask = df[split_col].astype(str).str.lower() == str(train_value).lower()
        return df.loc[mask].copy()

    if "split" in df.columns:
        return df.loc[df["split"].astype(str).str.lower() == "train"].copy()

    if "is_train" in df.columns:
        vals = df["is_train"]
        if pd.api.types.is_bool_dtype(vals):
            return df.loc[vals].copy()
        mask = vals.astype(str).str.lower().isin({"1", "true", "train", "yes"})
        return df.loc[mask].copy()

    if pipeline_outputs is not None:
        train_idx_path = Path(pipeline_outputs) / "metadata" / "shared_train_indices.csv"
        if train_idx_path.exists():
            idx = pd.read_csv(train_idx_path)["row_index"].astype(int).tolist()
            idx = [i for i in idx if 0 <= i < len(df)]
            if idx:
                return df.iloc[idx].copy()

    return df.copy()


def ensure_required_columns(df, columns, df_name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clinical and molecular validation figures."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "FCS_ml_test_input_data_rna_mutation.csv",
        help="Raw METABRIC input table or a merged table that already contains gene programmes.",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=None,
        help="Optional path to a separate training table when --input already contains gene programmes.",
    )
    parser.add_argument(
        "--pipeline-outputs",
        type=Path,
        default=Path("outputs"),
        help="Pipeline outputs directory containing metadata/, nmf/, and saved artefacts.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs") / "notebook_05",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--split-col",
        default=None,
        help="Optional column in --input used to identify training rows when --input already contains gene programmes.",
    )
    parser.add_argument(
        "--train-value",
        default="train",
        help="Value in --split-col that marks training rows.",
    )
    return parser.parse_args()


def prepare_analysis_tables(args):
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = load_table(args.input)
    has_gp = any(str(c).startswith("gene_programme_") for c in df.columns)

    if has_gp:
        df = apply_deterministic_fixes(df)
        df = apply_basic_analysis_imputation(
            df,
            train_idx=list(range(len(df))),
        )
        train_df = load_table(args.train_input) if args.train_input else None
        df_train = infer_train_df(
            df=df,
            train_df=train_df,
            split_col=args.split_col,
            train_value=args.train_value,
            pipeline_outputs=args.pipeline_outputs,
        )
        df_train = apply_basic_analysis_imputation(
            df_train,
            train_idx=list(range(len(df_train))),
        )
        gp_cols = sorted([c for c in df.columns if str(c).startswith("gene_programme_")])
        return df, df_train, gp_cols, "merged_table"

    df = apply_deterministic_fixes(df)

    train_idx_path = args.pipeline_outputs / "metadata" / "shared_train_indices.csv"
    if train_idx_path.exists():
        train_idx = pd.read_csv(train_idx_path)["row_index"].astype(int).tolist()
    else:
        train_idx = list(range(len(df)))

    df = apply_basic_analysis_imputation(df, train_idx)
    df, gp_cols = attach_gene_programmes(df, args.pipeline_outputs, train_idx)
    df_train = infer_train_df(
        df=df,
        train_df=None,
        split_col=None,
        train_value=args.train_value,
        pipeline_outputs=args.pipeline_outputs,
    )
    df_train = apply_basic_analysis_imputation(
        df_train,
        train_idx=list(range(len(df_train))),
    )
    return df, df_train, gp_cols, "raw_plus_pipeline_outputs"


def main():
    args = parse_args()

    df, df_train, gp_cols, mode = prepare_analysis_tables(args)

    numeric_cols = [
        "overall_survival_months",
        "overall_survival",
        "death_from_cancer",
        "neoplasm_histologic_grade",
        "tumor_stage",
        "tumor_size",
        "lymph_nodes_examined_positive",
        "mutation_count",
        "nottingham_prognostic_index",
        "cohort",
    ]
    df = coerce_numeric_columns(df, numeric_cols)
    df_train = coerce_numeric_columns(df_train, numeric_cols)

    if not gp_cols:
        raise ValueError(
            "No gene programme columns found or reconstructed. "
            "Run the pipeline first, or pass a merged table that already contains gene programmes."
        )

    required_all = [
        "pam50_+_claudin-low_subtype",
        "overall_survival_months",
        "overall_survival",
        "neoplasm_histologic_grade",
        "er_status",
        "her2_status",
        "tumor_size",
        "tumor_stage",
        "lymph_nodes_examined_positive",
        "mutation_count",
        "nottingham_prognostic_index",
        "death_from_cancer",
    ]
    required_train = required_all + ["cohort"]

    ensure_required_columns(df, required_all, "Analysis table")
    ensure_required_columns(df_train, required_train, "Training analysis table")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Input mode: {mode}")
    print(f"Loaded analysis table: {len(df):,} rows, {df.shape[1]:,} columns")
    print(f"Loaded training table: {len(df_train):,} rows, {df_train.shape[1]:,} columns")
    print(f"Detected {len(gp_cols)} gene programme columns")
    print(f"Writing figures to: {out}")

    fig_km_pam50(df, out)
    fig_km_grade_receptor(df, out)
    fig_gp_pam50_heatmap(df_train, gp_cols, out)
    fig_gp_spotlight(df_train, gp_cols, out)
    fig_clinical_correlations(df, out)
    fig_cohort_confounding_mitigation(df_train, gp_cols, out)
    fig_research_questions(df, gp_cols, df_train, out)

    manifest = {
        "input": str(Path(args.input)),
        "train_input": str(Path(args.train_input)) if args.train_input else None,
        "pipeline_outputs": str(Path(args.pipeline_outputs)),
        "outdir": str(out),
        "input_mode": mode,
        "n_rows_input": int(len(df)),
        "n_rows_train": int(len(df_train)),
        "n_gene_programmes": int(len(gp_cols)),
        "figures": [
            "16_km_pam50.png",
            "17_km_grade_receptor.png",
            "18_gp_pam50_heatmap.png",
            "19_gp_spotlight.png",
            "20_clinical_correlations.png",
            "21_cohort_confounding_mitigation.png",
            "22_research_questions.png",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved: {(out / 'manifest.json').name}")


if __name__ == "__main__":
    main()
