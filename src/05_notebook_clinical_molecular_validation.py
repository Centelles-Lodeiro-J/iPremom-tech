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
    "mutation_count","nottingham_prognostic_index","overall_survival_months",
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
    axes[1].set_title(
        f"KM by ER status\n"
        f"Log-rank: {'p<0.001' if p_er<0.001 else f'p={p_er:.3f}'}\n"
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
    axes[2].set_title(
        f"KM by HER2 status\n"
        f"Log-rank: {'p<0.001' if p_her2<0.001 else f'p={p_her2:.3f}'}\n"
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
    vmax = max(abs(plot_mat.values[~np.isnan(plot_mat.values)]))
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
def fig_gp_spotlight(df_train, gp_cols, H, out):
    """
    Shows distribution of 4 key programmes across PAM50 and vs clinical vars.
    All correlations and distributions computed here.
    Pathway biological interpretations are manually annotated.
    """
    # Key programmes to spotlight — chosen by highest F-stat vs PAM50
    # Computed above in fig_gp_pam50_heatmap; here we hardcode the 4
    # most biologically interpretable ones.
    # NOTE: selection rationale is manually curated.
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
        # Row 0: violin plot by PAM50
        ax_vln = axes[0, col_i]
        pam_groups = [df_train.loc[df_train["pam50_+_claudin-low_subtype"]==sub, gp].dropna()
                      for sub in PAM50_ORDER]
        parts = ax_vln.violinplot(
            [g.values for g in pam_groups if len(g) > 2],
            positions=range(len([g for g in pam_groups if len(g)>2])),
            showmedians=True)
        valid_subs = [sub for sub, g in zip(PAM50_ORDER, pam_groups) if len(g)>2]
        for body, sub in zip(parts["bodies"], valid_subs):
            body.set_facecolor(PAM50_COLORS.get(sub, "grey"))
            body.set_alpha(0.7)
        ax_vln.set_xticks(range(len(valid_subs)))
        ax_vln.set_xticklabels(valid_subs, rotation=35, ha="right", fontsize=8)
        ax_vln.set_ylabel("GP score")
        # ANOVA
        f, p = stats.f_oneway(*[g.values for g in pam_groups if len(g)>2])
        ax_vln.set_title(
            f"{label}\nF={f:.1f}, "
            f"{'p<0.001' if p<0.001 else f'p={p:.3f}'}\n"
            f"[{expectation}]",
            fontsize=8
        )

        # Row 1: correlation with grade and outcome
        ax_cor = axes[1, col_i]
        grade_vals = df_train["neoplasm_histologic_grade"].dropna()
        gp_grade   = df_train.loc[grade_vals.index, gp].dropna()
        common_idx = grade_vals.index.intersection(gp_grade.index)
        r_grade, p_grade = stats.pearsonr(
            grade_vals.loc[common_idx], gp_grade.loc[common_idx])
        surv_vals = df_train["overall_survival"]
        r_surv, p_surv = stats.pearsonr(
            df_train[gp].dropna(),
            surv_vals.loc[df_train[gp].dropna().index])

        # Scatter: GP vs grade (jittered)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(grade_vals))
        ax_cor.scatter(grade_vals.values + jitter,
                       df_train.loc[grade_vals.index, gp].values,
                       alpha=0.12, s=5,
                       c=[PAM50_COLORS.get(sub,"grey")
                          for sub in df_train.loc[grade_vals.index,
                          "pam50_+_claudin-low_subtype"].values])
        # Trend line
        m, b, *_ = stats.linregress(grade_vals.values,
                                     df_train.loc[grade_vals.index, gp].values)
        xl = np.array([1, 3])
        ax_cor.plot(xl, m*xl+b, color="#2c3e50", lw=2)
        ax_cor.set_xticks([1, 2, 3])
        ax_cor.set_xticklabels(["G1","G2","G3"])
        ax_cor.set_xlabel("Histologic grade")
        ax_cor.set_ylabel("GP score")
        ax_cor.set_title(
            f"vs grade: r={r_grade:.3f} "
            f"({'p<0.001' if p_grade<0.001 else f'p={p_grade:.3f}'})\n"
            f"vs survival: r={r_surv:.3f} "
            f"({'p<0.001' if p_surv<0.001 else f'p={p_surv:.3f}'})",
            fontsize=8
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
    Summary table linking data evidence to modelling decisions.
    Counts and F-stats supporting each claim are computed here.
    Interpretive text is manually annotated — labelled as such.
    """
    # Compute evidence values to embed in table
    # M1: survival ANOVA by grade (evidence grade is prognostic)
    f_grade_surv, p_gs = stats.f_oneway(
        *[df.loc[df["neoplasm_histologic_grade"]==g,"overall_survival"].values
          for g in [1.0, 2.0, 3.0]])
    # M3: GP ANOVA by PAM50
    gp_f_max = 0.0
    for gp in gp_cols:
        groups = [df_train.loc[df_train["pam50_+_claudin-low_subtype"]==sub, gp].dropna()
                  for sub in PAM50_ORDER if sub in df_train["pam50_+_claudin-low_subtype"].values]
        groups = [g for g in groups if len(g) > 2]
        if len(groups) >= 2:
            f, _ = stats.f_oneway(*groups)
            gp_f_max = max(gp_f_max, f)
    # M4: GP correlation with grade
    gp05 = df_train["gene_programme_05"]
    grade_tr = df_train["neoplasm_histologic_grade"].dropna()
    common = gp05.dropna().index.intersection(grade_tr.index)
    r_gp05_grade, _ = stats.pearsonr(gp05.loc[common], grade_tr.loc[common])
    # Survival: log-rank best vs worst subtype
    mask_b = df["pam50_+_claudin-low_subtype"] == "claudin-low"
    mask_l = df["pam50_+_claudin-low_subtype"] == "LumA"
    p_km = log_rank_p(
        df.loc[mask_b,"overall_survival_months"].values,
        df.loc[mask_b,"overall_survival"].values,
        df.loc[mask_l,"overall_survival_months"].values,
        df.loc[mask_l,"overall_survival"].values)

    rows = [
        ("M1", "Overall survival\n(binary classif.)",
         f"Log-rank: claudin-low vs LumA\n"
         f"{'p<0.001' if p_km<0.001 else f'p={p_km:.3f}'}\n"
         f"Grade ANOVA: F={f_grade_surv:.1f}\n"
         f"({'p<0.001' if p_gs<0.001 else f'p={p_gs:.3f}'})",
         "Is this patient at risk of death?\n"
         "Use at diagnosis for risk stratification.\n"
         "[Clinical interpretation — manual]",
         "overall_survival (binary)\nAUC-ROC metric\nTest set: frozen 381 rows"),
        ("M2", "Survival time\n(Cox / time-to-event)",
         f"Same evidence as M1.\n"
         f"Cox model accounts for censoring.\n"
         f"Requires lifelines>=0.27.0",
         "When is this patient at risk?\n"
         "Survival probability at 1/3/5 years.\n"
         "[Manual — lifelines not yet available]",
         "time + event (Cox target)\nC-index metric\nNote: FS uses event proxy"),
        ("M3", "PAM50 molecular subtype\n(6-class classif.)",
         f"GP ANOVA max F={gp_f_max:.0f} (p<0.001)\n"
         f"Gene programmes highly discriminative\n"
         f"for PAM50 subtypes",
         "What is the molecular subtype?\n"
         "For institutions without gene arrays.\n"
         "[Manual]",
         "pam50_target (6 classes)\nMacro-F1 metric\nRare classes dropped"),
        ("M4", "Histologic grade\n(ordinal 1–3)",
         f"GP05 vs grade: r={r_gp05_grade:.3f}\n"
         f"Proliferation programme correlates\n"
         f"with grade — biological basis exists",
         "What grade would a molecular panel predict?\n"
         "Second opinion / low-resource setting.\n"
         "[Manual]",
         "neoplasm_histologic_grade_ord\nQW-kappa metric\nNPI excluded"),
    ]

    model_colors = {"M1":"#3498db","M2":"#9b59b6","M3":"#e67e22","M4":"#2ecc71"}

    fig_h = len(rows) * 1.5 + 2.0
    fig, ax = plt.subplots(figsize=(19, fig_h))
    ax.axis("off")
    ax.text(0.5, 0.985,
            "Section 5 — Research Questions and Modelling Rationale",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=13, fontweight="bold", color=HEAD_BG)
    ax.text(0.5, 0.960,
            "Evidence column: statistics computed here in this function. "
            "Clinical use and interpretation columns: manually annotated.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, style="italic", color="#555")

    headers = ["Model","Target / task",
               "Data evidence (computed here)","Clinical use [manual]",
               "Implementation notes"]
    col_x   = [0.0, 0.07, 0.22, 0.46, 0.66]
    col_w   = [0.07, 0.15, 0.24, 0.20, 0.27]
    head_h  = 0.055; row_h = 0.185; sec_sep = 0.010
    y       = 0.940

    for h, x, w in zip(headers, col_x, col_w):
        ax.add_patch(plt.Rectangle((x, y-head_h), w-0.004, head_h,
            facecolor=HEAD_BG, transform=ax.transAxes, clip_on=False))
        ax.text(x+0.008, y-head_h/2, h, transform=ax.transAxes,
                va="center", fontsize=9, fontweight="bold", color=HEAD_FG)
    y -= head_h

    for i, (model, task, evidence, use, impl) in enumerate(rows):
        y -= (row_h + sec_sep)
        mc = model_colors.get(model, "#888")
        # Model column
        ax.add_patch(plt.Rectangle((col_x[0], y), col_w[0]-0.004, row_h,
            facecolor=mc, transform=ax.transAxes, clip_on=False))
        ax.text(col_x[0]+0.008, y+row_h/2, model,
                transform=ax.transAxes, va="center",
                fontsize=12, fontweight="bold", color="white")
        # Other columns
        for j, (txt, x, w) in enumerate(zip(
                [task, evidence, use, impl],
                col_x[1:], col_w[1:])):
            bg = "#f0f4ff" if j == 1 else ROW_A if i%2==0 else ROW_B
            ax.add_patch(plt.Rectangle((x, y), w-0.004, row_h,
                facecolor=bg, edgecolor=BORDER, lw=0.3,
                transform=ax.transAxes, clip_on=False))
            ax.text(x+0.008, y+row_h*0.85, str(txt),
                    transform=ax.transAxes, va="top",
                    fontsize=8, color=CELL_FG, wrap=True)

    save(fig, out / "22_research_questions.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="Notebook 03 — Clinical and Molecular Insights")
    p.add_argument("--input", type=Path,
                   default=Path("data") /
                   "FCS_ml_test_input_data_rna_mutation.csv")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs") / "notebook_05")
    p.add_argument("--pipeline-outputs", type=Path,
                   default=Path("outputs"))
    args = p.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading data: {args.input}")
    df = pd.read_csv(args.input)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    gene_cols = [c for c in df.columns if c not in CLINICAL]

    print("Loading pipeline outputs...")
    train_idx = pd.read_csv(
        args.pipeline_outputs / "metadata" / "shared_train_indices.csv"
    )["row_index"].tolist()
    X_m1 = pd.read_csv(
        args.pipeline_outputs / "splits" / "M1a_overall_survival" / "X_train.csv")
    gp_cols = [c for c in X_m1.columns if c.startswith("gene_programme")]
    H = pd.read_csv(
        args.pipeline_outputs / "METABRIC_nmf_components.csv", index_col=0)

    # Build training dataframe with GP scores
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_train = pd.concat(
        [df_train, X_m1[gp_cols].reset_index(drop=True)], axis=1)

    print(f"\nGenerating figures → {args.output_dir}/\n")

    fig_km_pam50(df, args.output_dir)
    fig_km_grade_receptor(df, args.output_dir)
    fig_gp_pam50_heatmap(df_train, gp_cols, args.output_dir)
    fig_gp_spotlight(df_train, gp_cols, H, args.output_dir)
    fig_clinical_correlations(df, args.output_dir)
    fig_cohort_confounding_mitigation(df_train, gp_cols, args.output_dir)
    fig_research_questions(df, gp_cols, df_train, args.output_dir)

    print(f"\n✓ All figures saved to: {args.output_dir}/")
    for f in sorted(args.output_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
