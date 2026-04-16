"use client";

import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  LineChart,
  Line,
  ReferenceLine,
  Legend,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  ClipboardList,
  FlaskConical,
  Info,
  ShieldAlert,
  Stethoscope,
  Gauge,
  Layers3,
  Microscope,
  Loader2,
} from "lucide-react";

type RiskTier = "Low" | "Intermediate" | "High";
type StabilityTask = "M1a" | "M1b" | "M3" | "M4";

type PipelineFormState = {
  patient_id: string;
  age_at_diagnosis_imputed: number;
  tumor_size: number;
  lymph_nodes_examined_positive: number;
  neoplasm_histologic_grade_ord: "1" | "2" | "3";
  er_status_bin: 0 | 1;
  her2_status_bin: 0 | 1;
  pr_status_bin: 0 | 1;
  pam50_subtype: "LumA" | "LumB" | "Her2" | "Basal" | "Normal" | "claudin-low";
  gene_programme_01: number;
  gene_programme_05: number;
  gene_programme_10: number;
  gene_programme_14: number;
};

type Contribution = {
  feature: string;
  display: string;
  value: string | number;
  points: number;
  rationale: string;
};

type ModelSummaryRow = {
  model: string;
  taskKey: string;
  algorithm: string;
  metric: string;
  score: number;
  note?: string;
};

type BinaryMetricsSummary = {
  algorithm: string;
  auc: number;
  prAuc: number;
  brier: number;
  ece: number;
  calibrationIntercept: number;
  calibrationSlope: number;
};

type ThresholdRow = {
  threshold: number;
  sensitivity: number;
  specificity: number;
  ppv: number;
  npv: number;
};

type RiskGroupRow = {
  group: string;
  eventRate: number;
  medianTime: number;
  n: number;
};

type CompetingRiskRow = {
  group: string;
  cancer60m: number;
  otherCause60m: number;
};

type FeatureSetRow = {
  task: StabilityTask;
  set: string;
  nFeatures: number;
  score: number;
};

type RepeatedOuterPoint = {
  task: StabilityTask;
  repeat: number;
  score: number;
};

type CohortTransportRow = {
  task: StabilityTask;
  cohort: number;
  score: number;
  n: number;
  metric: string;
};

type DashboardData = {
  modelSummary: ModelSummaryRow[];
  m1bBinaryMetrics: BinaryMetricsSummary | null;
  m1bThresholds: ThresholdRow[];
  m2bRiskGroups: RiskGroupRow[];
  m2bCompetingRisks: CompetingRiskRow[];
  featureRepresentation: FeatureSetRow[];
  repeatedOuter: RepeatedOuterPoint[];
  cohortTransport: CohortTransportRow[];
};

type DataPaths = {
  modelSummary: string;
  m1bBinaryMetrics: string;
  m1bThresholds: string;
  m2bRiskGroups: string;
  m2bCompetingRisks: string;
  featureRepresentation?: string;
  repeatedOuter?: string;
  cohortTransport?: string;
};

const DEFAULT_DATA_PATHS: DataPaths = {
  modelSummary: "/pipeline-data/30_model_summary.json",
  m1bBinaryMetrics: "/pipeline-data/28c_m1b_binary_metrics_summary.json",
  m1bThresholds: "/pipeline-data/28c_m1b_threshold_metrics.json",
  m2bRiskGroups: "/pipeline-data/28d_m2b_risk_group_summary.json",
  m2bCompetingRisks: "/pipeline-data/54_m2b_competing_risk_summary.json",
  featureRepresentation: "/pipeline-data/29_feature_representation_scores.json",
  repeatedOuter: "/pipeline-data/32_repeated_outer_validation.json",
  cohortTransport: "/pipeline-data/31_cohort_transport_sensitivity.json",
};

const defaultState: PipelineFormState = {
  patient_id: "MB-2026-014",
  age_at_diagnosis_imputed: 58,
  tumor_size: 28,
  lymph_nodes_examined_positive: 3,
  neoplasm_histologic_grade_ord: "3",
  er_status_bin: 0,
  her2_status_bin: 1,
  pr_status_bin: 0,
  pam50_subtype: "Her2",
  gene_programme_01: 0.18,
  gene_programme_05: 0.74,
  gene_programme_10: 0.81,
  gene_programme_14: 0.34,
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function log1pSafe(x: number) {
  return Math.log1p(Math.max(0, x));
}

function formatPct(v: number) {
  return `${Math.round(v * 100)}%`;
}

function formatNumber(v: number, digits = 3) {
  return Number.isFinite(v) ? v.toFixed(digits) : "NA";
}

function asArray(data: unknown): Record<string, unknown>[] {
  if (Array.isArray(data)) return data.filter((d): d is Record<string, unknown> => !!d && typeof d === "object");
  if (data && typeof data === "object" && "rows" in data) {
    return asArray((data as { rows?: unknown }).rows);
  }
  return [];
}

function pickNumber(row: Record<string, unknown>, keys: string[]): number {
  for (const key of keys) {
    const raw = row[key];
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
    if (typeof raw === "string" && raw.trim() !== "") {
      const parsed = Number(raw);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return NaN;
}

function pickString(row: Record<string, unknown>, keys: string[]): string {
  for (const key of keys) {
    const raw = row[key];
    if (typeof raw === "string" && raw.trim() !== "") return raw.trim();
    if (typeof raw === "number" && Number.isFinite(raw)) return String(raw);
  }
  return "";
}

function normalizeTaskKey(input: string): string {
  const value = input.toLowerCase();
  if (value.includes("m1a")) return "M1a";
  if (value.includes("m1b")) return "M1b";
  if (value.includes("m2a")) return "M2a";
  if (value.includes("m2b")) return "M2b";
  if (value.includes("m3")) return "M3";
  if (value.includes("m4")) return "M4";
  return input;
}

function humanModelLabel(taskKey: string, fallback: string) {
  const map: Record<string, string> = {
    M1a: "M1a overall survival",
    M1b: "M1b cancer-specific survival",
    M2a: "M2a overall survival Cox",
    M2b: "M2b cancer-specific Cox",
    M3: "M3 PAM50 subtype",
    M4: "M4 histologic grade",
  };
  return map[taskKey] ?? fallback;
}

function buildContributions(state: PipelineFormState): Contribution[] {
  const tumor_size_log = log1pSafe(state.tumor_size);
  const lymph_nodes_examined_positive_log = log1pSafe(state.lymph_nodes_examined_positive);

  return [
    {
      feature: "age_at_diagnosis_imputed",
      display: "age_at_diagnosis_imputed",
      value: state.age_at_diagnosis_imputed,
      points: clamp((state.age_at_diagnosis_imputed - 50) * 0.22, -4, 10),
      rationale: state.age_at_diagnosis_imputed >= 65
        ? "Older age still contributes risk, although less cleanly than in all-cause survival."
        : "Age contributes modestly at this level.",
    },
    {
      feature: "tumor_size_log",
      display: "tumor_size_log",
      value: tumor_size_log.toFixed(2),
      points: clamp((tumor_size_log - 2.8) * 7.5, -4, 12),
      rationale: state.tumor_size >= 20
        ? "Larger tumour burden increases cancer-specific risk."
        : "Smaller tumour size lowers the risk contribution.",
    },
    {
      feature: "lymph_nodes_examined_positive_log",
      display: "lymph_nodes_examined_positive_log",
      value: lymph_nodes_examined_positive_log.toFixed(2),
      points: clamp((lymph_nodes_examined_positive_log - 0.6) * 12, -2, 18),
      rationale: state.lymph_nodes_examined_positive > 0
        ? "Positive lymph nodes are among the strongest adverse signals in the pipeline."
        : "Node-negative disease reduces cancer-specific risk.",
    },
    {
      feature: "neoplasm_histologic_grade_ord",
      display: "neoplasm_histologic_grade_ord",
      value: state.neoplasm_histologic_grade_ord,
      points: state.neoplasm_histologic_grade_ord === "3" ? 11 : state.neoplasm_histologic_grade_ord === "2" ? 5 : 0,
      rationale: state.neoplasm_histologic_grade_ord === "3"
        ? "Grade 3 supports more aggressive tumour biology."
        : "Lower histologic grade is relatively favorable.",
    },
    {
      feature: "er_status_bin",
      display: "er_status_bin",
      value: state.er_status_bin,
      points: state.er_status_bin === 0 ? 7 : -3,
      rationale: state.er_status_bin === 0
        ? "ER negativity raises cancer-specific risk."
        : "ER positivity is prognostically favorable.",
    },
    {
      feature: "her2_status_bin",
      display: "her2_status_bin",
      value: state.her2_status_bin,
      points: state.her2_status_bin === 1 ? 4 : 0,
      rationale: state.her2_status_bin === 1
        ? "HER2 positivity can increase risk without treatment context."
        : "No additional HER2-associated risk signal.",
    },
    {
      feature: "pr_status_bin",
      display: "pr_status_bin",
      value: state.pr_status_bin,
      points: state.pr_status_bin === 0 ? 2.5 : -1.5,
      rationale: state.pr_status_bin === 0
        ? "PR negativity modestly worsens prognosis."
        : "PR positivity modestly offsets risk.",
    },
    {
      feature: "pam50_subtype",
      display: "pam50 subtype",
      value: state.pam50_subtype,
      points: state.pam50_subtype === "Basal"
          ? 13
          : state.pam50_subtype === "Her2"
            ? 9
            : state.pam50_subtype === "LumB"
              ? 5
              : state.pam50_subtype === "LumA"
                ? -5
                : 0,
      rationale: state.pam50_subtype === "Basal"
        ? "Basal subtype carries the strongest adverse molecular signal."
        : state.pam50_subtype === "LumA"
          ? "LumA lowers cancer-specific risk."
          : "Subtype contributes a moderate molecular signal.",
    },
    {
      feature: "gene_programme_05",
      display: "gene_programme_05 (proliferation)",
      value: state.gene_programme_05.toFixed(2),
      points: clamp((state.gene_programme_05 - 0.3) * 17, -5, 14),
      rationale: state.gene_programme_05 >= 0.5
        ? "High GP05 indicates stronger proliferation biology."
        : "Limited proliferation signal.",
    },
    {
      feature: "gene_programme_10",
      display: "gene_programme_10 (HER2/epithelial)",
      value: state.gene_programme_10.toFixed(2),
      points: clamp((state.gene_programme_10 - 0.3) * 10, -4, 8),
      rationale: state.gene_programme_10 >= 0.5
        ? "GP10 supports HER2-associated tumour biology."
        : "Limited HER2 programme contribution.",
    },
    {
      feature: "gene_programme_14",
      display: "gene_programme_14 (immune/IFN)",
      value: state.gene_programme_14.toFixed(2),
      points: clamp((0.35 - state.gene_programme_14) * 8, -6, 6),
      rationale: state.gene_programme_14 >= 0.5
        ? "Immune programme may partially offset risk."
        : "Low immune programme provides less protective signal.",
    },
    {
      feature: "gene_programme_01",
      display: "gene_programme_01 (luminal/hormone)",
      value: state.gene_programme_01.toFixed(2),
      points: clamp((0.3 - state.gene_programme_01) * 8, -5, 5),
      rationale: state.gene_programme_01 >= 0.5
        ? "Luminal programme is relatively favorable."
        : "Weak luminal programme contributes less favorable biology.",
    },
  ];
}

function calculateM1b(state: PipelineFormState) {
  const contributions = buildContributions(state);
  const score = contributions.reduce((sum, c) => sum + c.points, 0);
  const probability = clamp(1 / (1 + Math.exp(-(score - 16) / 10)), 0.03, 0.95);
  const riskPercent = Math.round(probability * 100);
  let tier: RiskTier = "Low";
  if (riskPercent >= 65) tier = "High";
  else if (riskPercent >= 35) tier = "Intermediate";

  const horizons = [12, 24, 36, 60].map((months, idx) => {
    const decay = [0.10, 0.19, 0.29, 0.46][idx];
    const survival = clamp(1 - probability * decay, 0.1, 0.99);
    return { horizon: months, label: `${months / 12}y`, survival: Math.round(survival * 100) };
  });

  const ranked = [...contributions].sort((a, b) => Math.abs(b.points) - Math.abs(a.points)).slice(0, 6);
  const operationalThreshold = 0.4;
  const operationalStatus = probability >= operationalThreshold ? "Escalate" : "Routine / contextual review";

  return { score, probability, riskPercent, tier, horizons, contributions, ranked, operationalThreshold, operationalStatus };
}

function calculateM2b(state: PipelineFormState, m1bProbability: number) {
  const lp =
    (state.age_at_diagnosis_imputed - 60) * 0.01 +
    log1pSafe(state.tumor_size) * 0.18 +
    log1pSafe(state.lymph_nodes_examined_positive) * 0.36 +
    (state.neoplasm_histologic_grade_ord === "3" ? 0.34 : state.neoplasm_histologic_grade_ord === "2" ? 0.12 : 0) +
    (state.er_status_bin === 0 ? 0.22 : -0.08) +
    (state.her2_status_bin === 1 ? 0.10 : 0) +
    (state.pr_status_bin === 0 ? 0.06 : -0.03) +
    (state.pam50_subtype === "Basal" ? 0.30 : state.pam50_subtype === "Her2" ? 0.22 : state.pam50_subtype === "LumA" ? -0.15 : 0) +
    (state.gene_programme_05 - 0.3) * 0.45 +
    (state.gene_programme_10 - 0.3) * 0.20 +
    (0.35 - state.gene_programme_14) * 0.14 +
    (0.30 - state.gene_programme_01) * 0.14;

  const hazardRatio = Math.exp(lp);
  const baseline = [
    { months: 12, label: "1y", base: 0.96 },
    { months: 24, label: "2y", base: 0.90 },
    { months: 36, label: "3y", base: 0.84 },
    { months: 60, label: "5y", base: 0.72 },
  ];

  const survivalCurve = baseline.map((b) => ({
    horizon: b.months,
    label: b.label,
    survival: Math.round(Math.pow(b.base, hazardRatio) * 100),
  }));

  const fiveYear = survivalCurve.find((d) => d.horizon === 60)?.survival ?? 72;
  const medianSurvivalBand =
    fiveYear >= 80 ? "> 10 years" : fiveYear >= 65 ? "7–10 years" : fiveYear >= 45 ? "4–7 years" : "< 4 years";

  return {
    hazardRatio: Number(hazardRatio.toFixed(2)),
    survivalCurve,
    fiveYearSurvival: fiveYear,
    medianSurvivalBand,
    cIndexLabel: "Illustrative M2b Cox view",
    linkedM1bRisk: Math.round(m1bProbability * 100),
  };
}

function tierTone(tier: RiskTier) {
  if (tier === "High") return "danger";
  if (tier === "Intermediate") return "warning";
  return "success";
}

function toneClass(tone: "default" | "success" | "warning" | "danger" | "info") {
  return `tone-${tone}`;
}

async function fetchJson(path: string) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load ${path} (${res.status})`);
  return res.json();
}

function normalizeModelSummary(data: unknown): ModelSummaryRow[] {
  return asArray(data)
    .map((row) => {
      const rawModel = pickString(row, ["Model", "model", "task", "Task"]);
      const taskKey = normalizeTaskKey(rawModel);
      return {
        model: humanModelLabel(taskKey, rawModel || taskKey),
        taskKey,
        algorithm: pickString(row, ["algorithm", "Algorithm", "winner", "best_algorithm"]),
        metric: pickString(row, ["metric", "Metric"]),
        score: pickNumber(row, ["score", "Score", "test_score", "value"]),
        note: pickString(row, ["note", "Note"]),
      };
    })
    .filter((row) => row.model && Number.isFinite(row.score));
}

function normalizeBinaryMetrics(data: unknown, modelSummary: ModelSummaryRow[]): BinaryMetricsSummary | null {
  const rows = asArray(data);
  const row = rows[0] ?? {};
  const m1bWinner = modelSummary.find((d) => d.taskKey === "M1b")?.algorithm ?? "";
  const auc = pickNumber(row, ["roc_auc", "auc", "AUC", "auc_roc"]);
  const prAuc = pickNumber(row, ["pr_auc", "prauc", "prAUC"]);
  const brier = pickNumber(row, ["brier_score", "brier", "Brier"]);
  const ece = pickNumber(row, ["ece", "ECE"]);
  const calibrationIntercept = pickNumber(row, ["calibration_intercept", "intercept", "calibrationIntercept"]);
  const calibrationSlope = pickNumber(row, ["calibration_slope", "slope", "calibrationSlope"]);
  if (![auc, prAuc, brier, ece, calibrationIntercept, calibrationSlope].some(Number.isFinite)) return null;
  return {
    algorithm: pickString(row, ["algorithm", "Algorithm"]) || m1bWinner || "Unknown",
    auc, prAuc, brier, ece, calibrationIntercept, calibrationSlope,
  };
}

function normalizeThresholds(data: unknown): ThresholdRow[] {
  return asArray(data)
    .map((row) => ({
      threshold: pickNumber(row, ["threshold", "Threshold"]),
      sensitivity: pickNumber(row, ["sensitivity", "Sensitivity"]),
      specificity: pickNumber(row, ["specificity", "Specificity"]),
      ppv: pickNumber(row, ["ppv", "PPV", "precision"]),
      npv: pickNumber(row, ["npv", "NPV"]),
    }))
    .filter((row) => Number.isFinite(row.threshold))
    .sort((a, b) => a.threshold - b.threshold);
}

function normalizeRiskGroups(data: unknown): RiskGroupRow[] {
  return asArray(data)
    .map((row) => ({
      group: pickString(row, ["group", "risk_group", "Group"]),
      eventRate: pickNumber(row, ["event_rate", "eventRate", "cancer_specific_event_rate"]),
      medianTime: pickNumber(row, ["median_time", "medianTime", "median_time_months"]),
      n: pickNumber(row, ["n", "N"]),
    }))
    .filter((row) => row.group);
}

function normalizeCompetingRisks(data: unknown): CompetingRiskRow[] {
  return asArray(data)
    .map((row) => ({
      group: pickString(row, ["group", "risk_group", "Group"]),
      cancer60m: pickNumber(row, ["cancer_60m", "cancer60m", "cancer_cif_60m", "cancer_specific_cif_60m"]),
      otherCause60m: pickNumber(row, ["other_cause_60m", "otherCause60m", "other_cause_cif_60m"]),
    }))
    .filter((row) => row.group);
}

function normalizeFeatureRepresentation(data: unknown): FeatureSetRow[] {
  return asArray(data)
    .map((row) => ({
      task: normalizeTaskKey(pickString(row, ["task", "Task", "model", "Model"])) as StabilityTask,
      set: pickString(row, ["set", "feature_set", "Set"]),
      nFeatures: pickNumber(row, ["n_features", "nFeatures", "features"]),
      score: pickNumber(row, ["score", "Score", "cv_score"]),
    }))
    .filter((row) => row.task && row.set && Number.isFinite(row.score));
}

function normalizeRepeatedOuter(data: unknown): RepeatedOuterPoint[] {
  return asArray(data)
    .map((row) => ({
      task: normalizeTaskKey(pickString(row, ["task", "Task", "model", "Model"])) as StabilityTask,
      repeat: pickNumber(row, ["repeat", "Repeat", "fold", "split"]),
      score: pickNumber(row, ["score", "Score", "cv_score", "outer_score"]),
    }))
    .filter((row) => row.task && Number.isFinite(row.repeat) && Number.isFinite(row.score));
}

function normalizeCohortTransport(data: unknown): CohortTransportRow[] {
  return asArray(data)
    .map((row) => ({
      task: normalizeTaskKey(pickString(row, ["task", "Task", "model", "Model"])) as StabilityTask,
      cohort: pickNumber(row, ["cohort", "Cohort"]),
      score: pickNumber(row, ["score", "Score"]),
      n: pickNumber(row, ["n", "N"]),
      metric: pickString(row, ["metric", "Metric"]),
    }))
    .filter((row) => row.task && Number.isFinite(row.cohort) && Number.isFinite(row.score));
}

function usePipelineDashboardData(paths: DataPaths) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [modelSummaryRaw, m1bBinaryRaw, m1bThresholdsRaw, m2bRiskGroupsRaw, m2bCompetingRaw, featureRepresentationRaw, repeatedOuterRaw, cohortTransportRaw] =
          await Promise.all([
            fetchJson(paths.modelSummary),
            fetchJson(paths.m1bBinaryMetrics),
            fetchJson(paths.m1bThresholds),
            fetchJson(paths.m2bRiskGroups),
            fetchJson(paths.m2bCompetingRisks),
            paths.featureRepresentation ? fetchJson(paths.featureRepresentation).catch(() => []) : Promise.resolve([]),
            paths.repeatedOuter ? fetchJson(paths.repeatedOuter).catch(() => []) : Promise.resolve([]),
            paths.cohortTransport ? fetchJson(paths.cohortTransport).catch(() => []) : Promise.resolve([]),
          ]);

        const modelSummary = normalizeModelSummary(modelSummaryRaw);
        const nextData: DashboardData = {
          modelSummary,
          m1bBinaryMetrics: normalizeBinaryMetrics(m1bBinaryRaw, modelSummary),
          m1bThresholds: normalizeThresholds(m1bThresholdsRaw),
          m2bRiskGroups: normalizeRiskGroups(m2bRiskGroupsRaw),
          m2bCompetingRisks: normalizeCompetingRisks(m2bCompetingRaw),
          featureRepresentation: normalizeFeatureRepresentation(featureRepresentationRaw),
          repeatedOuter: normalizeRepeatedOuter(repeatedOuterRaw),
          cohortTransport: normalizeCohortTransport(cohortTransportRaw),
        };
        if (!cancelled) setData(nextData);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load pipeline JSON files.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [paths]);

  return { data, loading, error };
}

function Card({ children, className = "" }: React.PropsWithChildren<{ className?: string }>) {
  return <div className={`card ${className}`}>{children}</div>;
}

function ButtonTab({
  active,
  onClick,
  children,
}: React.PropsWithChildren<{ active: boolean; onClick: () => void }>) {
  return (
    <button type="button" onClick={onClick} className={`tab ${active ? "tab-active" : ""}`}>
      {children}
    </button>
  );
}

function Badge({ children, tone = "default" }: React.PropsWithChildren<{ tone?: "default" | "success" | "warning" | "danger" | "info" }>) {
  return <span className={`badge ${toneClass(tone)}`}>{children}</span>;
}

function Alert({
  title,
  children,
  tone = "default",
  icon,
}: React.PropsWithChildren<{ title: string; tone?: "default" | "success" | "warning" | "danger" | "info"; icon?: React.ReactNode }>) {
  return (
    <div className={`alert ${toneClass(tone)}`}>
      <div className="alert-icon">{icon}</div>
      <div className="alert-copy">
        <div className="alert-title">{title}</div>
        <div className="alert-body">{children}</div>
      </div>
    </div>
  );
}

function StatCard({ label, value, children }: React.PropsWithChildren<{ label: string; value?: string }>) {
  return (
    <Card>
      <div className="stat-label">{label}</div>
      {value !== undefined ? <div className="stat-value">{value}</div> : null}
      {children}
    </Card>
  );
}

export default function ClinicianCancerSpecificRiskDashboard({ dataPaths = DEFAULT_DATA_PATHS }: { dataPaths?: DataPaths }) {
  const [state, setState] = useState<PipelineFormState>(defaultState);
  const [activeTab, setActiveTab] = useState<"patient" | "locked" | "thresholds" | "stability">("patient");
  const [selectedThreshold, setSelectedThreshold] = useState<string>("0.40");
  const [selectedStabilityTask, setSelectedStabilityTask] = useState<StabilityTask>("M1b");

  const { data, loading, error } = usePipelineDashboardData(dataPaths);
  const m1b = useMemo(() => calculateM1b(state), [state]);
  const m2b = useMemo(() => calculateM2b(state, m1b.probability), [state, m1b.probability]);

  const update = <K extends keyof PipelineFormState>(key: K, value: PipelineFormState[K]) => {
    setState((prev) => ({ ...prev, [key]: value }));
  };

  const contributionChart = m1b.ranked.map((c) => ({
    feature: c.display,
    points: Number(c.points.toFixed(1)),
  }));

  const thresholdRows = data?.m1bThresholds ?? [];
  const thresholdRow = useMemo(
    () => thresholdRows.find((d) => d.threshold.toFixed(2) === selectedThreshold) ?? thresholdRows[0] ?? null,
    [selectedThreshold, thresholdRows],
  );

  const modelSummary = data?.modelSummary ?? [];
  const m1bModel = modelSummary.find((d) => d.taskKey === "M1b");
  const m2bModel = modelSummary.find((d) => d.taskKey === "M2b");
  const binaryMetrics = data?.m1bBinaryMetrics;

  const featureRepresentation = data?.featureRepresentation ?? [];
  const repeatedOuter = data?.repeatedOuter ?? [];
  const cohortTransport = data?.cohortTransport ?? [];

  const featureSetForTask = featureRepresentation.filter((d) => d.task === selectedStabilityTask);
  const repeatedSeries = repeatedOuter.filter((d) => d.task === selectedStabilityTask).sort((a, b) => a.repeat - b.repeat);
  const transportSeries = cohortTransport.filter((d) => d.task === selectedStabilityTask).sort((a, b) => a.cohort - b.cohort);

  const repeatedMeta = useMemo(() => {
    if (repeatedSeries.length === 0) return null;
    const scores = repeatedSeries.map((d) => d.score);
    const mean = scores.reduce((sum, v) => sum + v, 0) / scores.length;
    const variance = scores.reduce((sum, v) => sum + (v - mean) ** 2, 0) / scores.length;
    const metric = transportSeries[0]?.metric || modelSummary.find((d) => d.taskKey === selectedStabilityTask)?.metric || "score";
    return { mean, sd: Math.sqrt(variance), metric };
  }, [repeatedSeries, transportSeries, modelSummary, selectedStabilityTask]);

  return (
    <div className="page-shell">
      <div className="page-grid">
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="hero-grid">
          <Card>
            <div className="hero-head">
              <div className="icon-bubble"><Stethoscope size={20} /></div>
              <div>
                <h1 className="title">M1b / M2b clinician dashboard connected to exported pipeline JSON</h1>
                <p className="subtitle">
                  Illustrative patient sandbox plus live benchmark layer for the final pipeline outputs.
                </p>
              </div>
            </div>

            {loading ? (
              <Alert title="Loading pipeline outputs" tone="info" icon={<Loader2 className="spin" size={16} />}>
                Reading exported JSON artefacts from <code>/pipeline-data/</code>.
              </Alert>
            ) : error ? (
              <Alert title="Could not load benchmark files" tone="danger" icon={<AlertTriangle size={16} />}>
                {error}
              </Alert>
            ) : (
              <Alert title="Live benchmark layer" tone="info" icon={<Info size={16} />}>
                Hardcoded benchmark constants have been removed. This dashboard renders from exported JSON artefacts and
                updates when those files change.
              </Alert>
            )}

            <div className="top-cards">
              <StatCard label="patient_id">
                <div className="patient-id">{state.patient_id}</div>
              </StatCard>

              <StatCard label="M1b cancer-specific risk" value={`${m1b.riskPercent}%`} />

              <StatCard label="M1b risk tier">
                <div className="mt-12">
                  <Badge tone={tierTone(m1b.tier)}>{m1b.tier}</Badge>
                </div>
              </StatCard>

              <StatCard label="M2b hazard ratio" value={String(m2b.hazardRatio)} />
            </div>
          </Card>

          <Card>
            <h2 className="section-title">Locked benchmark summary</h2>
            <p className="section-subtitle">Directly sourced from the final pipeline artefacts.</p>

            <div className="two-col">
              <div className="subcard">
                <div className="mini-label">M1b locked winner</div>
                <div className="mini-value">{m1bModel?.algorithm || "—"}</div>
                <div className="mini-copy">
                  AUC {binaryMetrics ? formatNumber(binaryMetrics.auc) : "—"} · PR-AUC {binaryMetrics ? formatNumber(binaryMetrics.prAuc) : "—"}
                </div>
              </div>

              <div className="subcard">
                <div className="mini-label">M2b locked winner</div>
                <div className="mini-value">{m2bModel?.algorithm || "—"}</div>
                <div className="mini-copy">
                  {m2bModel?.metric || "metric"} {m2bModel ? formatNumber(m2bModel.score) : "—"} · competing-risks sensitivity reported separately
                </div>
              </div>
            </div>

            <div className="spaced-row">
              <span className="mini-copy">Operational threshold shown below</span>
              <span className="mini-copy">{selectedThreshold}</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Number(selectedThreshold) * 100}%` }} />
            </div>

            <Alert
              title={m1b.operationalStatus}
              tone={tierTone(m1b.tier)}
              icon={m1b.tier === "High" ? <ShieldAlert size={16} /> : <Activity size={16} />}
            >
              {thresholdRow
                ? `At threshold ${selectedThreshold}, the locked test metrics are sensitivity ${formatPct(thresholdRow.sensitivity)}, specificity ${formatPct(thresholdRow.specificity)}, PPV ${formatPct(thresholdRow.ppv)} and NPV ${formatPct(thresholdRow.npv)}.`
                : "Threshold metrics will appear once the threshold JSON is loaded."}
            </Alert>

            <div className="summary-list">
              <div className="summary-item">
                <span className="mini-copy">M2b median survival band</span>
                <strong>{m2b.medianSurvivalBand}</strong>
              </div>
              {m2b.survivalCurve.map((h) => (
                <div key={h.horizon} className="summary-item">
                  <span className="mini-copy">Illustrative {h.label} cancer-specific survival</span>
                  <strong>{h.survival}%</strong>
                </div>
              ))}
            </div>
          </Card>
        </motion.div>

        <div className="tabs-wrap">
          <div className="tabs-row">
            <ButtonTab active={activeTab === "patient"} onClick={() => setActiveTab("patient")}>Patient sandbox</ButtonTab>
            <ButtonTab active={activeTab === "locked"} onClick={() => setActiveTab("locked")}>Locked models</ButtonTab>
            <ButtonTab active={activeTab === "thresholds"} onClick={() => setActiveTab("thresholds")}>Calibration / thresholds</ButtonTab>
            <ButtonTab active={activeTab === "stability"} onClick={() => setActiveTab("stability")}>Stability / transport</ButtonTab>
          </div>
        </div>

        {activeTab === "patient" && (
          <div className="main-grid">
            <Card>
              <h2 className="section-title">Pipeline-aligned patient inputs</h2>
              <p className="section-subtitle">Inputs mirror engineered variables and major biological programmes used downstream.</p>

              <div className="form-grid">
                <label className="field">
                  <span>patient_id</span>
                  <input value={state.patient_id} onChange={(e) => update("patient_id", e.target.value)} />
                </label>

                <label className="field">
                  <span>age_at_diagnosis_imputed</span>
                  <input type="number" value={state.age_at_diagnosis_imputed} onChange={(e) => update("age_at_diagnosis_imputed", Number(e.target.value))} />
                </label>

                <label className="field">
                  <span>tumor_size (mm)</span>
                  <input type="number" value={state.tumor_size} onChange={(e) => update("tumor_size", Number(e.target.value))} />
                </label>

                <label className="field">
                  <span>lymph_nodes_examined_positive</span>
                  <input type="number" value={state.lymph_nodes_examined_positive} onChange={(e) => update("lymph_nodes_examined_positive", Number(e.target.value))} />
                </label>

                <label className="field">
                  <span>neoplasm_histologic_grade_ord</span>
                  <select value={state.neoplasm_histologic_grade_ord} onChange={(e) => update("neoplasm_histologic_grade_ord", e.target.value as PipelineFormState["neoplasm_histologic_grade_ord"])}>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                  </select>
                </label>

                <label className="field">
                  <span>er_status_bin</span>
                  <select value={String(state.er_status_bin)} onChange={(e) => update("er_status_bin", Number(e.target.value) as 0 | 1)}>
                    <option value="1">1 (Positive)</option>
                    <option value="0">0 (Negative)</option>
                  </select>
                </label>

                <label className="field">
                  <span>her2_status_bin</span>
                  <select value={String(state.her2_status_bin)} onChange={(e) => update("her2_status_bin", Number(e.target.value) as 0 | 1)}>
                    <option value="1">1 (Positive)</option>
                    <option value="0">0 (Negative)</option>
                  </select>
                </label>

                <label className="field">
                  <span>pr_status_bin</span>
                  <select value={String(state.pr_status_bin)} onChange={(e) => update("pr_status_bin", Number(e.target.value) as 0 | 1)}>
                    <option value="1">1 (Positive)</option>
                    <option value="0">0 (Negative)</option>
                  </select>
                </label>

                <label className="field full">
                  <span>pam50 subtype</span>
                  <select value={state.pam50_subtype} onChange={(e) => update("pam50_subtype", e.target.value as PipelineFormState["pam50_subtype"])}>
                    <option value="LumA">LumA</option>
                    <option value="LumB">LumB</option>
                    <option value="Her2">Her2</option>
                    <option value="Basal">Basal</option>
                    <option value="Normal">Normal</option>
                    <option value="claudin-low">claudin-low</option>
                  </select>
                </label>

                {[
                  ["gene_programme_05", state.gene_programme_05],
                  ["gene_programme_10", state.gene_programme_10],
                  ["gene_programme_14", state.gene_programme_14],
                  ["gene_programme_01", state.gene_programme_01],
                ].map(([key, value]) => (
                  <label key={key} className="field">
                    <div className="field-head">
                      <span>{key}</span>
                      <span className="mini-copy">{value}</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      step={1}
                      value={Number(value) * 100}
                      onChange={(e) => update(key as keyof PipelineFormState, Number((Number(e.target.value) / 100).toFixed(2)) as PipelineFormState[keyof PipelineFormState])}
                    />
                  </label>
                ))}
              </div>
            </Card>

            <div className="side-stack">
              <Card className="chart-card">
                <h2 className="section-title">M1b top risk drivers</h2>
                <p className="section-subtitle">Largest absolute contributions to the current illustrative M1b estimate.</p>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={contributionChart} layout="vertical" margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={150} />
                      <Tooltip />
                      <ReferenceLine x={0} />
                      <Bar dataKey="points" radius={[8, 8, 8, 8]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card className="chart-card">
                <h2 className="section-title">M2b survival curve snapshot</h2>
                <p className="section-subtitle">Illustrative cancer-specific survival view aligned to M2b output concepts.</p>
                <div className="chart-wrap small">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={m2b.survivalCurve}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Line type="monotone" dataKey="survival" strokeWidth={3} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === "locked" && (
          <div className="stack-gap">
            <div className="metrics-grid">
              {modelSummary.map((m) => (
                <Card key={m.model}>
                  <div className="mini-label">{m.model}</div>
                  <div className="stat-value">{formatNumber(m.score)}</div>
                  <div className="mini-copy">{m.algorithm} · {m.metric}</div>
                  {m.note ? <div className="small-note">{m.note}</div> : null}
                </Card>
              ))}
            </div>

            <div className="two-panel">
              <Card className="chart-card">
                <h2 className="section-title">Final locked scores across the 6-task pipeline</h2>
                <p className="section-subtitle">Classification winners were selected by training-only CV; test scores are final locked estimates.</p>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={modelSummary} layout="vertical" margin={{ left: 20, right: 20, top: 10, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" domain={[0, 0.9]} />
                      <YAxis type="category" dataKey="model" width={180} />
                      <Tooltip formatter={(value: number, _name, entry: { payload?: ModelSummaryRow }) => [formatNumber(value), entry?.payload?.metric ?? "score"]} />
                      <Bar dataKey="score" radius={[8, 8, 8, 8]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card>
                <h2 className="section-title">What changed in the final pipeline</h2>
                <p className="section-subtitle">The benchmark cards and plots are driven by exported JSON, not embedded constants.</p>
                <div className="note-stack">
                  <div className="feature-note">
                    <div className="feature-title"><Gauge size={16} /> Locked binary diagnostics</div>
                    <p>M1b AUC, PR-AUC, Brier and ECE are loaded from exported metric summaries.</p>
                  </div>
                  <div className="feature-note">
                    <div className="feature-title"><Layers3 size={16} /> Threshold reporting</div>
                    <p>Threshold trade-offs update directly from exported JSON rather than being hardcoded in the frontend.</p>
                  </div>
                  <div className="feature-note">
                    <div className="feature-title"><Microscope size={16} /> Survival sensitivity</div>
                    <p>M2b risk groups and competing-risks summaries are loaded from exported support and sensitivity artefacts.</p>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === "thresholds" && (
          <div className="stack-gap">
            <div className="two-panel">
              <Card>
                <h2 className="section-title">M1b locked calibration and threshold view</h2>
                <p className="section-subtitle">Metrics reflect the exported notebook 04c JSON artefacts.</p>

                <div className="metrics-grid compact">
                  <Card><div className="mini-label">AUC-ROC</div><div className="mini-value">{binaryMetrics ? formatNumber(binaryMetrics.auc) : "—"}</div></Card>
                  <Card><div className="mini-label">PR-AUC</div><div className="mini-value">{binaryMetrics ? formatNumber(binaryMetrics.prAuc) : "—"}</div></Card>
                  <Card><div className="mini-label">Winner</div><div className="mini-value">{binaryMetrics?.algorithm || m1bModel?.algorithm || "—"}</div></Card>
                  <Card><div className="mini-label">Brier</div><div className="mini-value">{binaryMetrics ? formatNumber(binaryMetrics.brier) : "—"}</div></Card>
                  <Card><div className="mini-label">ECE</div><div className="mini-value">{binaryMetrics ? formatNumber(binaryMetrics.ece) : "—"}</div></Card>
                  <Card><div className="mini-label">Calibration slope</div><div className="mini-value">{binaryMetrics ? formatNumber(binaryMetrics.calibrationSlope) : "—"}</div></Card>
                </div>

                <Alert title="Calibration caution" tone="warning" icon={<AlertTriangle size={16} />}>
                  {binaryMetrics
                    ? `The locked test set shows calibration intercept ${formatNumber(binaryMetrics.calibrationIntercept)} and slope ${formatNumber(binaryMetrics.calibrationSlope)}. This is useful for product framing: the model stratifies risk, but probabilities still need cautious interpretation.`
                    : "Calibration details will appear once the binary metrics JSON is loaded."}
                </Alert>

                <div className="inline-controls">
                  <label className="field narrow">
                    <span>Operational threshold</span>
                    <select value={selectedThreshold} onChange={(e) => setSelectedThreshold(e.target.value)}>
                      {thresholdRows.map((t) => (
                        <option key={t.threshold} value={t.threshold.toFixed(2)}>{t.threshold.toFixed(2)}</option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="metrics-grid compact">
                  <Card><div className="mini-label">Sensitivity</div><div className="mini-value">{thresholdRow ? formatPct(thresholdRow.sensitivity) : "—"}</div></Card>
                  <Card><div className="mini-label">Specificity</div><div className="mini-value">{thresholdRow ? formatPct(thresholdRow.specificity) : "—"}</div></Card>
                  <Card><div className="mini-label">PPV</div><div className="mini-value">{thresholdRow ? formatPct(thresholdRow.ppv) : "—"}</div></Card>
                  <Card><div className="mini-label">NPV</div><div className="mini-value">{thresholdRow ? formatPct(thresholdRow.npv) : "—"}</div></Card>
                </div>
              </Card>

              <Card className="chart-card">
                <h2 className="section-title">M1b threshold trade-offs</h2>
                <p className="section-subtitle">Locked test-set curves loaded from exported threshold JSON.</p>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={thresholdRows}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="threshold" type="number" domain={["dataMin", "dataMax"]} tickCount={5} />
                      <YAxis domain={[0, 1.05]} />
                      <Tooltip formatter={(value: number) => `${Math.round(value * 100)}%`} />
                      <Legend />
                      <Line type="monotone" dataKey="sensitivity" strokeWidth={2.5} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="specificity" strokeWidth={2.5} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="ppv" strokeWidth={2.5} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="npv" strokeWidth={2.5} dot={{ r: 4 }} />
                      {thresholdRow ? <ReferenceLine x={thresholdRow.threshold} strokeDasharray="4 4" /> : null}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <Card>
              <h2 className="section-title">M2b risk-group and competing-risks summary</h2>
              <p className="section-subtitle">Directly aligned to the exported M2b support and sensitivity outputs.</p>
              <div className="two-panel">
                <div className="chart-wrap medium">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data?.m2bRiskGroups ?? []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="group" />
                      <YAxis domain={[0, 0.6]} />
                      <Tooltip formatter={(value: number, name: string) => [name === "eventRate" ? `${Math.round(value * 100)}%` : value, name]} />
                      <Legend />
                      <Bar dataKey="eventRate" name="Cancer-specific event rate" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="summary-list">
                  {(data?.m2bRiskGroups ?? []).map((row) => (
                    <div key={row.group} className="summary-item vertical">
                      <strong>{row.group}</strong>
                      <span className="mini-copy">Event rate {formatPct(row.eventRate)} · median time {row.medianTime} months · n={row.n}</span>
                    </div>
                  ))}
                  <Alert title="Competing-risks sensitivity" tone="info" icon={<Info size={16} />}>
                    The M2b competing-risks panel is loaded from the exported summary JSON rather than embedded values. It remains a sensitivity analysis, not a replacement for the main cause-specific Cox model.
                  </Alert>
                </div>
              </div>

              <div className="chart-wrap medium">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data?.m2bCompetingRisks ?? []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="group" />
                    <YAxis domain={[0, 0.25]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                    <Tooltip formatter={(value: number) => `${Math.round(value * 100)}%`} />
                    <Legend />
                    <Bar dataKey="cancer60m" name="Cancer CIF at 60m" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="otherCause60m" name="Other-cause CIF at 60m" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        )}

        {activeTab === "stability" && (
          <div className="two-panel">
            <Card>
              <h2 className="section-title">Training-data sensitivity checks</h2>
              <p className="section-subtitle">Repeated outer validation and leave-one-cohort-out transport are descriptive stability checks, not replacements for external validation.</p>

              <div className="inline-controls">
                <label className="field narrow">
                  <span>Task</span>
                  <select value={selectedStabilityTask} onChange={(e) => setSelectedStabilityTask(e.target.value as StabilityTask)}>
                    <option value="M1a">M1a</option>
                    <option value="M1b">M1b</option>
                    <option value="M3">M3</option>
                    <option value="M4">M4</option>
                  </select>
                </label>
              </div>

              <div className="two-col">
                <div className="subcard">
                  <div className="mini-label">Repeated outer validation</div>
                  <div className="mini-value">{repeatedMeta ? `${formatNumber(repeatedMeta.mean)} ± ${formatNumber(repeatedMeta.sd)}` : "—"}</div>
                  <div className="mini-copy">Metric: {repeatedMeta?.metric || "—"}</div>
                </div>
                <div className="subcard">
                  <div className="mini-label">Best feature representation for this task</div>
                  <div className="mini-value">
                    {featureSetForTask.length > 0 ? featureSetForTask.reduce((best, row) => row.score > best.score ? row : best, featureSetForTask[0]).set : "—"}
                  </div>
                  <div className="mini-copy">Training-CV descriptive comparison only</div>
                </div>
              </div>

              <div className="chart-wrap medium">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={repeatedSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="repeat" />
                    <YAxis domain={["dataMin - 0.02", "dataMax + 0.02"]} />
                    <Tooltip formatter={(value: number) => value.toFixed(3)} />
                    <Line type="monotone" dataKey="score" strokeWidth={2.5} dot={{ r: 4 }} />
                    {repeatedMeta ? <ReferenceLine y={repeatedMeta.mean} strokeDasharray="4 4" /> : null}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <div className="side-stack">
              <Card className="chart-card">
                <h2 className="section-title">Leave-one-cohort-out transport sensitivity</h2>
                <p className="section-subtitle">Training-data-only descriptive transport check.</p>
                <div className="chart-wrap small">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={transportSeries}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="cohort" />
                      <YAxis domain={[0, 0.9]} />
                      <Tooltip formatter={(value: number, _name, entry: { payload?: CohortTransportRow }) => [formatNumber(value), entry?.payload?.metric ?? "score"]} />
                      <Bar dataKey="score" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card className="chart-card">
                <h2 className="section-title">Feature representation comparison</h2>
                <p className="section-subtitle">Descriptive training-CV comparison of sets A/B/C/D from exported notebook 07 JSON.</p>
                <div className="chart-wrap small">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={featureSetForTask}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="set" />
                      <YAxis domain={[0, 0.85]} />
                      <Tooltip formatter={(value: number, _name, entry: { payload?: FeatureSetRow }) => [formatNumber(value), `${entry?.payload?.nFeatures ?? 0} features`]} />
                      <Bar dataKey="score" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>
          </div>
        )}

        <div className="bottom-grid">
          <Card>
            <h2 className="section-title">Explain prediction using pipeline features</h2>
            <p className="section-subtitle">Readable clinician-facing explanation using engineered variables and gene programmes.</p>
            <div className="explain-stack">
              {m1b.ranked.map((item) => (
                <div key={item.feature} className="explain-card">
                  <div>
                    <div className="feature-name">{item.display}</div>
                    <div className="mini-copy">Raw value: {item.value}</div>
                    <div className="feature-rationale">{item.rationale}</div>
                  </div>
                  <Badge tone={item.points >= 0 ? "warning" : "success"}>
                    {item.points >= 0 ? "+" : ""}{item.points.toFixed(1)}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <h2 className="section-title">Product notes</h2>
            <p className="section-subtitle">Deployment framing aligned to the updated pipeline outputs.</p>

            <div className="two-col">
              <div className="feature-note">
                <div className="feature-title"><ClipboardList size={16} /> Intended use</div>
                <p>Use M1b_cancer_specific_survival for risk stratification and M2b_cancer_specific_cox for cancer-specific survival discussion in tumour board or oncology review.</p>
              </div>

              <div className="feature-note">
                <div className="feature-title"><FlaskConical size={16} /> Output style</div>
                <p>Locked model metadata, threshold trade-offs, calibration metrics, survival risk groups, transport sensitivity and competing-risks notes sit alongside the illustrative patient sandbox.</p>
              </div>
            </div>

            <Alert title="Safety framing" tone="warning" icon={<AlertTriangle size={16} />}>
              This should support clinician judgment, not automate treatment selection. M1b is for risk stratification; M2b is better framed as survival ranking and prognosis support rather than an exact survival clock. External validation is still needed.
            </Alert>

            <div className="feature-note">
              <div className="feature-title">Expected exported JSON files</div>
              <ul className="file-list">
                <li>/pipeline-data/30_model_summary.json</li>
                <li>/pipeline-data/28c_m1b_binary_metrics_summary.json</li>
                <li>/pipeline-data/28c_m1b_threshold_metrics.json</li>
                <li>/pipeline-data/28d_m2b_risk_group_summary.json</li>
                <li>/pipeline-data/54_m2b_competing_risk_summary.json</li>
                <li>/pipeline-data/29_feature_representation_scores.json (optional)</li>
                <li>/pipeline-data/31_cohort_transport_sensitivity.json (optional)</li>
                <li>/pipeline-data/32_repeated_outer_validation.json (optional)</li>
              </ul>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
