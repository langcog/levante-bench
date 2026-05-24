(function () {
  const modelsEl = document.getElementById("models");
  const runsEl = document.getElementById("runs");
  const metricEl = document.getElementById("metric");
  const tasksEl = document.getElementById("tasks");
  const languagesEl = document.getElementById("languages");
  const compareToChanceEl = document.getElementById("compare-to-chance");
  const tableBody = document.querySelector("#results-table tbody");
  const metricColumnHeaderEl = document.getElementById("metric-column-header");
  const closestBinColumnHeaderEl = document.getElementById("closest-bin-column-header");
  const gapColumnHeaderEl = document.getElementById("gap-column-header");
  const summaryStatsEl = document.getElementById("summary-stats");
  const languageComparisonEl = document.getElementById("language-comparison");
  const languageComparisonContentEl = document.getElementById("language-comparison-content");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const allModelsBtn = document.getElementById("all-models");
  const clearModelsBtn = document.getElementById("clear-models");
  const allRunsBtn = document.getElementById("all-runs");
  const clearRunsBtn = document.getElementById("clear-runs");
  const allTasksBtn = document.getElementById("all-tasks");
  const allLanguagesBtn = document.getElementById("all-languages");
  const tabModelsBtn = document.getElementById("tab-models");
  const tabAdditionalModelsBtn = document.getElementById("tab-additional-models");
  const tabRunsBtn = document.getElementById("tab-runs");
  const panelModels = document.getElementById("panel-models");
  const panelRuns = document.getElementById("panel-runs");
  const refreshDataBtn = document.getElementById("refresh-data");
  const helpMenuButton = document.getElementById("help-menu-button");
  const helpMenuDropdown = document.getElementById("help-menu-dropdown");
  const helpMenuItems = Array.from(document.querySelectorAll(".help-menu-item"));
  const helpModal = document.getElementById("help-modal");
  const helpModalClose = document.getElementById("help-modal-close");
  const helpModalTitle = document.getElementById("help-modal-title");
  const helpModalContent = document.getElementById("help-modal-content");
  const ageEqFeatureEnabled = (() => {
    const query = new URLSearchParams(window.location.search);
    const raw =
      query.get("enable_age_eq") ||
      query.get("enableAgeEq") ||
      String(window.LEVANTE_ENABLE_AGE_EQ || "");
    const normalized = String(raw).trim().toLowerCase();
    return normalized === "1" || normalized === "true" || normalized === "on";
  })();

  let chart = null;
  let accuracyModelRecords = [];
  let additionalAccuracyModelRecords = [];
  let accuracyRunRecords = [];
  let klModelRecords = [];
  let ageEqModelRecords = [];
  let ageEqAccModelRecords = [];
  let ageEquivalencyIndex = new Map();
  let ageEquivalencyAccuracyIndex = new Map();
  let metaBase = null;
  let activeSeriesTab = "models";
  const preferredTaskOrder = [
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
  ];
  const levantePalette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#e6550d",
    "#31a354",
    "#756bb1",
    "#636363",
    "#fdae6b",
    "#9ecae1",
    "#74c476",
    "#bcbddc",
  ];
  const excludedLanguages = new Set(["tl", "tlh", "klingon"]);
  const chanceByTask = {
    "egma-math": 0.25,
    "matrix-reasoning": 0.25,
    "mental-rotation": 0.5,
    "theory-of-mind": 0.25,
    trog: 0.25,
    vocab: 0.25,
  };
  const MODEL_SIZE_UNDERSCORE_RE = /^(?<model>[A-Za-z0-9.-]+)_(?<size>[0-9]+(?:\.[0-9]+)?[A-Za-z]+)$/;
  const MODEL_SIZE_DASH_RE =
    /^(?<model>[A-Za-z0-9._-]+)-(?<size>(?:\d+(?:\.\d+)?[A-Za-z]+|[A-Za-z]+\d+[A-Za-z]*)(?:-(?:it|instruct))?)$/;
  const LANG_SUFFIX_RE = /^(?<base>.+)-(?<lang>[a-z]{2})$/;
  const questionMarkPointPlugin = {
    id: "questionMarkPointPlugin",
    afterDatasetsDraw(chart) {
      const { ctx } = chart;
      ctx.save();
      ctx.font = "bold 14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      chart.data.datasets.forEach((dataset, datasetIndex) => {
        const mask = dataset.questionMarkMask || [];
        if (!mask.length) {
          return;
        }
        const meta = chart.getDatasetMeta(datasetIndex);
        meta.data.forEach((pt, i) => {
          if (!mask[i]) {
            return;
          }
          ctx.fillStyle = dataset.borderColor || "#0f172a";
          ctx.fillText("?", pt.x, pt.y);
        });
      });
      ctx.restore();
    },
  };
  if (typeof Chart !== "undefined" && Chart.register) {
    Chart.register(questionMarkPointPlugin);
  }
  const helpContentByTopic = {
    project: {
      title: "Levante Bench",
      html: `
        <p>
          <strong>Levante-Bench</strong> is an open benchmark for evaluating vision-language
          models on child-centered cognitive tasks from the LEVANTE framework.
        </p>
        <h3>What it measures</h3>
        <ul>
          <li>Task accuracy across multiple cognitive domains.</li>
          <li>Model performance under multilingual prompt conditions.</li>
          <li>Comparability across model families and parameter sizes.</li>
        </ul>
        <h3>What this dashboard does</h3>
        <p>
          It lets researchers compare model runs across tasks with shared model,
          task, and language filters.
        </p>
        <ul>
          <li><strong>Models tab:</strong> select one or more model families/sizes.</li>
          <li><strong>Languages filter:</strong> applies to model series.</li>
          <li><strong>Tasks filter:</strong> applies to model series.</li>
        </ul>
      `,
    },
    dataset: {
      title: "Our Dataset (v1)",
      html: `
        <p>
          <strong>v1</strong> is the first stable benchmark snapshot used for reproducible
          model comparisons in this dashboard.
        </p>
        <h3>Task coverage</h3>
        <ul>
          <li>egma-math</li>
          <li>matrix-reasoning</li>
          <li>mental-rotation</li>
          <li>theory-of-mind</li>
          <li>trog</li>
          <li>vocab</li>
        </ul>
        <h3>Versioning</h3>
        <p>
          Results are stored with deterministic paths under
          <code>results/&lt;version&gt;/&lt;model-size[-lang]&gt;/</code> so runs can be tracked
          and compared across releases.
        </p>
      `,
    },
    models: {
      title: "Models",
      html: `
        <p>
          The dashboard selectors are loaded from published <code>v1</code> bucket results.
          Current paper runs include these model families:
        </p>
        <h3>Gemini</h3>
        <p>Gemini Pro baseline runs, including English, Spanish, and German prompt variants.</p>
        <h3>GPT-5.3</h3>
        <p>GPT-5.3 baseline runs.</p>
        <h3>Gemma 4</h3>
        <p><a href="https://huggingface.co/google" target="_blank" rel="noopener noreferrer">Google Gemma model family</a> (<code>gemma4-E2B-it</code>, <code>gemma4-E4B-it</code>, and <code>gemma4-26B-A4B-it</code>).</p>
        <h3>InternVL3.5</h3>
        <p><a href="https://huggingface.co/OpenGVLab" target="_blank" rel="noopener noreferrer">OpenGVLab InternVL3.5 family</a> (<code>1B</code>, <code>2B</code>, <code>4B</code>, <code>8B</code>, <code>14B</code>, and <code>38B</code>).</p>
        <h3>Qwen3.5</h3>
        <p><a href="https://huggingface.co/Qwen" target="_blank" rel="noopener noreferrer">Qwen model family</a> (<code>0.8B</code>, <code>2B</code>, <code>4B</code>, <code>9B</code>, and <code>27B</code>).</p>
        <h3>SmolVLM2</h3>
        <p><a href="https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Instruct" target="_blank" rel="noopener noreferrer">HuggingFaceTB/SmolVLM2-256M-Instruct</a></p>
        <p><a href="https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct" target="_blank" rel="noopener noreferrer">HuggingFaceTB/SmolVLM2-500M-Instruct</a></p>
        <p><a href="https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct" target="_blank" rel="noopener noreferrer">HuggingFaceTB/SmolVLM2-2.2B-Instruct</a></p>
        <h3>TinyLLaVA</h3>
        <p><a href="https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B" target="_blank" rel="noopener noreferrer">tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B</a></p>
      `,
    },
    benchmark: {
      title: "Benchmark Process",
      html: `
        <h3>Step 1: Run evaluation</h3>
        <p>
          Each model is evaluated on all benchmark tasks for a fixed data version
          (for example <code>v1</code>) and selected prompt language.
        </p>
        <h3>Step 2: Write task outputs</h3>
        <p>
          Per-task predictions are written to task CSV files, and per-model
          task accuracies are collected into <code>summary.csv</code>.
        </p>
        <h3>Step 3: Aggregate and publish</h3>
        <p>
          Published results are synced to the levante-bench bucket and this dashboard
          computes cross-model comparison JSON from bucket summaries on refresh.
        </p>
        <h3>Step 4: Compare model series</h3>
        <p>
          Model comparison series are loaded for selected tasks and languages,
          then plotted side-by-side in the dashboard.
        </p>
      `,
    },
    parser: {
      title: "Our Parser",
      html: `
        <p>
          LEVANTE-Bench uses a layered parser with provenance to convert raw model
          text into canonical benchmark answers.
        </p>
        <h3>How it works</h3>
        <ul>
          <li>Model-specific cleanup in <code>parse_response()</code> removes wrappers and formatting artifacts.</li>
          <li>Shared v2 answer parsing extracts labels/numbers and records parse provenance fields.</li>
          <li>Each trial logs <code>parse_method</code>, <code>parse_confidence</code>, and <code>parse_raw_candidate</code> for auditability.</li>
        </ul>
        <h3>Recent parser improvements</h3>
        <ul>
          <li>Punctuation-wrapped label extraction (for outputs like <code>; A:</code>).</li>
          <li>Broader explicit-phrase capture (for forms like <code>Final answer -&gt; (C)</code>, <code>choose option D</code>, and <code>Option B is correct</code>).</li>
          <li>Fallback logic for harder outputs in selected model adapters.</li>
        </ul>
        <h3>Validation tooling</h3>
        <p>
          Use <code>scripts/analysis/check_parser_glitches.py</code> to scan all
          result CSVs, surface parser-risk clusters, and generate fix suggestions.
        </p>
      `,
    },
    "getting-started": {
      title: "Getting Started",
      html: `
        <p>
          Quick path for researchers to run LEVANTE-Bench, compare models, and
          publish reproducible results.
        </p>
        <h3>1) Clone and set up</h3>
        <ul>
          <li>Clone the repository and create a Python virtual environment.</li>
          <li>Install dependencies and set required API keys in <code>.env</code> (for hosted models).</li>
          <li>Download benchmark assets with <code>scripts/download_levante_assets.py</code>.</li>
        </ul>
        <h3>2) Run evaluations</h3>
        <ul>
          <li>Run via CLI (<code>python -m levante_bench.cli run-eval ...</code>) or experiment configs.</li>
          <li>Use canonical output layout: <code>results/&lt;version&gt;/&lt;model-size[-lang]&gt;/</code>.</li>
          <li>Each run should include task CSVs, <code>summary.csv</code>, and <code>metadata.json</code>.</li>
        </ul>
        <h3>3) Analyze and review quality</h3>
        <ul>
          <li>Build comparison JSON with <code>scripts/analysis/build_model_comparison_report.py</code>.</li>
          <li>Audit parsing behavior with <code>scripts/analysis/check_parser_glitches.py</code>.</li>
          <li>Refresh this dashboard to pull latest bucket-backed model comparison data.</li>
        </ul>
        <h3>4) Add your own model or runs</h3>
        <ul>
          <li>Add a model config in <code>configs/models/</code> and adapter implementation in <code>src/levante_bench/models/</code>.</li>
          <li>Register the model and run a small smoke evaluation before full runs.</li>
          <li>Upload completed results under the canonical version/model folder in the bucket.</li>
        </ul>
      `,
    },
  };

  function selectedValues(selectEl) {
    return new Set(Array.from(selectEl.selectedOptions).map((o) => o.value));
  }

  function setAllSelected(selectEl) {
    Array.from(selectEl.options).forEach((option) => {
      option.selected = true;
    });
  }

  function clearAllSelected(selectEl) {
    Array.from(selectEl.options).forEach((option) => {
      option.selected = false;
    });
  }

  function uniqueSorted(values) {
    return Array.from(new Set(values)).sort((a, b) => String(a).localeCompare(String(b)));
  }

  function splitModelSizeLanguage(modelTag) {
    let m = String(modelTag || "").match(MODEL_SIZE_UNDERSCORE_RE);
    if (m && m.groups) {
      return { model: m.groups.model, size: m.groups.size, language: null };
    }
    m = String(modelTag || "").match(MODEL_SIZE_DASH_RE);
    if (m && m.groups) {
      return { model: m.groups.model, size: m.groups.size, language: null };
    }

    let base = String(modelTag || "");
    let language = null;
    const lm = base.match(LANG_SUFFIX_RE);
    if (lm && lm.groups) {
      base = lm.groups.base;
      language = lm.groups.lang;
    }

    m = base.match(MODEL_SIZE_UNDERSCORE_RE);
    if (m && m.groups) {
      return { model: m.groups.model, size: m.groups.size, language };
    }
    m = base.match(MODEL_SIZE_DASH_RE);
    if (m && m.groups) {
      return { model: m.groups.model, size: m.groups.size, language };
    }
    return { model: base, size: null, language };
  }

  function sortAgeBins(values) {
    const toStartAge = (value) => {
      const match = String(value).trim().match(/^(\d+)/);
      return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
    };
    return Array.from(new Set(values)).sort((a, b) => {
      const sa = toStartAge(a);
      const sb = toStartAge(b);
      if (sa !== sb) {
        return sa - sb;
      }
      return String(a).localeCompare(String(b), undefined, { numeric: true });
    });
  }

  function mean(values) {
    if (!values.length) {
      return NaN;
    }
    return values.reduce((acc, val) => acc + val, 0) / values.length;
  }

  function languageLabel(language) {
    const normalized = String(language || "en").trim().toLowerCase() || "en";
    const labels = {
      en: "English",
      de: "German",
      es: "Spanish",
    };
    return labels[normalized] || normalized.toUpperCase();
  }

  function isExcludedLanguage(language) {
    const normalized = String(language || "").trim().toLowerCase();
    if (!normalized) {
      return false;
    }
    return (
      excludedLanguages.has(normalized) ||
      normalized.startsWith("tlh") ||
      normalized.startsWith("klingon")
    );
  }

  function languageChartStyle(language) {
    const normalized = String(language || "en").trim().toLowerCase() || "en";
    const styles = {
      en: { borderDash: [], pointStyle: "circle" },
      de: { borderDash: [7, 4], pointStyle: "rectRot" },
      es: { borderDash: [2, 4], pointStyle: "triangle" },
    };
    return styles[normalized] || { borderDash: [5, 3, 1, 3], pointStyle: "rect" };
  }

  function seriesDisplayLabel(row) {
    const language = String((row && row.language) || "en").trim().toLowerCase() || "en";
    const model = String((row && row.model) || "unknown").trim() || "unknown";
    return language === "en" ? model : `${model} · ${languageLabel(language)}`;
  }

  function meanForRow(row) {
    const values = Object.values((row && row.taskMeans) || {}).filter((value) =>
      Number.isFinite(value),
    );
    return mean(values);
  }

  function groupRowsByModel(rows) {
    const grouped = new Map();
    (rows || []).forEach((row) => {
      if (!row || row.kind !== "model") {
        return;
      }
      const model = String(row.model || "unknown").trim() || "unknown";
      const language = String(row.language || "en").trim().toLowerCase() || "en";
      if (!grouped.has(model)) {
        grouped.set(model, new Map());
      }
      grouped.get(model).set(language, row);
    });
    return grouped;
  }

  function buildLanguageComparisonRows(rows) {
    const grouped = groupRowsByModel(rows);
    return Array.from(grouped.entries())
      .map(([model, languageRows]) => {
        const languageStats = Array.from(languageRows.entries())
          .map(([language, row]) => ({
            language,
            label: languageLabel(language),
            mean: meanForRow(row),
          }))
          .filter((entry) => Number.isFinite(entry.mean))
          .sort((a, b) => {
            if (a.language === "en") {
              return -1;
            }
            if (b.language === "en") {
              return 1;
            }
            return a.label.localeCompare(b.label);
          });
        const english = languageStats.find((entry) => entry.language === "en");
        const best = languageStats.slice().sort((a, b) => b.mean - a.mean)[0] || null;
        return {
          model,
          englishMean: english ? english.mean : NaN,
          languages: languageStats.map((entry) => ({
            ...entry,
            deltaFromEnglish: english ? entry.mean - english.mean : NaN,
          })),
          bestLanguage: best ? best.label : "n/a",
          bestMean: best ? best.mean : NaN,
        };
      })
      .filter((entry) => entry.languages.length > 1)
      .sort((a, b) => a.model.localeCompare(b.model, undefined, { numeric: true }));
  }

  function formatDelta(value) {
    if (!Number.isFinite(value)) {
      return "n/a";
    }
    if (Math.abs(value) < 0.0005) {
      return "0.000";
    }
    return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
  }

  function currentMetric() {
    return metricEl ? metricEl.value : "accuracy";
  }

  function configureMetricOptions() {
    if (!metricEl || ageEqFeatureEnabled) {
      return;
    }
    Array.from(metricEl.options).forEach((option) => {
      if (option.value === "age_eq" || option.value === "age_eq_acc") {
        option.remove();
      }
    });
    if (metricEl.value === "age_eq" || metricEl.value === "age_eq_acc") {
      metricEl.value = "accuracy";
    }
  }

  function isKlMetric() {
    return currentMetric() === "d_kl";
  }

  function isAgeEqMetric() {
    return currentMetric() === "age_eq";
  }

  function isAgeEqAccuracyMetric() {
    return currentMetric() === "age_eq_acc";
  }

  function isComparedToChanceEnabled() {
    return Boolean(compareToChanceEl && compareToChanceEl.checked && currentMetric() === "accuracy");
  }

  function currentModelRecords() {
    if (activeSeriesTab === "additional") {
      return additionalAccuracyModelRecords;
    }
    if (isKlMetric()) {
      return klModelRecords;
    }
    if (isAgeEqMetric()) {
      return ageEqModelRecords;
    }
    if (isAgeEqAccuracyMetric()) {
      return ageEqAccModelRecords;
    }
    return accuracyModelRecords;
  }

  function currentRunRecords() {
    return accuracyRunRecords;
  }

  function currentSelectorSourceRecords() {
    if (activeSeriesTab === "additional") {
      return additionalAccuracyModelRecords;
    }
    if (activeSeriesTab === "runs") {
      return accuracyModelRecords.concat(klModelRecords).concat(ageEqModelRecords).concat(ageEqAccModelRecords);
    }
    return accuracyModelRecords.concat(klModelRecords).concat(ageEqModelRecords).concat(ageEqAccModelRecords);
  }

  function sortTasks(taskIds) {
    return taskIds.sort((a, b) => {
      const ia = preferredTaskOrder.indexOf(a);
      const ib = preferredTaskOrder.indexOf(b);
      if (ia >= 0 && ib >= 0) {
        return ia - ib;
      }
      if (ia >= 0) {
        return -1;
      }
      if (ib >= 0) {
        return 1;
      }
      return a.localeCompare(b);
    });
  }

  function parseModelRecords(report) {
    const byModel = report.by_model || {};
    const out = [];
    Object.values(byModel).forEach((entry) => {
      const taskStats = entry.task_stats || {};
      const taskMeans = {};
      Object.entries(taskStats).forEach(([taskId, stats]) => {
        if (typeof stats.mean === "number") {
          taskMeans[taskId] = stats.mean;
        }
      });
      const baseModelName =
        entry.size && String(entry.size).trim()
          ? `${entry.model || "unknown"}-${entry.size}`
          : entry.model || "unknown";
      const language = entry.language || "en";
      out.push({
        label:
          entry.canonical_model_tag ||
          (language === "en" ? baseModelName : `${baseModelName}-${language}`),
        kind: "model",
        model: baseModelName,
        language,
        taskMeans,
      });
    });
    return out;
  }

  function modelLanguageKey(row) {
    const model = String((row && row.model) || "").trim();
    const language = String((row && row.language) || "en").trim().toLowerCase() || "en";
    return `${model}|${language}`;
  }

  function parseRunRecords(report) {
    const rows = (report && report.runs) || [];
    const out = [];
    rows.forEach((row) => {
      const runId = String(row.run_id || "").trim();
      const taskMeans = row.task_metrics || {};
      const model = row.size ? `${row.model}-${row.size}` : String(row.model || "unknown");
      const language = row.language || "en";
      const runLabel = runId.split("/").slice(-1)[0] || "";
      // Only expose numbered runs in the Runs tab.
      if (!/^\d+$/.test(runLabel)) {
        return;
      }
      out.push({
        label: `${model} ${runLabel}`,
        kind: "run",
        model,
        language,
        runId,
        runLabel,
        taskMeans,
      });
    });
    return out.sort((a, b) => a.runLabel.localeCompare(b.runLabel, undefined, { numeric: true }));
  }

  function parseKlModelRecords(payload) {
    const report = (payload && payload.report) || {};
    const byModel = report.by_model || {};
    const out = [];
    Object.values(byModel).forEach((entry) => {
      const taskStats = entry.task_stats || {};
      const taskMeans = {};
      const closestBins = {};
      Object.entries(taskStats).forEach(([taskId, stats]) => {
        if (typeof stats.mean === "number") {
          taskMeans[taskId] = stats.mean;
        }
        if (stats && typeof stats.closest_ability_bin === "string") {
          closestBins[taskId] = stats.closest_ability_bin;
        }
      });
      const baseModelName =
        entry.size && String(entry.size).trim()
          ? `${entry.model || "unknown"}-${entry.size}`
          : entry.model || "unknown";
      const language = entry.language || "en";
      out.push({
        label:
          entry.canonical_model_tag ||
          (language === "en" ? baseModelName : `${baseModelName}-${language}`),
        kind: "model",
        model: baseModelName,
        language,
        taskMeans,
        closestBins,
        ageEqByTask: {},
      });
    });
    return out;
  }

  function parseAgeEquivalencyIndex(records) {
    const index = new Map();
    (records || []).forEach((row) => {
      const task = String(row.task || "").trim();
      const modelTag = String(row.model || "").trim();
      if (!task || !modelTag) {
        return;
      }
      const modelParts = splitModelSizeLanguage(modelTag);
      const baseModelName =
        modelParts.size && String(modelParts.size).trim()
          ? `${modelParts.model || "unknown"}-${modelParts.size}`
          : modelParts.model || "unknown";
      const language = modelParts.language || "en";
      const key = `${baseModelName}|${language}|${task}`;
      index.set(key, {
        soft_age_eq_mean: Number(row.soft_age_eq_mean),
        soft_age_eq_median: Number(row.soft_age_eq_median),
        closest_ability_bin: String(row.closest_ability_bin || "").trim(),
      });
    });
    return index;
  }

  function parseAgeEquivalencyAccuracyIndex(records) {
    const index = new Map();
    (records || []).forEach((row) => {
      const task = String(row.task || "").trim();
      const modelTag = String(row.model || "").trim();
      if (!task || !modelTag) {
        return;
      }
      const modelParts = splitModelSizeLanguage(modelTag);
      const baseModelName =
        modelParts.size && String(modelParts.size).trim()
          ? `${modelParts.model || "unknown"}-${modelParts.size}`
          : modelParts.model || "unknown";
      const language = modelParts.language || String(row.language || "en").toLowerCase() || "en";
      const key = `${baseModelName}|${language}|${task}`;
      index.set(key, {
        soft_age_eq_accuracy: Number(row.soft_age_eq_accuracy),
        extrapolated_age_eq_accuracy: Number(row.extrapolated_age_eq_accuracy),
        age_eq_status: row.age_eq_status ? String(row.age_eq_status) : null,
        nearest_age_bin: row.nearest_age_bin ? String(row.nearest_age_bin) : null,
        accuracy_gap: Number(row.accuracy_gap),
      });
    });
    return index;
  }

  function attachAgeEquivalency(rows, index) {
    return rows.map((row) => {
      if (row.kind !== "model") {
        return row;
      }
      const ageEqByTask = {};
      Object.keys(row.taskMeans || {}).forEach((taskId) => {
        const key = `${row.model}|${row.language || "en"}|${taskId}`;
        const rec = index.get(key);
        if (rec && Number.isFinite(rec.soft_age_eq_mean)) {
          ageEqByTask[taskId] = rec.soft_age_eq_mean;
        }
      });
      return { ...row, ageEqByTask };
    });
  }

  function parseAgeEquivalencyModelRecords(records) {
    const grouped = new Map();
    (records || []).forEach((row) => {
      const task = String(row.task || "").trim();
      const modelTag = String(row.model || "").trim();
      const ageEq = Number(row.soft_age_eq_mean);
      if (!task || !modelTag || !Number.isFinite(ageEq)) {
        return;
      }
      const modelParts = splitModelSizeLanguage(modelTag);
      const baseModelName =
        modelParts.size && String(modelParts.size).trim()
          ? `${modelParts.model || "unknown"}-${modelParts.size}`
          : modelParts.model || "unknown";
      const language = modelParts.language || "en";
      const key = `${baseModelName}|${language}`;
      if (!grouped.has(key)) {
        grouped.set(key, {
          label: language === "en" ? baseModelName : `${baseModelName}-${language}`,
          kind: "model",
          model: baseModelName,
          language,
          taskMeans: {},
          closestBins: {},
          ageEqByTask: {},
        });
      }
      const rec = grouped.get(key);
      rec.taskMeans[task] = ageEq;
      rec.ageEqByTask[task] = ageEq;
      if (row.closest_ability_bin) {
        rec.closestBins[task] = String(row.closest_ability_bin);
      }
    });
    return Array.from(grouped.values()).sort((a, b) => a.label.localeCompare(b.label));
  }

  function parseAgeEquivalencyAccuracyModelRecords(records) {
    const grouped = new Map();
    (records || []).forEach((row) => {
      const task = String(row.task || "").trim();
      const modelTag = String(row.model || "").trim();
      const ageEq = Number(row.soft_age_eq_accuracy);
      if (!task || !modelTag || !Number.isFinite(ageEq)) {
        return;
      }
      const modelParts = splitModelSizeLanguage(modelTag);
      const baseModelName =
        modelParts.size && String(modelParts.size).trim()
          ? `${modelParts.model || "unknown"}-${modelParts.size}`
          : modelParts.model || "unknown";
      const language = modelParts.language || String(row.language || "en").toLowerCase() || "en";
      const key = `${baseModelName}|${language}`;
      if (!grouped.has(key)) {
        grouped.set(key, {
          label: language === "en" ? baseModelName : `${baseModelName}-${language}`,
          kind: "model",
          model: baseModelName,
          language,
          taskMeans: {},
          closestBins: {},
          ageEqByTask: {},
        });
      }
      const rec = grouped.get(key);
      rec.taskMeans[task] = ageEq;
      rec.ageEqByTask[task] = ageEq;
      if (row.nearest_age_bin) {
        rec.closestBins[task] = String(row.nearest_age_bin);
      }
      if (!rec.ageEqMetaByTask) {
        rec.ageEqMetaByTask = {};
      }
      rec.ageEqMetaByTask[task] = {
        age_eq_status: row.age_eq_status ? String(row.age_eq_status) : null,
        accuracy_gap: Number(row.accuracy_gap),
        extrapolated_age_eq_accuracy: Number(row.extrapolated_age_eq_accuracy),
      };
    });
    return Array.from(grouped.values()).sort((a, b) => a.label.localeCompare(b.label));
  }

  function renderSelectors({ preserveSelection = false } = {}) {
    const previousSelection = {
      models: selectedValues(modelsEl),
      runs: selectedValues(runsEl),
      tasks: selectedValues(tasksEl),
      languages: selectedValues(languagesEl),
    };
    const modelsSource = currentSelectorSourceRecords();
    const runsSource = accuracyRunRecords;
    const models = uniqueSorted(modelsSource.map((r) => r.model));
    const selectedModelValues = preserveSelection ? previousSelection.models : new Set(models);
    const selectedModelList = Array.from(selectedModelValues);
    const runOptions =
      activeSeriesTab === "runs" && selectedModelList.length === 1
        ? uniqueSorted(
            runsSource
              .filter((r) => r.model === selectedModelList[0])
              .map((r) => r.runLabel),
          )
        : [];
    const tasks = sortTasks(
      uniqueSorted(
        modelsSource
          .concat(runsSource)
          .flatMap((r) => Object.keys(r.taskMeans)),
      ),
    );
    const languages = uniqueSorted(
      modelsSource.concat(runsSource).map((r) => r.language || "unknown"),
    );

    modelsEl.innerHTML = models.map((v) => `<option value="${v}">${v}</option>`).join("");
    if (activeSeriesTab === "runs" && selectedModelList.length === 1) {
      runsEl.innerHTML = runOptions.map((v) => `<option value="${v}">${v}</option>`).join("");
    } else {
      runsEl.innerHTML =
        '<option value="" disabled>Select exactly one model to view runs</option>';
    }
    tasksEl.innerHTML = tasks.map((v) => `<option value="${v}">${v}</option>`).join("");
    languagesEl.innerHTML = languages.map((v) => `<option value="${v}">${v}</option>`).join("");

    if (preserveSelection) {
      setSelectedFromSet(modelsEl, previousSelection.models);
      setSelectedFromSet(runsEl, previousSelection.runs);
      setSelectedFromSet(tasksEl, previousSelection.tasks);
      setSelectedFromSet(languagesEl, previousSelection.languages);
      // Guard against stale selections after source/schema changes.
      if (!modelsEl.selectedOptions.length) {
        setAllSelected(modelsEl);
      }
      if (!tasksEl.selectedOptions.length) {
        setAllSelected(tasksEl);
      }
      if (!languagesEl.selectedOptions.length) {
        setAllSelected(languagesEl);
      }
      if (activeSeriesTab === "runs" && runsEl && runOptions.length && !runsEl.selectedOptions.length) {
        setAllSelected(runsEl);
      }
    } else {
      setAllSelected(modelsEl);
      setAllSelected(tasksEl);
      if (activeSeriesTab === "runs" && runOptions.length) {
        setAllSelected(runsEl);
      }
      const hasEnglish = Array.from(languagesEl.options).some((option) => option.value === "en");
      if (hasEnglish) {
        clearAllSelected(languagesEl);
        setSelectedFromSet(languagesEl, new Set(["en"]));
      } else {
        setAllSelected(languagesEl);
      }
    }
  }

  function filteredRecords() {
    const modelSet = selectedValues(modelsEl);
    const runSet = selectedValues(runsEl);
    const taskSet = selectedValues(tasksEl);
    const langSet = selectedValues(languagesEl);
    const applyTaskFilter = (r) => {
      const filteredTaskMeans = {};
      const filteredClosestBins = {};
      const filteredAgeEqByTask = {};
      const filteredAgeEqMetaByTask = {};
      Object.entries(r.taskMeans).forEach(([taskId, value]) => {
        if (taskSet.has(taskId)) {
          filteredTaskMeans[taskId] = value;
          if (r.closestBins && r.closestBins[taskId]) {
            filteredClosestBins[taskId] = r.closestBins[taskId];
          }
          if (r.ageEqByTask && Number.isFinite(r.ageEqByTask[taskId])) {
            filteredAgeEqByTask[taskId] = r.ageEqByTask[taskId];
          }
          if (r.ageEqMetaByTask && r.ageEqMetaByTask[taskId]) {
            filteredAgeEqMetaByTask[taskId] = r.ageEqMetaByTask[taskId];
          }
        }
      });
      return {
        ...r,
        taskMeans: filteredTaskMeans,
        closestBins: filteredClosestBins,
        ageEqByTask: filteredAgeEqByTask,
        ageEqMetaByTask: filteredAgeEqMetaByTask,
      };
    };

    if (activeSeriesTab === "runs") {
      return currentRunRecords()
        .map(applyTaskFilter)
        .filter(
          (r) =>
            modelSet.has(r.model) &&
            runSet.has(r.runLabel) &&
            langSet.has(r.language || "en") &&
            Object.keys(r.taskMeans).length > 0,
        );
    }

    return currentModelRecords()
      .map(applyTaskFilter)
      .filter(
        (r) =>
          modelSet.has(r.model) &&
          langSet.has(r.language || "en") &&
          Object.keys(r.taskMeans).length > 0,
      );
  }

  function renderTable(rows) {
    if (!rows.length) {
      tableBody.innerHTML =
        '<tr><td colspan="6">No rows for current filter selection.</td></tr>';
      return;
    }
    const html = rows
      .map((row) => {
        const taskCount = Object.keys(row.taskMeans).length;
        const avg = mean(Object.values(row.taskMeans));
        const selectedTasks = Object.keys(row.taskMeans);
        let closestBinText = "n/a";
        let gapText = "n/a";
        if ((isKlMetric() || isAgeEqMetric() || isAgeEqAccuracyMetric()) && row.kind === "model") {
          if (selectedTasks.length === 1) {
            const onlyTask = selectedTasks[0];
            closestBinText = (row.closestBins && row.closestBins[onlyTask]) || "n/a";
            if (isAgeEqAccuracyMetric()) {
              const meta = row.ageEqMetaByTask && row.ageEqMetaByTask[onlyTask];
              if (meta && Number.isFinite(meta.accuracy_gap)) {
                gapText = meta.accuracy_gap.toFixed(3);
              }
            }
          } else if (selectedTasks.length > 1) {
            closestBinText = "select 1 task";
            if (isAgeEqAccuracyMetric()) {
              const gaps = selectedTasks
                .map((taskId) =>
                  row.ageEqMetaByTask && row.ageEqMetaByTask[taskId]
                    ? Number(row.ageEqMetaByTask[taskId].accuracy_gap)
                    : NaN,
                )
                .filter((v) => Number.isFinite(v));
              gapText = gaps.length ? mean(gaps).toFixed(3) : "n/a";
            }
          }
        }
        return `<tr>
          <td>${row.kind === "run" ? `${seriesDisplayLabel(row)} / ${row.runLabel}` : seriesDisplayLabel(row)}</td>
          <td>${languageLabel(row.language)}</td>
          <td>${taskCount}</td>
          <td>${closestBinText}</td>
          <td>${gapText}</td>
          <td>${Number.isNaN(avg) ? "n/a" : avg.toFixed(4)}</td>
        </tr>`;
      })
      .join("");
    tableBody.innerHTML = html;
  }

  function renderSummary(rows) {
    if (!rows.length) {
      summaryStatsEl.textContent = "No summary available for current selection.";
      return;
    }
    const means = rows
      .map((row) => mean(Object.values(row.taskMeans)))
      .filter((v) => !Number.isNaN(v));
    const overallMean = mean(means);
    const best = rows
      .map((row) => ({
        label: seriesDisplayLabel(row),
        score: mean(Object.values(row.taskMeans)),
      }))
      .sort((a, b) => (isKlMetric() ? a.score - b.score : b.score - a.score))[0];
    const metricName = isKlMetric()
      ? "D_KL"
      : isAgeEqMetric() || isAgeEqAccuracyMetric()
        ? "age equivalency"
        : "accuracy";
    const bestLabel = isKlMetric()
      ? "Best (lowest)"
      : isAgeEqMetric() || isAgeEqAccuracyMetric()
        ? "Highest mean age eq"
        : "Best mean";
    summaryStatsEl.textContent =
      `Series shown: ${rows.length} | ${bestLabel}: ${best.label} (${best.score.toFixed(4)}) | Overall mean ${metricName}: ${overallMean.toFixed(4)}`;
  }

  function renderLanguageComparison(rows) {
    if (!languageComparisonEl || !languageComparisonContentEl) {
      return;
    }
    const shouldShow =
      (activeSeriesTab === "models" || activeSeriesTab === "additional") &&
      currentMetric() === "accuracy";
    languageComparisonEl.classList.toggle("hidden", !shouldShow);
    if (!shouldShow) {
      languageComparisonContentEl.innerHTML = "";
      return;
    }

    const comparisons = buildLanguageComparisonRows(rows);
    if (!comparisons.length) {
      languageComparisonContentEl.innerHTML =
        '<p class="comparison-empty">Select at least one model with multiple language runs to compare non-English performance against English.</p>';
      return;
    }

    const languageSet = new Set();
    comparisons.forEach((comparison) => {
      comparison.languages.forEach((entry) => languageSet.add(entry.language));
    });
    const languages = Array.from(languageSet).sort((a, b) => {
      if (a === "en") {
        return -1;
      }
      if (b === "en") {
        return 1;
      }
      return languageLabel(a).localeCompare(languageLabel(b));
    });

    const headerCells = languages
      .map((language) => `<th>${languageLabel(language)}</th>`)
      .join("");
    const deltaHeaderCells = languages
      .filter((language) => language !== "en")
      .map((language) => `<th>${languageLabel(language)} vs English</th>`)
      .join("");
    const rowsHtml = comparisons
      .map((comparison) => {
        const byLanguage = new Map(
          comparison.languages.map((entry) => [entry.language, entry]),
        );
        const meanCells = languages
          .map((language) => {
            const entry = byLanguage.get(language);
            return `<td>${entry ? entry.mean.toFixed(4) : "n/a"}</td>`;
          })
          .join("");
        const deltaCells = languages
          .filter((language) => language !== "en")
          .map((language) => {
            const entry = byLanguage.get(language);
            const delta = entry ? entry.deltaFromEnglish : NaN;
            const className = Number.isFinite(delta)
              ? delta > 0.0005
                ? "delta-positive"
                : delta < -0.0005
                  ? "delta-negative"
                  : "delta-neutral"
              : "delta-missing";
            return `<td class="${className}">${formatDelta(delta)}</td>`;
          })
          .join("");
        const bestText = Number.isFinite(comparison.bestMean)
          ? `${comparison.bestLanguage} (${comparison.bestMean.toFixed(4)})`
          : "n/a";
        return `<tr>
          <td>${comparison.model}</td>
          ${meanCells}
          ${deltaCells}
          <td>${bestText}</td>
        </tr>`;
      })
      .join("");

    languageComparisonContentEl.innerHTML = `
      <div class="comparison-table-scroll">
        <table class="language-comparison-table">
          <thead>
            <tr>
              <th>Model</th>
              ${headerCells}
              ${deltaHeaderCells}
              <th>Best Language</th>
            </tr>
          </thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      </div>
    `;
  }

  function renderChart(rows) {
    const taskSet = new Set();
    rows.forEach((row) => {
      Object.keys(row.taskMeans).forEach((taskId) => taskSet.add(taskId));
    });
    const labels = sortTasks(Array.from(taskSet));
    const modelColors = new Map();
    uniqueSorted(rows.map((row) => row.model)).forEach((model, idx) => {
      modelColors.set(model, levantePalette[idx % levantePalette.length]);
    });
    const datasets = rows.map((row) => {
      const color = modelColors.get(row.model) || levantePalette[0];
      const style = languageChartStyle(row.language);
      const points = labels.map((taskId) => {
        if (!Object.prototype.hasOwnProperty.call(row.taskMeans, taskId)) {
          return { value: null, question: false };
        }
        let value = row.taskMeans[taskId];
        let question = false;
        if (isAgeEqAccuracyMetric() && row.ageEqMetaByTask && row.ageEqMetaByTask[taskId]) {
          const meta = row.ageEqMetaByTask[taskId];
          if (
            meta &&
            meta.age_eq_status === "below_youngest_bin" &&
            Number.isFinite(meta.accuracy_gap) &&
            Number.isFinite(value)
          ) {
            value = value - meta.accuracy_gap;
            question = true;
          }
        }
        if (compareToChance && Number.isFinite(value)) {
          const chance = chanceByTask[taskId];
          if (Number.isFinite(chance)) {
            value = value - chance;
          }
        }
        return { value, question };
      });
      return {
        label: seriesDisplayLabel(row),
        data: points.map((p) => p.value),
        questionMarkMask: points.map((p) => p.question),
        borderColor: color,
        backgroundColor: `${color}55`,
        pointBackgroundColor: color,
        pointBorderColor: "#ffffff",
        pointStyle: style.pointStyle,
        borderDash: style.borderDash,
        pointRadius: (ctx) => (ctx.dataset.questionMarkMask?.[ctx.dataIndex] ? 0 : 4),
        pointHoverRadius: (ctx) => (ctx.dataset.questionMarkMask?.[ctx.dataIndex] ? 0 : 5),
        borderWidth: 2.4,
        tension: 0.22,
        spanGaps: true,
      };
    });
    const ctx = document.getElementById("results-chart");
    const klMetric = isKlMetric();
    const ageEqMetric = isAgeEqMetric();
    const ageEqAccMetric = isAgeEqAccuracyMetric();
    const compareToChance = isComparedToChanceEnabled();
    const chanceAdjustedValues = [];

    if (chart) {
      chart.destroy();
    }
    if (compareToChance) {
      labels.forEach((taskId) => {
        const chance = chanceByTask[taskId];
        if (!Number.isFinite(chance)) {
          return;
        }
        rows.forEach((row) => {
          if (Object.prototype.hasOwnProperty.call(row.taskMeans, taskId)) {
            const val = Number(row.taskMeans[taskId]);
            if (Number.isFinite(val)) {
              chanceAdjustedValues.push(val - chance);
            }
          }
        });
      });
    }
    const maxChanceDelta = compareToChance
      ? Math.max(
          0.05,
          ...chanceAdjustedValues.map((v) => Math.abs(v)),
        )
      : null;
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: {
              color: "#334155",
              boxWidth: 16,
              boxHeight: 8,
              usePointStyle: true,
            },
          },
          tooltip: {
            mode: "nearest",
            intersect: false,
            callbacks: {
              label(context) {
                const datasetLabel = context.dataset && context.dataset.label ? context.dataset.label : "Series";
                const taskId = context.label;
                const value = Number(context.parsed.y);
                if (compareToChance && Number.isFinite(value)) {
                  const chance = chanceByTask[taskId];
                  if (Number.isFinite(chance)) {
                    const rawAccuracy = value + chance;
                    return `${datasetLabel}: ${(rawAccuracy * 100).toFixed(1)}% (${(value * 100).toFixed(1)} pp vs chance ${(chance * 100).toFixed(0)}%)`;
                  }
                }
                return `${datasetLabel}: ${Number.isFinite(value) ? value.toFixed(4) : "n/a"}`;
              },
            },
          },
        },
        interaction: {
          mode: "nearest",
          intersect: false,
        },
        scales: {
          x: {
            ticks: {
              color: "#475569",
            },
            grid: {
              color: "rgba(148, 163, 184, 0.24)",
            },
          },
          y: {
            min:
              compareToChance && Number.isFinite(maxChanceDelta) ? -maxChanceDelta : 0,
            max:
              compareToChance && Number.isFinite(maxChanceDelta)
                ? maxChanceDelta
                : klMetric || ageEqMetric || ageEqAccMetric
                  ? undefined
                  : 1,
            title: {
              display: true,
              text: klMetric
                ? "D_KL (lower is better)"
                : ageEqMetric
                  ? "Age equivalency (years)"
                  : ageEqAccMetric
                    ? "Age equivalency from accuracy (years)"
                    : compareToChance
                      ? "Accuracy vs chance (0 = chance)"
                      : "Accuracy",
              color: "#334155",
            },
            ticks: {
              color: "#475569",
              callback(value) {
                if (!compareToChance) {
                  return value;
                }
                return `${(Number(value) * 100).toFixed(0)} pp`;
              },
            },
            grid: {
              color: "rgba(148, 163, 184, 0.28)",
            },
          },
        },
      },
    });
  }

  function rerender() {
    const klMetric = isKlMetric();
    const ageEqMetric = isAgeEqMetric();
    const ageEqAccMetric = isAgeEqAccuracyMetric();
    if (metricColumnHeaderEl) {
      metricColumnHeaderEl.textContent = klMetric
        ? "Mean D_KL"
        : ageEqMetric
          ? "Mean Age Eq"
          : ageEqAccMetric
            ? "Mean Age Eq (Acc)"
          : "Mean Accuracy";
    }
    if (closestBinColumnHeaderEl) {
      closestBinColumnHeaderEl.textContent = klMetric || ageEqMetric || ageEqAccMetric
        ? ageEqAccMetric
          ? "Nearest Age Bin"
          : "Closest IRT Bin"
        : "Closest IRT Bin (KL only)";
    }
    if (gapColumnHeaderEl) {
      gapColumnHeaderEl.textContent = ageEqAccMetric ? "Gap (lower better)" : "Gap";
    }
    if (compareToChanceEl) {
      compareToChanceEl.disabled = currentMetric() !== "accuracy";
      if (compareToChanceEl.disabled) {
        compareToChanceEl.checked = false;
      }
    }
    const rows = filteredRecords();
    statusEl.textContent =
      activeSeriesTab === "runs"
        ? `Showing ${rows.length} selected run series (accuracy)`
        : activeSeriesTab === "additional"
          ? `Showing ${rows.length} add'l model series (v1_additional_models)`
        : klMetric
          ? `Showing ${rows.length} model series (D_KL)`
          : ageEqMetric
            ? `Showing ${rows.length} model series (Age Equivalency)`
            : ageEqAccMetric
              ? `Showing ${rows.length} model series (Age Eq from Accuracy)`
              : `Showing ${rows.length} series entries`;
    renderLanguageComparison(rows);
    renderSummary(rows);
    renderTable(rows);
    renderChart(rows);
  }

  function setSelectedFromSet(selectEl, valueSet) {
    Array.from(selectEl.options).forEach((option) => {
      option.selected = valueSet.has(option.value);
    });
  }

  function updateMetaText() {
    if (!metaBase) {
      return;
    }
    const ageEqMeta = ageEqFeatureEnabled
      ? `AgeEq source: ${metaBase.ageEqSource} | AgeEq rows: ${metaBase.ageEqRows} | AgeEqAcc source: ${metaBase.ageEqAccSource} | AgeEqAcc rows: ${metaBase.ageEqAccRows}`
      : "AgeEq metrics: disabled (enable with ?enable_age_eq=1)";
    const summariesMeta = metaBase.resultsRoot ? `Summaries: ${metaBase.resultsRoot} | ` : "";
    const additionalMeta = metaBase.additionalResultsRoot
      ? `Add'l summaries: ${metaBase.additionalResultsRoot} | Add'l source: ${metaBase.additionalModelSource} | Add'l generated: ${metaBase.additionalModelsGenerated} | `
      : "";
    metaEl.textContent = `${summariesMeta}${additionalMeta}Model source: ${metaBase.modelSource} | Models generated: ${metaBase.modelsGenerated} | KL source: ${metaBase.klSource} | KL rows: ${metaBase.klRows} | ${ageEqMeta} | Note: Age Eq is task-specific and approximate.`;
  }

  async function loadReportData({ preserveSelection = false } = {}) {
    try {
      if (refreshDataBtn) {
        refreshDataBtn.disabled = true;
        refreshDataBtn.textContent = "Refreshing...";
      }
      statusEl.textContent = "Loading report...";
      const baseRequests = [
        fetch(`/api/results-report?t=${Date.now()}`),
        fetch(`/api/results-report?results_prefix=results/v1_additional_models&t=${Date.now()}`),
        fetch(`/api/kl-report?t=${Date.now()}`),
      ];
      const ageEqRequests = ageEqFeatureEnabled
        ? [
            fetch(`/api/model-age-equivalency?t=${Date.now()}`),
            fetch(`/api/model-age-equivalency-accuracy?t=${Date.now()}`),
          ]
        : [];
      const responses = await Promise.all(baseRequests.concat(ageEqRequests));
      const modelResponse = responses[0];
      const additionalModelResponse = responses[1];
      const klResponse = responses[2];
      const ageEqResponse = ageEqFeatureEnabled ? responses[3] : null;
      const ageEqAccResponse = ageEqFeatureEnabled ? responses[4] : null;
      if (!modelResponse.ok) {
        throw new Error(`Model report HTTP ${modelResponse.status}`);
      }
      const additionalUnavailable = !additionalModelResponse.ok;
      const klUnavailable = !klResponse.ok;
      if (ageEqFeatureEnabled && ageEqResponse && !ageEqResponse.ok) {
        throw new Error(`Age-equivalency report HTTP ${ageEqResponse.status}`);
      }
      if (ageEqFeatureEnabled && ageEqAccResponse && !ageEqAccResponse.ok) {
        throw new Error(`Age-equivalency-accuracy report HTTP ${ageEqAccResponse.status}`);
      }
      const payload = await modelResponse.json();
      const additionalPayload = additionalUnavailable
        ? { source: "unavailable", report: { by_model: {}, runs: [] } }
        : await additionalModelResponse.json();
      const klPayload = klUnavailable ? { source: "unavailable", records: [] } : await klResponse.json();
      const ageEqPayload = ageEqFeatureEnabled && ageEqResponse ? await ageEqResponse.json() : null;
      const ageEqAccPayload =
        ageEqFeatureEnabled && ageEqAccResponse ? await ageEqAccResponse.json() : null;
      accuracyModelRecords = parseModelRecords(payload.report || {}).filter(
        (row) => !isExcludedLanguage(row.language),
      );
      additionalAccuracyModelRecords = parseModelRecords(additionalPayload.report || {}).filter(
        (row) => !isExcludedLanguage(row.language),
      );
      accuracyRunRecords = parseRunRecords(payload.report || {}).filter(
        (row) => !isExcludedLanguage(row.language),
      );
      ageEquivalencyIndex = ageEqFeatureEnabled
        ? parseAgeEquivalencyIndex((ageEqPayload && ageEqPayload.records) || [])
        : new Map();
      klModelRecords = attachAgeEquivalency(
        parseKlModelRecords(klPayload || {}),
        ageEquivalencyIndex,
      ).filter((row) => !isExcludedLanguage(row.language));
      ageEqModelRecords = ageEqFeatureEnabled
        ? parseAgeEquivalencyModelRecords((ageEqPayload && ageEqPayload.records) || []).filter(
            (row) => !isExcludedLanguage(row.language),
          )
        : [];
      ageEquivalencyAccuracyIndex = ageEqFeatureEnabled
        ? parseAgeEquivalencyAccuracyIndex((ageEqAccPayload && ageEqAccPayload.records) || [])
        : new Map();
      ageEqAccModelRecords = ageEqFeatureEnabled
        ? parseAgeEquivalencyAccuracyModelRecords(
            (ageEqAccPayload && ageEqAccPayload.records) || [],
          ).filter((row) => !isExcludedLanguage(row.language))
        : [];

      // Keep all metric tabs aligned to the canonical v1 model set loaded from
      // /api/results-report.
      const allowedModelLanguage = new Set(accuracyModelRecords.map((r) => modelLanguageKey(r)));
      const filterToAllowedModels = (rows) =>
        (rows || []).filter((row) => allowedModelLanguage.has(modelLanguageKey(row)));
      klModelRecords = filterToAllowedModels(klModelRecords);
      ageEqModelRecords = filterToAllowedModels(ageEqModelRecords);
      ageEqAccModelRecords = filterToAllowedModels(ageEqAccModelRecords);
      metaBase = {
        resultsRoot: (payload.report && payload.report.results_root) || null,
        modelSource: payload.source || "unknown",
        modelsGenerated: (payload.report && payload.report.generated_at) || "n/a",
        additionalResultsRoot:
          (additionalPayload.report && additionalPayload.report.results_root) || null,
        additionalModelSource: additionalUnavailable
          ? "unavailable"
          : additionalPayload.source || "unknown",
        additionalModelsGenerated:
          (additionalPayload.report && additionalPayload.report.generated_at) || "n/a",
        klSource: klUnavailable ? "unavailable" : klPayload.source || "unknown",
        klRows: Array.isArray(klPayload.records) ? klPayload.records.length : 0,
        ageEqSource: ageEqFeatureEnabled
          ? (ageEqPayload && ageEqPayload.source) || "unknown"
          : "disabled",
        ageEqRows: ageEqFeatureEnabled
          ? Array.isArray(ageEqPayload && ageEqPayload.records)
            ? ageEqPayload.records.length
            : 0
          : 0,
        ageEqAccSource: ageEqFeatureEnabled
          ? (ageEqAccPayload && ageEqAccPayload.source) || "unknown"
          : "disabled",
        ageEqAccRows: ageEqFeatureEnabled
          ? Array.isArray(ageEqAccPayload && ageEqAccPayload.records)
            ? ageEqAccPayload.records.length
            : 0
          : 0,
      };
      updateMetaText();
      renderSelectors({ preserveSelection });
      rerender();
      if (klUnavailable || additionalUnavailable) {
        const notices = [];
        if (additionalUnavailable) {
          notices.push("Add'l models unavailable");
        }
        if (klUnavailable) {
          notices.push("KL data unavailable");
        }
        statusEl.textContent = `${notices.join("; ")}; showing available model data only.`;
      }
    } catch (error) {
      const message = String(error && error.message ? error.message : error);
      if (message.includes("Children report HTTP")) {
        // Defensive fallback for stale clients: child API failures should not
        // block model-only dashboard rendering.
        statusEl.textContent = "Children data unavailable; showing model data only.";
        metaEl.textContent = message;
        return;
      }
      statusEl.textContent = "Failed to load report data.";
      metaEl.textContent = message;
    } finally {
      if (refreshDataBtn) {
        refreshDataBtn.disabled = false;
        refreshDataBtn.textContent = "Refresh Data";
      }
    }
  }

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  async function loadParserIssuesReport() {
    const response = await fetch(`/api/parser-glitch-report?t=${Date.now()}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.text();
  }

  function openHelpModal(topicId) {
    const item = helpContentByTopic[topicId];
    if (!item && topicId !== "parser-issues") {
      return;
    }
    if (topicId === "parser-issues") {
      helpModalTitle.textContent = "Current Parser Issues";
      helpModalContent.innerHTML = "<p>Loading parser glitch report...</p>";
      helpModal.classList.remove("hidden");
      helpMenuDropdown.setAttribute("aria-hidden", "true");
      helpMenuButton.setAttribute("aria-expanded", "false");
      loadParserIssuesReport()
        .then((reportText) => {
          helpModalContent.innerHTML = `<pre>${escapeHtml(reportText)}</pre>`;
        })
        .catch((error) => {
          helpModalContent.innerHTML = `<p>Failed to load parser glitch report: ${escapeHtml(
            String(error && error.message ? error.message : error),
          )}</p>`;
        });
      return;
    }
    helpModalTitle.textContent = item.title;
    helpModalContent.innerHTML = item.html;
    helpModal.classList.remove("hidden");
    helpMenuDropdown.setAttribute("aria-hidden", "true");
    helpMenuButton.setAttribute("aria-expanded", "false");
  }

  function closeHelpModal() {
    helpModal.classList.add("hidden");
  }

  function activateSeriesTab(tabName) {
    activeSeriesTab =
      tabName === "runs" ? "runs" : tabName === "additional" ? "additional" : "models";
    const isModels = activeSeriesTab === "models";
    const isAdditional = activeSeriesTab === "additional";
    if ((activeSeriesTab === "runs" || isAdditional) && metricEl) {
      metricEl.value = "accuracy";
      metricEl.disabled = true;
    } else if (metricEl) {
      metricEl.disabled = false;
    }
    tabModelsBtn.classList.toggle("active", isModels);
    if (tabAdditionalModelsBtn) {
      tabAdditionalModelsBtn.classList.toggle("active", isAdditional);
    }
    if (tabRunsBtn) {
      tabRunsBtn.classList.toggle("active", activeSeriesTab === "runs");
    }
    panelModels.classList.toggle("active", isModels || isAdditional);
    if (panelRuns) {
      panelRuns.classList.toggle("active", activeSeriesTab === "runs");
    }
    renderSelectors({ preserveSelection: true });
    rerender();
  }

  async function boot() {
    configureMetricOptions();
    activateSeriesTab("models");
    await loadReportData({ preserveSelection: false });
  }

  if (modelsEl) {
    modelsEl.addEventListener("change", () => {
      renderSelectors({ preserveSelection: true });
      rerender();
    });
  }
  [runsEl, tasksEl, languagesEl].forEach((el) => {
    el.addEventListener("change", rerender);
  });
  if (metricEl) {
    metricEl.addEventListener("change", rerender);
  }
  if (compareToChanceEl) {
    compareToChanceEl.addEventListener("change", rerender);
  }
  allModelsBtn.addEventListener("click", () => {
    setAllSelected(modelsEl);
    rerender();
  });
  clearModelsBtn.addEventListener("click", () => {
    clearAllSelected(modelsEl);
    rerender();
  });
  if (allRunsBtn && runsEl) {
    allRunsBtn.addEventListener("click", () => {
      setAllSelected(runsEl);
      rerender();
    });
  }
  if (clearRunsBtn && runsEl) {
    clearRunsBtn.addEventListener("click", () => {
      clearAllSelected(runsEl);
      rerender();
    });
  }
  allTasksBtn.addEventListener("click", () => {
    setAllSelected(tasksEl);
    rerender();
  });
  allLanguagesBtn.addEventListener("click", () => {
    setAllSelected(languagesEl);
    rerender();
  });
  if (refreshDataBtn) {
    refreshDataBtn.addEventListener("click", async () => {
      await loadReportData({ preserveSelection: true });
    });
  }
  if (helpMenuButton) {
    helpMenuButton.addEventListener("click", () => {
      const isHidden = helpMenuDropdown.getAttribute("aria-hidden") !== "false";
      helpMenuDropdown.setAttribute("aria-hidden", isHidden ? "false" : "true");
      helpMenuButton.setAttribute("aria-expanded", isHidden ? "true" : "false");
    });
  }
  tabModelsBtn.addEventListener("click", () => activateSeriesTab("models"));
  if (tabAdditionalModelsBtn) {
    tabAdditionalModelsBtn.addEventListener("click", () => activateSeriesTab("additional"));
  }
  if (tabRunsBtn) {
    tabRunsBtn.addEventListener("click", () => activateSeriesTab("runs"));
  }
  helpMenuItems.forEach((btn) => {
    btn.addEventListener("click", () => {
      openHelpModal(btn.dataset.helpTopic || "");
    });
  });
  if (helpModalClose) {
    helpModalClose.addEventListener("click", closeHelpModal);
  }
  if (helpModal) {
    helpModal.addEventListener("click", (event) => {
      if (event.target === helpModal) {
        closeHelpModal();
      }
    });
  }
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeHelpModal();
      helpMenuDropdown.setAttribute("aria-hidden", "true");
      helpMenuButton.setAttribute("aria-expanded", "false");
    }
  });
  document.addEventListener("click", (event) => {
    if (!helpMenuButton || !helpMenuDropdown) {
      return;
    }
    if (
      helpMenuDropdown.getAttribute("aria-hidden") === "false" &&
      !helpMenuDropdown.contains(event.target) &&
      !helpMenuButton.contains(event.target)
    ) {
      helpMenuDropdown.setAttribute("aria-hidden", "true");
      helpMenuButton.setAttribute("aria-expanded", "false");
    }
  });

  boot();
})();
