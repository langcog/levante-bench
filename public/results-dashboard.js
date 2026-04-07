(function () {
  const modelsEl = document.getElementById("models");
  const childrenEl = document.getElementById("children");
  const metricEl = document.getElementById("metric");
  const tasksEl = document.getElementById("tasks");
  const languagesEl = document.getElementById("languages");
  const tableBody = document.querySelector("#results-table tbody");
  const metricColumnHeaderEl = document.getElementById("metric-column-header");
  const closestBinColumnHeaderEl = document.getElementById("closest-bin-column-header");
  const ageEqColumnHeaderEl = document.getElementById("age-eq-column-header");
  const gapColumnHeaderEl = document.getElementById("gap-column-header");
  const summaryStatsEl = document.getElementById("summary-stats");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const allModelsBtn = document.getElementById("all-models");
  const clearModelsBtn = document.getElementById("clear-models");
  const allChildrenBtn = document.getElementById("all-children");
  const clearChildrenBtn = document.getElementById("clear-children");
  const allTasksBtn = document.getElementById("all-tasks");
  const allLanguagesBtn = document.getElementById("all-languages");
  const tabModelsBtn = document.getElementById("tab-models");
  const tabChildrenBtn = document.getElementById("tab-children");
  const panelModels = document.getElementById("panel-models");
  const panelChildren = document.getElementById("panel-children");
  const refreshDataBtn = document.getElementById("refresh-data");
  const helpMenuButton = document.getElementById("help-menu-button");
  const helpMenuDropdown = document.getElementById("help-menu-dropdown");
  const helpMenuItems = Array.from(document.querySelectorAll(".help-menu-item"));
  const helpModal = document.getElementById("help-modal");
  const helpModalClose = document.getElementById("help-modal-close");
  const helpModalTitle = document.getElementById("help-modal-title");
  const helpModalContent = document.getElementById("help-modal-content");

  let chart = null;
  let accuracyModelRecords = [];
  let accuracyChildRecords = [];
  let klModelRecords = [];
  let ageEqModelRecords = [];
  let ageEqAccModelRecords = [];
  let ageEquivalencyIndex = new Map();
  let ageEquivalencyAccuracyIndex = new Map();
  const preferredTaskOrder = [
    "egma-math",
    "matrix-reasoning",
    "mental-rotation",
    "theory-of-mind",
    "trog",
    "vocab",
  ];
  const levantePalette = [
    "#ff6f00",
    "#00a0de",
    "#ff9d2d",
    "#00bafc",
    "#f59e0b",
    "#38bdf8",
    "#fb923c",
    "#22d3ee",
    "#f97316",
    "#0ea5e9",
  ];
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
          ctx.fillStyle = dataset.borderColor || "#f8fafc";
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
          It lets researchers compare model runs against aggregated child performance
          by age bin, with shared task and language filters.
        </p>
        <ul>
          <li><strong>Models tab:</strong> select one or more model families/sizes.</li>
          <li><strong>Children tab:</strong> select one or more child age bins.</li>
          <li><strong>Languages filter:</strong> applies to both models and children.</li>
          <li><strong>Tasks filter:</strong> applies to both models and children.</li>
        </ul>
        <p>
          Child lines are computed from <code>trials.csv</code> and exposed through
          <code>/api/human-age-accuracy</code>.
        </p>
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
        <p>Current benchmark runs include the following model families:</p>
        <h3>Aquila-VL</h3>
        <p><a href="https://huggingface.co/BAAI/Aquila-VL-2B-llava-qwen" target="_blank" rel="noopener noreferrer">BAAI/Aquila-VL-2B-llava-qwen</a></p>
        <p><a href="https://github.com/BAAI-DCAI/Aquila-VL" target="_blank" rel="noopener noreferrer">Aquila-VL project repository</a></p>
        <h3>SmolVLM2</h3>
        <p><a href="https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Instruct" target="_blank" rel="noopener noreferrer">HuggingFaceTB/SmolVLM2-256M-Instruct</a></p>
        <p><a href="https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct" target="_blank" rel="noopener noreferrer">HuggingFaceTB/SmolVLM2-500M-Instruct</a></p>
        <h3>Qwen3.5</h3>
        <p><a href="https://huggingface.co/Qwen" target="_blank" rel="noopener noreferrer">Qwen model family (Qwen3.5 variants)</a></p>
        <h3>Gemma</h3>
        <p><a href="https://huggingface.co/google/gemma-3-4b-it" target="_blank" rel="noopener noreferrer">google/gemma-3-4b-it</a></p>
        <p><a href="https://huggingface.co/google/gemma-4-E2B-it" target="_blank" rel="noopener noreferrer">google/gemma-4-E2B-it</a></p>
        <p><a href="https://huggingface.co/google/gemma-4-E4B-it" target="_blank" rel="noopener noreferrer">google/gemma-4-E4B-it</a></p>
        <h3>InternVL3.5</h3>
        <p><a href="https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF" target="_blank" rel="noopener noreferrer">OpenGVLab/InternVL3_5-1B-HF</a></p>
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
        <h3>Step 4: Compare with children</h3>
        <p>
          Human child accuracy lines are aggregated from Redivis trial data by
          age bin, task, and language, then loaded by the dashboard alongside model runs.
        </p>
      `,
    },
    "age-equivalency": {
      title: "Age Equivalency (KL-D)",
      html: `
        <p>
          <strong>Age Eq</strong> is a task-specific estimate derived from how closely
          a model's response distribution matches child response distributions across
          IRT ability bins.
        </p>
        <h3>How it's computed</h3>
        <ul>
          <li>Compute mean D_KL for each <code>ability_bin</code> within a task.</li>
          <li>Convert KL values to soft weights (lower KL = higher weight).</li>
          <li>Map each ability bin to child age stats from the same task.</li>
          <li>Report weighted expected age as <code>soft_age_eq_mean</code>.</li>
        </ul>
        <h3>How to read it safely</h3>
        <ul>
          <li>It is <strong>not</strong> a literal developmental age.</li>
          <li>Interpret within-task; do not over-compare raw values across tasks.</li>
          <li>Treat low-confidence matches as weak evidence.</li>
        </ul>
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
          <li>Build child age/language comparison data with <code>scripts/analysis/plot_human_accuracy_by_age_lines.py</code>.</li>
          <li>Audit parsing behavior with <code>scripts/analysis/check_parser_glitches.py</code>.</li>
          <li>Refresh this dashboard to pull latest bucket-backed model and child comparison data.</li>
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

  function currentMetric() {
    return metricEl ? metricEl.value : "accuracy";
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

  function currentModelRecords() {
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

  function currentChildRecords() {
    return isKlMetric() || isAgeEqMetric() || isAgeEqAccuracyMetric() ? [] : accuracyChildRecords;
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

  function parseChildRecords(payload) {
    const rows = (payload && payload.records) || [];
    const byAgeBinAndLanguage = new Map();
    rows.forEach((row) => {
      const ageBin = String(row.age_bin || "").trim();
      const taskId = String(row.task_id || "").trim();
      const language = String(row.language || "unknown").trim().toLowerCase() || "unknown";
      const accuracy = Number(row.accuracy);
      if (!ageBin || !taskId || !Number.isFinite(accuracy)) {
        return;
      }
      const key = `${ageBin}|${language}`;
      if (!byAgeBinAndLanguage.has(key)) {
        byAgeBinAndLanguage.set(key, { ageBin, language, taskMeans: {} });
      }
      byAgeBinAndLanguage.get(key).taskMeans[taskId] = accuracy;
    });

    return Array.from(byAgeBinAndLanguage.values())
      .sort((a, b) => {
        const ageCmp = a.ageBin.localeCompare(b.ageBin, undefined, { numeric: true });
        if (ageCmp !== 0) {
          return ageCmp;
        }
        return a.language.localeCompare(b.language);
      })
      .map(({ ageBin, language, taskMeans }) => ({
        label: `Children ${ageBin} (${language})`,
        kind: "child",
        child: ageBin,
        ageBin,
        model: `Children (${ageBin})`,
        language,
        taskMeans,
      }));
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
      children: selectedValues(childrenEl),
      tasks: selectedValues(tasksEl),
      languages: selectedValues(languagesEl),
    };
    const modelsSource = accuracyModelRecords
      .concat(klModelRecords)
      .concat(ageEqModelRecords)
      .concat(ageEqAccModelRecords);
    const childrenSource = accuracyChildRecords;
    const models = uniqueSorted(modelsSource.map((r) => r.model));
    const children = sortAgeBins(childrenSource.map((r) => r.child));
    const tasks = sortTasks(
      uniqueSorted(
        modelsSource
          .concat(childrenSource)
          .flatMap((r) => Object.keys(r.taskMeans)),
      ),
    );
    const languages = uniqueSorted(
      modelsSource.concat(childrenSource).map((r) => r.language || "unknown"),
    );

    modelsEl.innerHTML = models.map((v) => `<option value="${v}">${v}</option>`).join("");
    childrenEl.innerHTML = children.map((v) => `<option value="${v}">${v}</option>`).join("");
    tasksEl.innerHTML = tasks.map((v) => `<option value="${v}">${v}</option>`).join("");
    languagesEl.innerHTML = languages.map((v) => `<option value="${v}">${v}</option>`).join("");

    if (preserveSelection) {
      setSelectedFromSet(modelsEl, previousSelection.models);
      setSelectedFromSet(childrenEl, previousSelection.children);
      setSelectedFromSet(tasksEl, previousSelection.tasks);
      setSelectedFromSet(languagesEl, previousSelection.languages);
    } else {
      setAllSelected(modelsEl);
      setAllSelected(childrenEl);
      setAllSelected(tasksEl);
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
    const childSet = selectedValues(childrenEl);
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

    const filteredModels = currentModelRecords()
      .map(applyTaskFilter)
      .filter(
        (r) =>
          modelSet.has(r.model) &&
          langSet.has(r.language || "en") &&
          Object.keys(r.taskMeans).length > 0,
      );

    const filteredChildren = currentChildRecords()
      .map(applyTaskFilter)
      .filter(
        (r) =>
          childSet.has(r.child) &&
          langSet.has(r.language || "unknown") &&
          Object.keys(r.taskMeans).length > 0,
      );

    return filteredModels.concat(filteredChildren);
  }

  function renderTable(rows) {
    if (!rows.length) {
      tableBody.innerHTML =
        '<tr><td colspan="7">No rows for current filter selection.</td></tr>';
      return;
    }
    const html = rows
      .map((row) => {
        const taskCount = Object.keys(row.taskMeans).length;
        const avg = mean(Object.values(row.taskMeans));
        const selectedTasks = Object.keys(row.taskMeans);
        let closestBinText = "n/a";
        let ageEqText = "n/a";
        let gapText = "n/a";
        if ((isKlMetric() || isAgeEqMetric() || isAgeEqAccuracyMetric()) && row.kind === "model") {
          if (selectedTasks.length === 1) {
            const onlyTask = selectedTasks[0];
            closestBinText = (row.closestBins && row.closestBins[onlyTask]) || "n/a";
            const ageEq = row.ageEqByTask && row.ageEqByTask[onlyTask];
            if (isAgeEqAccuracyMetric()) {
              const meta = row.ageEqMetaByTask && row.ageEqMetaByTask[onlyTask];
              if (meta && Number.isFinite(meta.accuracy_gap)) {
                gapText = meta.accuracy_gap.toFixed(3);
              }
              if (
                meta &&
                (meta.age_eq_status === "below_youngest_bin" ||
                  meta.age_eq_status === "above_oldest_bin") &&
                Number.isFinite(meta.extrapolated_age_eq_accuracy)
              ) {
                ageEqText = `${ageEq.toFixed(2)}* (${meta.extrapolated_age_eq_accuracy.toFixed(2)})`;
              } else {
                ageEqText = Number.isFinite(ageEq) ? ageEq.toFixed(2) : "n/a";
              }
            } else {
              ageEqText = Number.isFinite(ageEq) ? ageEq.toFixed(2) : "n/a";
            }
          } else if (selectedTasks.length > 1) {
            closestBinText = "select 1 task";
            ageEqText = "select 1 task";
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
          <td>${row.kind === "child" ? `Children (${row.ageBin})` : row.model}</td>
          <td>${row.language}</td>
          <td>${taskCount}</td>
          <td>${closestBinText}</td>
          <td>${ageEqText}</td>
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
        label: row.label,
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

  function renderChart(rows) {
    const taskSet = new Set();
    rows.forEach((row) => {
      Object.keys(row.taskMeans).forEach((taskId) => taskSet.add(taskId));
    });
    const labels = sortTasks(Array.from(taskSet));
    const datasets = rows.map((row, idx) => {
      const color = levantePalette[idx % levantePalette.length];
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
        return { value, question };
      });
      return {
        label: row.label,
        data: points.map((p) => p.value),
        questionMarkMask: points.map((p) => p.question),
        borderColor: color,
        backgroundColor: `${color}55`,
        pointBackgroundColor: color,
        pointBorderColor: "#f8fafc",
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

    if (chart) {
      chart.destroy();
    }
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
              color: "#e2e8f0",
              boxWidth: 16,
              boxHeight: 2,
            },
          },
          tooltip: {
            mode: "nearest",
            intersect: false,
          },
        },
        interaction: {
          mode: "nearest",
          intersect: false,
        },
        scales: {
          x: {
            ticks: {
              color: "#cbd5e1",
            },
            grid: {
              color: "rgba(148, 163, 184, 0.15)",
            },
          },
          y: {
            min: 0,
            max: klMetric || ageEqMetric || ageEqAccMetric ? undefined : 1,
            title: {
              display: true,
              text: klMetric
                ? "D_KL (lower is better)"
                : ageEqMetric
                  ? "Age equivalency (years)"
                  : ageEqAccMetric
                    ? "Age equivalency from accuracy (years)"
                  : "Accuracy",
              color: "#cbd5e1",
            },
            ticks: {
              color: "#cbd5e1",
            },
            grid: {
              color: "rgba(148, 163, 184, 0.25)",
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
    if (ageEqColumnHeaderEl) {
      ageEqColumnHeaderEl.textContent =
        klMetric || ageEqMetric || ageEqAccMetric ? "Age Eq (years)" : "Age Eq (KL only)";
    }
    if (gapColumnHeaderEl) {
      gapColumnHeaderEl.textContent = ageEqAccMetric ? "Gap (lower better)" : "Gap";
    }
    const rows = filteredRecords();
    statusEl.textContent = klMetric
      ? `Showing ${rows.length} model series (D_KL)`
      : ageEqMetric
        ? `Showing ${rows.length} model series (Age Equivalency)`
        : ageEqAccMetric
          ? `Showing ${rows.length} model series (Age Eq from Accuracy)`
        : `Showing ${rows.length} series entries`;
    renderSummary(rows);
    renderTable(rows);
    renderChart(rows);
  }

  function setSelectedFromSet(selectEl, valueSet) {
    Array.from(selectEl.options).forEach((option) => {
      option.selected = valueSet.has(option.value);
    });
  }

  async function loadReportData({ preserveSelection = false } = {}) {
    try {
      if (refreshDataBtn) {
        refreshDataBtn.disabled = true;
        refreshDataBtn.textContent = "Refreshing...";
      }
      statusEl.textContent = "Loading report...";
      const [modelResponse, childResponse, klResponse, ageEqResponse, ageEqAccResponse] =
        await Promise.all([
        fetch(`/api/results-report?t=${Date.now()}`),
        fetch(`/api/human-age-accuracy?t=${Date.now()}`),
        fetch(`/api/kl-report?t=${Date.now()}`),
        fetch(`/api/model-age-equivalency?t=${Date.now()}`),
        fetch(`/api/model-age-equivalency-accuracy?t=${Date.now()}`),
      ]);
      if (!modelResponse.ok) {
        throw new Error(`Model report HTTP ${modelResponse.status}`);
      }
      if (!childResponse.ok) {
        throw new Error(`Children report HTTP ${childResponse.status}`);
      }
      if (!klResponse.ok) {
        throw new Error(`KL report HTTP ${klResponse.status}`);
      }
      if (!ageEqResponse.ok) {
        throw new Error(`Age-equivalency report HTTP ${ageEqResponse.status}`);
      }
      if (!ageEqAccResponse.ok) {
        throw new Error(`Age-equivalency-accuracy report HTTP ${ageEqAccResponse.status}`);
      }
      const payload = await modelResponse.json();
      const childPayload = await childResponse.json();
      const klPayload = await klResponse.json();
      const ageEqPayload = await ageEqResponse.json();
      const ageEqAccPayload = await ageEqAccResponse.json();
      accuracyModelRecords = parseModelRecords(payload.report || {});
      accuracyChildRecords = parseChildRecords(childPayload || {});
      ageEquivalencyIndex = parseAgeEquivalencyIndex(
        (ageEqPayload && ageEqPayload.records) || [],
      );
      klModelRecords = attachAgeEquivalency(parseKlModelRecords(klPayload || {}), ageEquivalencyIndex);
      ageEqModelRecords = parseAgeEquivalencyModelRecords(
        (ageEqPayload && ageEqPayload.records) || [],
      );
      ageEquivalencyAccuracyIndex = parseAgeEquivalencyAccuracyIndex(
        (ageEqAccPayload && ageEqAccPayload.records) || [],
      );
      ageEqAccModelRecords = parseAgeEquivalencyAccuracyModelRecords(
        (ageEqAccPayload && ageEqAccPayload.records) || [],
      );
      metaEl.textContent = `Model source: ${payload.source || "unknown"} | Models generated: ${
        (payload.report && payload.report.generated_at) || "n/a"
      } | Children source: ${childPayload.source || "unknown"} | Children rows: ${
        Array.isArray(childPayload.records) ? childPayload.records.length : 0
      } | KL source: ${klPayload.source || "unknown"} | KL rows: ${
        Array.isArray(klPayload.records) ? klPayload.records.length : 0
      } | AgeEq source: ${ageEqPayload.source || "unknown"} | AgeEq rows: ${
        Array.isArray(ageEqPayload.records) ? ageEqPayload.records.length : 0
      } | AgeEqAcc source: ${ageEqAccPayload.source || "unknown"} | AgeEqAcc rows: ${
        Array.isArray(ageEqAccPayload.records) ? ageEqAccPayload.records.length : 0
      } | Note: Age Eq is task-specific and approximate.`;
      renderSelectors({ preserveSelection });
      rerender();
    } catch (error) {
      statusEl.textContent = "Failed to load report data.";
      metaEl.textContent = String(error && error.message ? error.message : error);
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
    const isModels = tabName === "models";
    tabModelsBtn.classList.toggle("active", isModels);
    tabChildrenBtn.classList.toggle("active", !isModels);
    panelModels.classList.toggle("active", isModels);
    panelChildren.classList.toggle("active", !isModels);
  }

  async function boot() {
    await loadReportData({ preserveSelection: false });
  }

  [modelsEl, childrenEl, tasksEl, languagesEl].forEach((el) => {
    el.addEventListener("change", rerender);
  });
  if (metricEl) {
    metricEl.addEventListener("change", rerender);
  }
  allModelsBtn.addEventListener("click", () => {
    setAllSelected(modelsEl);
    rerender();
  });
  clearModelsBtn.addEventListener("click", () => {
    clearAllSelected(modelsEl);
    rerender();
  });
  allChildrenBtn.addEventListener("click", () => {
    setAllSelected(childrenEl);
    rerender();
  });
  clearChildrenBtn.addEventListener("click", () => {
    clearAllSelected(childrenEl);
    rerender();
  });
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
  tabChildrenBtn.addEventListener("click", () => activateSeriesTab("children"));
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
