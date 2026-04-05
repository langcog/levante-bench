(function () {
  const modelsEl = document.getElementById("models");
  const tasksEl = document.getElementById("tasks");
  const languagesEl = document.getElementById("languages");
  const tableBody = document.querySelector("#results-table tbody");
  const summaryStatsEl = document.getElementById("summary-stats");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const allModelsBtn = document.getElementById("all-models");
  const allTasksBtn = document.getElementById("all-tasks");
  const allLanguagesBtn = document.getElementById("all-languages");
  const helpMenuButton = document.getElementById("help-menu-button");
  const helpMenuDropdown = document.getElementById("help-menu-dropdown");
  const helpMenuItems = Array.from(document.querySelectorAll(".help-menu-item"));
  const helpModal = document.getElementById("help-modal");
  const helpModalClose = document.getElementById("help-modal-close");
  const helpModalTitle = document.getElementById("help-modal-title");
  const helpModalContent = document.getElementById("help-modal-content");

  let chart = null;
  let records = [];
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
          It lets researchers filter model/task/language combinations and compare
          task-level accuracy curves and summary statistics.
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

  function uniqueSorted(values) {
    return Array.from(new Set(values)).sort((a, b) => String(a).localeCompare(String(b)));
  }

  function mean(values) {
    if (!values.length) {
      return NaN;
    }
    return values.reduce((acc, val) => acc + val, 0) / values.length;
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

  function parseRecords(report) {
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
        model: baseModelName,
        language,
        taskMeans,
      });
    });
    return out;
  }

  function renderSelectors() {
    const models = uniqueSorted(records.map((r) => r.model));
    const tasks = sortTasks(uniqueSorted(records.flatMap((r) => Object.keys(r.taskMeans))));
    const languages = uniqueSorted(records.map((r) => r.language || "en"));

    modelsEl.innerHTML = models.map((v) => `<option value="${v}">${v}</option>`).join("");
    tasksEl.innerHTML = tasks.map((v) => `<option value="${v}">${v}</option>`).join("");
    languagesEl.innerHTML = languages.map((v) => `<option value="${v}">${v}</option>`).join("");

    setAllSelected(modelsEl);
    setAllSelected(tasksEl);
    setAllSelected(languagesEl);
  }

  function filteredRecords() {
    const modelSet = selectedValues(modelsEl);
    const taskSet = selectedValues(tasksEl);
    const langSet = selectedValues(languagesEl);

    return records
      .map((r) => {
        const filteredTaskMeans = {};
        Object.entries(r.taskMeans).forEach(([taskId, value]) => {
          if (taskSet.has(taskId)) {
            filteredTaskMeans[taskId] = value;
          }
        });
        return { ...r, taskMeans: filteredTaskMeans };
      })
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
        '<tr><td colspan="4">No rows for current filter selection.</td></tr>';
      return;
    }
    const html = rows
      .map((row) => {
        const taskCount = Object.keys(row.taskMeans).length;
        const avg = mean(Object.values(row.taskMeans));
        return `<tr>
          <td>${row.model}</td>
          <td>${row.language}</td>
          <td>${taskCount}</td>
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
      .sort((a, b) => b.score - a.score)[0];
    summaryStatsEl.textContent =
      `Models shown: ${rows.length} | Best mean: ${best.label} (${best.score.toFixed(4)}) | Overall mean: ${overallMean.toFixed(4)}`;
  }

  function renderChart(rows) {
    const taskSet = new Set();
    rows.forEach((row) => {
      Object.keys(row.taskMeans).forEach((taskId) => taskSet.add(taskId));
    });
    const labels = sortTasks(Array.from(taskSet));
    const datasets = rows.map((row, idx) => {
      const color = levantePalette[idx % levantePalette.length];
      return {
        label: row.label,
        data: labels.map((taskId) =>
          Object.prototype.hasOwnProperty.call(row.taskMeans, taskId)
            ? row.taskMeans[taskId]
            : null,
        ),
        borderColor: color,
        backgroundColor: `${color}55`,
        pointBackgroundColor: color,
        pointBorderColor: "#f8fafc",
        pointRadius: 4,
        pointHoverRadius: 5,
        borderWidth: 2.4,
        tension: 0.22,
        spanGaps: true,
      };
    });
    const ctx = document.getElementById("results-chart");

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
            max: 1,
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
    const rows = filteredRecords();
    statusEl.textContent = `Showing ${rows.length} model entries`;
    renderSummary(rows);
    renderTable(rows);
    renderChart(rows);
  }

  function openHelpModal(topicId) {
    const item = helpContentByTopic[topicId];
    if (!item) {
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

  async function boot() {
    try {
      const response = await fetch("/api/results-report");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      records = parseRecords(payload.report || {});
      metaEl.textContent = `Source: ${payload.source || "unknown"} | Generated: ${
        (payload.report && payload.report.generated_at) || "n/a"
      }`;
      renderSelectors();
      rerender();
    } catch (error) {
      statusEl.textContent = "Failed to load report data.";
      metaEl.textContent = String(error && error.message ? error.message : error);
    }
  }

  [modelsEl, tasksEl, languagesEl].forEach((el) => {
    el.addEventListener("change", rerender);
  });
  allModelsBtn.addEventListener("click", () => {
    setAllSelected(modelsEl);
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
  if (helpMenuButton) {
    helpMenuButton.addEventListener("click", () => {
      const isHidden = helpMenuDropdown.getAttribute("aria-hidden") !== "false";
      helpMenuDropdown.setAttribute("aria-hidden", isHidden ? "false" : "true");
      helpMenuButton.setAttribute("aria-expanded", isHidden ? "true" : "false");
    });
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
