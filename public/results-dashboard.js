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

  boot();
})();
