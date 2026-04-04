(function () {
  const modelsEl = document.getElementById("models");
  const tasksEl = document.getElementById("tasks");
  const languagesEl = document.getElementById("languages");
  const tableBody = document.querySelector("#results-table tbody");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const allModelsBtn = document.getElementById("all-models");
  const allTasksBtn = document.getElementById("all-tasks");
  const allLanguagesBtn = document.getElementById("all-languages");

  let chart = null;
  let records = [];

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
      out.push({
        label: entry.canonical_model_tag || entry.model,
        model: entry.model || "unknown",
        language: entry.language || "en",
        taskMeans,
      });
    });
    return out;
  }

  function renderSelectors() {
    const models = uniqueSorted(records.map((r) => r.model));
    const tasks = uniqueSorted(records.flatMap((r) => Object.keys(r.taskMeans)));
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

  function renderChart(rows) {
    const labels = rows.map((r) => r.label);
    const values = rows.map((r) => mean(Object.values(r.taskMeans)));
    const ctx = document.getElementById("results-chart");

    if (chart) {
      chart.destroy();
    }
    chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Mean Accuracy",
            data: values,
            backgroundColor: "rgba(34, 211, 238, 0.55)",
            borderColor: "rgba(34, 211, 238, 1)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: {
            min: 0,
            max: 1,
          },
        },
      },
    });
  }

  function rerender() {
    const rows = filteredRecords();
    statusEl.textContent = `Showing ${rows.length} model entries`;
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
