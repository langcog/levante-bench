const fs = require("fs/promises");
const path = require("path");

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const MODEL_SIZE_UNDERSCORE_RE = /^(?<model>[A-Za-z0-9.-]+)_(?<size>[0-9]+(?:\.[0-9]+)?[A-Za-z]+)$/;
const MODEL_SIZE_DASH_RE =
  /^(?<model>[A-Za-z0-9._-]+)-(?<size>(?:\d+(?:\.\d+)?[A-Za-z]+|[A-Za-z]+\d+[A-Za-z]*)(?:-(?:it|instruct))?)$/;
const LANG_SUFFIX_RE = /^(?<base>.+)-(?<lang>[a-z]{2})$/;

async function readLocalReport() {
  const reportPath = path.join(process.cwd(), "results", "model-comparison-report.json");
  const raw = await fs.readFile(reportPath, "utf8");
  return JSON.parse(raw);
}

async function readRemoteReport(url) {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    throw new Error(`Remote fetch failed: HTTP ${response.status}`);
  }
  return await response.json();
}

async function readText(url) {
  const response = await fetch(url, {
    headers: { Accept: "text/plain" },
  });
  if (!response.ok) {
    throw new Error(`Fetch failed (${url}): HTTP ${response.status}`);
  }
  return await response.text();
}

function inferModelTagFromPath(relativeSummaryPath) {
  const parts = relativeSummaryPath.split("/").slice(0, -1); // drop summary.csv
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    if (!DATE_RE.test(parts[i])) {
      return parts[i];
    }
  }
  return parts[parts.length - 1] || "unknown";
}

function splitModelSizeLanguage(modelTag) {
  let m = modelTag.match(MODEL_SIZE_UNDERSCORE_RE);
  if (m && m.groups) {
    return { model: m.groups.model, size: m.groups.size, language: null };
  }
  m = modelTag.match(MODEL_SIZE_DASH_RE);
  if (m && m.groups) {
    return { model: m.groups.model, size: m.groups.size, language: null };
  }

  let base = modelTag;
  let language = null;
  const lm = modelTag.match(LANG_SUFFIX_RE);
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

function canonicalModelTag(model, size, language) {
  const base = size ? `${model}-${size}` : model;
  return language ? `${base}-${language}` : base;
}

function parseSummaryCsv(csvText) {
  const lines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return {};
  }
  const header = lines[0].split(",");
  const taskIdx = header.indexOf("task_id");
  const accIdx = header.indexOf("accuracy");
  if (taskIdx < 0 || accIdx < 0) {
    return {};
  }
  const out = {};
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    const taskId = (cols[taskIdx] || "").trim();
    const acc = Number(cols[accIdx]);
    if (taskId && Number.isFinite(acc)) {
      out[taskId] = acc;
    }
  }
  return out;
}

function mean(values) {
  if (!values.length) {
    return null;
  }
  return values.reduce((a, b) => a + b, 0) / values.length;
}

async function listBucketObjects(bucketName, prefix) {
  const objects = [];
  let pageToken = null;
  do {
    const url = new URL(`https://storage.googleapis.com/storage/v1/b/${bucketName}/o`);
    url.searchParams.set("prefix", prefix);
    url.searchParams.set("fields", "items(name,updated),nextPageToken");
    if (pageToken) {
      url.searchParams.set("pageToken", pageToken);
    }
    const response = await fetch(url.toString(), { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`Bucket listing failed: HTTP ${response.status}`);
    }
    const payload = await response.json();
    for (const item of payload.items || []) {
      objects.push(item);
    }
    pageToken = payload.nextPageToken || null;
  } while (pageToken);
  return objects;
}

async function buildReportFromBucket(bucketName, prefix) {
  const allObjects = await listBucketObjects(bucketName, prefix);
  const summaryObjects = allObjects.filter((obj) => obj.name.endsWith("/summary.csv"));

  const runs = [];
  for (const obj of summaryObjects) {
    const relativePath = obj.name.startsWith(prefix) ? obj.name.slice(prefix.length) : obj.name;
    const cleanedRelative = relativePath.replace(/^\/+/, "");
    const summaryUrl = `https://storage.googleapis.com/${bucketName}/${obj.name}`;
    const csvText = await readText(summaryUrl);
    const taskMetrics = parseSummaryCsv(csvText);
    const modelTag = inferModelTagFromPath(cleanedRelative);
    const { model, size, language } = splitModelSizeLanguage(modelTag);
    runs.push({
      summary_path: summaryUrl,
      relative_path: cleanedRelative,
      run_id: cleanedRelative.replace(/\/summary\.csv$/, ""),
      model,
      size,
      language,
      model_tag: modelTag,
      canonical_model_tag: canonicalModelTag(model, size, language),
      version: null,
      modified_at: obj.updated || null,
      task_metrics: taskMetrics,
      mean_accuracy: mean(Object.values(taskMetrics)),
    });
  }

  const grouped = new Map();
  for (const run of runs) {
    const key = `${run.model}|${run.size || ""}|${run.language || ""}`;
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key).push(run);
  }

  const byModel = {};
  for (const [key, groupRuns] of Array.from(grouped.entries()).sort((a, b) => a[0].localeCompare(b[0]))) {
    const sample = groupRuns[0];
    const taskValues = {};
    for (const run of groupRuns) {
      for (const [taskId, acc] of Object.entries(run.task_metrics || {})) {
        if (!taskValues[taskId]) {
          taskValues[taskId] = [];
        }
        taskValues[taskId].push(acc);
      }
    }
    const taskStats = {};
    for (const taskId of Object.keys(taskValues).sort()) {
      const values = taskValues[taskId];
      taskStats[taskId] = {
        count: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        mean: mean(values),
        values,
      };
    }
    byModel[key] = {
      model: sample.model,
      size: sample.size,
      language: sample.language,
      canonical_model_tag: canonicalModelTag(sample.model, sample.size, sample.language),
      model_tag_examples: Array.from(new Set(groupRuns.map((r) => r.model_tag))).sort(),
      run_count: groupRuns.length,
      runs: groupRuns.map((r) => r.run_id).sort(),
      versions_seen: [],
      task_stats: taskStats,
    };
  }

  return {
    generated_at: new Date().toISOString(),
    results_root: `gs://${bucketName}/${prefix}`,
    summary_file_count: runs.length,
    runs,
    by_model: byModel,
  };
}

module.exports = async function handler(req, res) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");

  const sourceMode = process.env.RESULTS_SOURCE_MODE || "bucket_compute";
  const reportUrl = process.env.RESULTS_REPORT_URL;
  const bucketName = process.env.RESULTS_BUCKET_NAME || "levante-bench";
  const bucketPrefix = (process.env.RESULTS_BUCKET_PREFIX || "results").replace(/^\/+|\/+$/g, "");

  try {
    let payload = null;
    let source = "unknown";

    if (sourceMode === "bucket_compute" || sourceMode === "auto") {
      try {
        payload = await buildReportFromBucket(bucketName, bucketPrefix);
        source = "bucket-computed";
      } catch (bucketErr) {
        if (sourceMode === "bucket_compute") {
          throw bucketErr;
        }
      }
    }

    if (!payload && (sourceMode === "remote" || sourceMode === "auto") && reportUrl) {
      payload = await readRemoteReport(reportUrl);
      source = "remote";
    }
    if (!payload && (sourceMode === "local" || sourceMode === "auto")) {
      payload = await readLocalReport();
      source = "local";
    }
    if (!payload) {
      throw new Error("No data source succeeded.");
    }

    res.status(200).send(
      JSON.stringify(
        {
          source,
          report: payload,
        },
        null,
        2,
      ),
    );
  } catch (error) {
    res.status(500).send(
      JSON.stringify(
        {
          error: "Could not load model comparison report.",
          details: String(error && error.message ? error.message : error),
          hints: [
            "Use RESULTS_SOURCE_MODE=bucket_compute for live bucket aggregation.",
            "Or set RESULTS_REPORT_URL in Vercel to a public JSON endpoint.",
            "Or run scripts/analysis/build_model_comparison_report.py locally so results/model-comparison-report.json exists.",
          ],
        },
        null,
        2,
      ),
    );
  }
};
