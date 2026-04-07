const fs = require("fs/promises");
const path = require("path");

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const MODEL_SIZE_UNDERSCORE_RE = /^(?<model>[A-Za-z0-9.-]+)_(?<size>[0-9]+(?:\.[0-9]+)?[A-Za-z]+)$/;
const MODEL_SIZE_DASH_RE =
  /^(?<model>[A-Za-z0-9._-]+)-(?<size>(?:\d+(?:\.\d+)?[A-Za-z]+|[A-Za-z]+\d+[A-Za-z]*)(?:-(?:it|instruct))?)$/;
const LANG_SUFFIX_RE = /^(?<base>.+)-(?<lang>[a-z]{2})$/;

function mean(values) {
  if (!values.length) {
    return null;
  }
  return values.reduce((a, b) => a + b, 0) / values.length;
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

function parseKlCsv(csvText) {
  const lines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    return [];
  }
  const header = lines[0].split(",");
  const idxTask = header.indexOf("task");
  const idxModel = header.indexOf("model");
  const idxBin = header.indexOf("ability_bin");
  const idxDkl = header.indexOf("D_KL") >= 0 ? header.indexOf("D_KL") : header.indexOf("d_kl");
  if (idxTask < 0 || idxModel < 0 || idxBin < 0 || idxDkl < 0) {
    return [];
  }

  const out = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    const task_id = (cols[idxTask] || "").trim();
    const model = (cols[idxModel] || "").trim();
    const ability_bin = (cols[idxBin] || "").trim();
    const d_kl = Number(cols[idxDkl]);
    if (!task_id || !model || !ability_bin || !Number.isFinite(d_kl)) {
      continue;
    }
    out.push({ task_id, model, ability_bin, d_kl });
  }
  return out;
}

function aggregateByModel(records) {
  const grouped = new Map();
  for (const row of records) {
    if (!grouped.has(row.model)) {
      grouped.set(row.model, []);
    }
    grouped.get(row.model).push(row);
  }

  const byModel = {};
  for (const [modelTag, rows] of Array.from(grouped.entries()).sort((a, b) => a[0].localeCompare(b[0]))) {
    const { model, size, language } = splitModelSizeLanguage(modelTag);
    const taskMap = {};
    for (const row of rows) {
      if (!taskMap[row.task_id]) {
        taskMap[row.task_id] = [];
      }
      taskMap[row.task_id].push(row.d_kl);
    }
    const task_stats = {};
    for (const taskId of Object.keys(taskMap).sort()) {
      const values = taskMap[taskId];
      task_stats[taskId] = {
        count: values.length,
        min: Math.min(...values),
        max: Math.max(...values),
        mean: mean(values),
        values,
      };
    }
    byModel[modelTag] = {
      model,
      size,
      language,
      canonical_model_tag: canonicalModelTag(model, size, language),
      model_tag_examples: [modelTag],
      run_count: 1,
      runs: [],
      versions_seen: [],
      task_stats,
    };
  }
  return byModel;
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

async function readBucketText(url) {
  const response = await fetch(url, { headers: { Accept: "text/plain,text/csv" } });
  if (!response.ok) {
    throw new Error(`Bucket CSV fetch failed: HTTP ${response.status}`);
  }
  return await response.text();
}

async function readBucketRecords(bucketName, prefix) {
  const cleanPrefix = String(prefix || "").replace(/^\/+|\/+$/g, "");
  const objects = await listBucketObjects(bucketName, cleanPrefix);
  const csvObjects = objects.filter((obj) => obj.name.endsWith("_d_kl.csv"));
  const records = [];
  for (const obj of csvObjects) {
    const url = `https://storage.googleapis.com/${bucketName}/${obj.name}`;
    const csvText = await readBucketText(url);
    records.push(...parseKlCsv(csvText));
  }
  return records;
}

async function readLocalRecords(localDir) {
  const dirPath = path.join(process.cwd(), localDir);
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  const records = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith("_d_kl.csv")) {
      continue;
    }
    const raw = await fs.readFile(path.join(dirPath, entry.name), "utf8");
    records.push(...parseKlCsv(raw));
  }
  return records;
}

module.exports = async function handler(_req, res) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");

  const sourceMode = process.env.KL_REPORT_SOURCE_MODE || "auto";
  const bucketName = process.env.RESULTS_BUCKET_NAME || "levante-bench";
  const bucketPrefix = process.env.KL_REPORT_BUCKET_PREFIX || "results/comparison";
  const localDir = process.env.KL_REPORT_LOCAL_DIR || "results/comparison";

  try {
    let records = null;
    let source = "unknown";
    if (sourceMode === "bucket" || sourceMode === "auto") {
      try {
        records = await readBucketRecords(bucketName, bucketPrefix);
        source = "bucket";
      } catch (err) {
        if (sourceMode === "bucket") {
          throw err;
        }
      }
    }
    if (!records && (sourceMode === "local" || sourceMode === "auto")) {
      records = await readLocalRecords(localDir);
      source = "local";
    }
    if (!records) {
      throw new Error("No source for KL report succeeded.");
    }

    const report = {
      generated_at: new Date().toISOString(),
      metric: "d_kl",
      summary_file_count: records.length,
      by_model: aggregateByModel(records),
    };

    res.status(200).send(
      JSON.stringify(
        {
          source,
          generated_at: new Date().toISOString(),
          records,
          report,
        },
        null,
        2,
      ),
    );
  } catch (error) {
    res.status(500).send(
      JSON.stringify(
        {
          error: "Could not load KL report data.",
          details: String(error && error.message ? error.message : error),
        },
        null,
        2,
      ),
    );
  }
};
