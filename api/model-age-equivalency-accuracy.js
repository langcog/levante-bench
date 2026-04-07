const fs = require("fs/promises");
const path = require("path");

function parseCsv(csvText) {
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
  const idxLang = header.indexOf("language");
  const idxSoftAge = header.indexOf("soft_age_eq_accuracy");
  const idxNearestBin = header.indexOf("nearest_age_bin");
  const idxGap = header.indexOf("accuracy_gap");
  if (idxTask < 0 || idxModel < 0 || idxSoftAge < 0) {
    return [];
  }
  const out = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    const task = (cols[idxTask] || "").trim();
    const model = (cols[idxModel] || "").trim();
    const language = idxLang >= 0 ? (cols[idxLang] || "").trim().toLowerCase() : "en";
    const soft_age_eq_accuracy = Number(cols[idxSoftAge]);
    const nearest_age_bin = idxNearestBin >= 0 ? (cols[idxNearestBin] || "").trim() : "";
    const accuracy_gap = idxGap >= 0 ? Number(cols[idxGap]) : null;
    if (!task || !model || !Number.isFinite(soft_age_eq_accuracy)) {
      continue;
    }
    out.push({
      task,
      model,
      language: language || "en",
      soft_age_eq_accuracy,
      nearest_age_bin: nearest_age_bin || null,
      accuracy_gap: Number.isFinite(accuracy_gap) ? accuracy_gap : null,
    });
  }
  return out;
}

async function readBucketCsv(bucketName, bucketPrefix, objectName) {
  const cleanPrefix = String(bucketPrefix || "results/comparison").replace(/^\/+|\/+$/g, "");
  const objectPath = cleanPrefix ? `${cleanPrefix}/${objectName}` : objectName;
  const url = `https://storage.googleapis.com/${bucketName}/${objectPath}?t=${Date.now()}`;
  const response = await fetch(url, {
    cache: "no-store",
    headers: { Accept: "text/csv,text/plain;q=0.9,*/*;q=0.1" },
  });
  if (!response.ok) {
    throw new Error(`Bucket CSV fetch failed: HTTP ${response.status}`);
  }
  return await response.text();
}

async function readLocalCsv(localPath) {
  const csvPath = path.join(process.cwd(), localPath);
  return await fs.readFile(csvPath, "utf8");
}

module.exports = async function handler(_req, res) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");

  const sourceMode = process.env.MODEL_AGE_EQ_ACC_SOURCE_MODE || "bucket";
  // Pin this endpoint to the canonical production artifact location.
  const bucketName = "levante-bench";
  const bucketPrefix = "results/comparison";
  const bucketObjectName = "model_age_equivalency_accuracy.csv";
  const localCsvPath =
    process.env.MODEL_AGE_EQ_ACC_LOCAL_CSV ||
    "results/comparison/model_age_equivalency_accuracy.csv";

  try {
    let csvText = null;
    let source = "unknown";
    if (sourceMode === "bucket" || sourceMode === "auto") {
      try {
        csvText = await readBucketCsv(bucketName, bucketPrefix, bucketObjectName);
        source = "bucket";
      } catch (err) {
        if (sourceMode === "bucket") {
          throw err;
        }
      }
    }
    if (!csvText && (sourceMode === "local" || sourceMode === "auto")) {
      csvText = await readLocalCsv(localCsvPath);
      source = "local";
    }
    if (!csvText) {
      throw new Error("No source for model age-equivalency (accuracy) CSV succeeded.");
    }
    const records = parseCsv(csvText);
    res.status(200).send(
      JSON.stringify(
        {
          source,
          generated_at: new Date().toISOString(),
          records,
        },
        null,
        2,
      ),
    );
  } catch (error) {
    res.status(500).send(
      JSON.stringify(
        {
          error: "Could not load model age-equivalency (accuracy) data.",
          details: String(error && error.message ? error.message : error),
        },
        null,
        2,
      ),
    );
  }
};
