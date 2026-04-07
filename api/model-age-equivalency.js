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
  const idxClosestBin = header.indexOf("closest_ability_bin");
  const idxSoftAgeMean = header.indexOf("soft_age_eq_mean");
  const idxSoftAgeMedian = header.indexOf("soft_age_eq_median");
  const idxConfidence = header.indexOf("soft_match_confidence");
  if (idxTask < 0 || idxModel < 0 || idxClosestBin < 0 || idxSoftAgeMean < 0) {
    return [];
  }

  const out = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    const task = (cols[idxTask] || "").trim();
    const model = (cols[idxModel] || "").trim();
    const closest_ability_bin = (cols[idxClosestBin] || "").trim();
    const soft_age_eq_mean = Number(cols[idxSoftAgeMean]);
    const soft_age_eq_median = idxSoftAgeMedian >= 0 ? Number(cols[idxSoftAgeMedian]) : null;
    const soft_match_confidence = idxConfidence >= 0 ? Number(cols[idxConfidence]) : null;
    if (!task || !model || !closest_ability_bin || !Number.isFinite(soft_age_eq_mean)) {
      continue;
    }
    out.push({
      task,
      model,
      closest_ability_bin,
      soft_age_eq_mean,
      soft_age_eq_median: Number.isFinite(soft_age_eq_median) ? soft_age_eq_median : null,
      soft_match_confidence: Number.isFinite(soft_match_confidence)
        ? soft_match_confidence
        : null,
    });
  }
  return out;
}

async function readBucketCsv(bucketName, bucketPrefix, objectName) {
  const cleanPrefix = String(bucketPrefix || "results/comparison").replace(/^\/+|\/+$/g, "");
  const objectPath = cleanPrefix ? `${cleanPrefix}/${objectName}` : objectName;
  const url = `https://storage.googleapis.com/${bucketName}/${objectPath}`;
  const response = await fetch(url, {
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

  const sourceMode = process.env.MODEL_AGE_EQ_SOURCE_MODE || "bucket";
  const bucketName = process.env.RESULTS_BUCKET_NAME || "levante-bench";
  const bucketPrefix = process.env.MODEL_AGE_EQ_BUCKET_PREFIX || "results/comparison";
  const bucketObjectName =
    process.env.MODEL_AGE_EQ_BUCKET_OBJECT || "model_age_equivalency.csv";
  const localCsvPath =
    process.env.MODEL_AGE_EQ_LOCAL_CSV || "results/comparison/model_age_equivalency.csv";

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
      throw new Error("No source for model age equivalency CSV succeeded.");
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
          error: "Could not load model age equivalency data.",
          details: String(error && error.message ? error.message : error),
        },
        null,
        2,
      ),
    );
  }
};
