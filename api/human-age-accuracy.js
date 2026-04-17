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
  const idxAge = header.indexOf("age_bin");
  const idxTask = header.indexOf("task_id");
  const idxLanguage = header.indexOf("language");
  const idxAcc = header.indexOf("accuracy");
  const idxN = header.indexOf("n");
  if (idxAge < 0 || idxTask < 0 || idxAcc < 0) {
    return [];
  }
  const out = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    const age_bin = (cols[idxAge] || "").trim();
    const task_id = (cols[idxTask] || "").trim();
    const language = idxLanguage >= 0 ? (cols[idxLanguage] || "").trim().toLowerCase() : "unknown";
    const accuracy = Number(cols[idxAcc]);
    const n = idxN >= 0 ? Number(cols[idxN]) : null;
    if (!age_bin || !task_id || !Number.isFinite(accuracy)) {
      continue;
    }
    out.push({
      age_bin,
      task_id,
      language: language || "unknown",
      accuracy,
      n: Number.isFinite(n) ? n : null,
    });
  }
  return out;
}

async function readBucketCsv(bucketName, bucketPrefix, objectName) {
  const cleanPrefix = String(bucketPrefix || "results").replace(/^\/+|\/+$/g, "");
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

  const sourceMode = process.env.HUMAN_AGE_SOURCE_MODE || "bucket";
  const bucketName = process.env.RESULTS_BUCKET_NAME || "levante-bench";
  const bucketPrefix = process.env.RESULTS_BUCKET_PREFIX || "results";
  const bucketObjectName = process.env.HUMAN_AGE_BUCKET_OBJECT || "human-accuracy-by-age-lines.csv";
  const localCsvPath = process.env.HUMAN_AGE_LOCAL_CSV || "results/human-accuracy-by-age-lines.csv";

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
      throw new Error("No source for human age CSV succeeded.");
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
          error: "Could not load human age accuracy data.",
          details: String(error && error.message ? error.message : error),
        },
        null,
        2,
      ),
    );
  }
};
