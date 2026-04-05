const fs = require("fs/promises");
const path = require("path");

async function readBucketMarkdown(bucketName, bucketPrefix) {
  const cleanPrefix = String(bucketPrefix || "results").replace(/^\/+|\/+$/g, "");
  const objectPath = cleanPrefix ? `${cleanPrefix}/parser-glitch-report.md` : "parser-glitch-report.md";
  const url = `https://storage.googleapis.com/${bucketName}/${objectPath}`;
  const response = await fetch(url, {
    headers: { Accept: "text/markdown,text/plain;q=0.9,*/*;q=0.1" },
  });
  if (!response.ok) {
    throw new Error(`Bucket report fetch failed: HTTP ${response.status}`);
  }
  return await response.text();
}

async function readLocalMarkdown() {
  const reportPath = path.join(process.cwd(), "results", "parser-glitch-report.md");
  return await fs.readFile(reportPath, "utf8");
}

module.exports = async function handler(_req, res) {
  res.setHeader("Content-Type", "text/markdown; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");

  const sourceMode = process.env.PARSER_GLITCH_REPORT_SOURCE || "bucket";
  const bucketName = process.env.RESULTS_BUCKET_NAME || "levante-bench";
  const bucketPrefix = process.env.RESULTS_BUCKET_PREFIX || "results";

  try {
    let markdown = null;
    if (sourceMode === "bucket" || sourceMode === "auto") {
      try {
        markdown = await readBucketMarkdown(bucketName, bucketPrefix);
      } catch (err) {
        if (sourceMode === "bucket") {
          throw err;
        }
      }
    }
    if (!markdown && (sourceMode === "local" || sourceMode === "auto")) {
      markdown = await readLocalMarkdown();
    }
    if (!markdown) {
      throw new Error("No parser glitch report source succeeded.");
    }
    res.status(200).send(markdown);
  } catch (error) {
    res
      .status(500)
      .send(
        `# Parser Glitch Report Unavailable\n\n${String(error && error.message ? error.message : error)}\n`,
      );
  }
};
