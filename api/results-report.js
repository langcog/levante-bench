const fs = require("fs/promises");
const path = require("path");

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

module.exports = async function handler(req, res) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");

  const reportUrl = process.env.RESULTS_REPORT_URL;
  try {
    const payload = reportUrl ? await readRemoteReport(reportUrl) : await readLocalReport();
    res.status(200).send(
      JSON.stringify(
        {
          source: reportUrl ? "remote" : "local",
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
            "Set RESULTS_REPORT_URL in Vercel to a public JSON endpoint, or",
            "Run scripts/analysis/build_model_comparison_report.py locally so results/model-comparison-report.json exists.",
          ],
        },
        null,
        2,
      ),
    );
  }
};
