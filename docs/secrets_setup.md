# Secrets setup

The LEVANTE assets bucket is **public**; no authentication is required. The bucket base URL is set in config (`levante_bench.config.defaults`) and can be overridden with the environment variable `LEVANTE_ASSETS_BUCKET_URL`. Do **not** commit real secrets (API keys, private bucket URLs) in the repository.

If you add features that require secrets (e.g. a private Redivis token or GCP credentials), store them in a **gitignored** file such as `.secrets` at the project root, and document the expected format here. Never commit `.secrets` or any file containing secrets.

## Remote model credentials

For remote/provider-backed model configs used by runtime exports:

- `gemini_pro` requires `GEMINI_API_KEY`
- `gpt53` / `gpt52` require `OPENAI_API_KEY`

Set these in your shell or a gitignored env file before calling runtime APIs:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
```
