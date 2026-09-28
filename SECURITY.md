# Security Policy

This is a research project, but a few things are worth handling carefully.

## Reporting a vulnerability

If you find a security issue (for example, an accidental credential commit or a
dependency vulnerability), please **do not open a public issue**. Instead, use
GitHub's [private vulnerability reporting](https://github.com/vardhjain/graphrag-pubmedqa-ablation/security/advisories/new)
or email the maintainer at vardhjain20@gmail.com. You can expect an acknowledgement within a few days.

## Secrets

- Never commit real credentials. ArangoDB, Neo4j (`NEO4J_*`), and LLM settings
  (including `GEMINI_API_KEY`) are read from the environment (or a local `.env`,
  which is git-ignored); the hosted backend's secrets live in Render's dashboard. Use `.env.example` as a
  template, and Colab **Secrets** for notebook runs.
- If a secret is ever committed, rotate it immediately — removing it from the
  latest commit is not enough, as it remains in git history. The `gitleaks`
  pre-commit hook (`pre-commit install`) scans staged changes for secrets.

## Supported versions

The latest release on `main` is supported. This project pins minimum dependency
versions in `requirements.txt`; run `pip list --outdated` periodically.
