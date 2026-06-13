## Summary

<!-- What does this PR change, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Docs
- [ ] Benchmark / results

## Checklist

- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Docs/README updated if behavior changed

## Fairness (retrieval/evaluation changes only)

- [ ] Confounders (embedder, reranker, prompt, LLM, top-k, seed, n) stay in
      `config.py` and identical across arms
- [ ] No benchmark question/answer can leak into a retrieved context
      (the leakage regression test still passes)

## Notes

<!-- Anything reviewers should know: trade-offs, follow-ups, screenshots. -->
