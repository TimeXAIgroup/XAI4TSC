---
name: Bug report
about: Report something that is not working as expected
title: "[Bug] "
labels: bug
assignees: ''
---

## Description

A clear and concise description of what the bug is.

## Steps to reproduce

A minimal reproducer makes this far easier to fix. Prefer one of:

- A small **YAML config** run with
  `python -m experiment_runner.main --conf <your-config>.yaml`, or
- A short **package snippet** (a few lines using the `xai4tsc` API).

```text
# config or code here
```

## Expected behavior

What you expected to happen.

## Actual behavior

What actually happened. Include the full traceback or relevant log output
(run with `--debug` for more detail).

```text
# logs / traceback here
```

## Environment

- OS:
- Python version:
- XAI4TSC version / commit:
- Usage: [ ] Standalone runner [ ] package

## Additional context

Anything else that might help — dataset, model, explainer, or metric involved.
