# Disclaimer

XAI4TSC is a research framework. Please read the points below before relying on
its outputs for scientific conclusions.

## Implementation correctness

Many of the models, explainers, and metrics shipped with XAI4TSC are
**clean-room re-implementations** of methods from the research literature. These
implementations have **not been verified by the original authors** and may
differ from the source papers or any reference code — in subtleties of
normalisation, default parameters, or numerical detail that can change results.

Where XAI4TSC instead wraps a third-party library (for example, Captum for
feature attributions or Quantus for evaluation metrics), the behaviour and
caveats of that library apply, and those wrapped implementations carry their own
assumptions that are likewise not guaranteed to match a method's original
description.

## Evaluation metrics are interpretations

Explanation-quality metrics are an **empirical, and sometimes contested, proxy**
for what a "good" explanation is. A metric measures what its implementation
computes, which may not coincide with what the original authors intended it to
capture, and different metrics can disagree. High coverage of a metric's code,
or a confident-looking number, does not by itself establish that an explanation
is faithful or useful.

## Using XAI4TSC responsibly

When using XAI4TSC for research or in publications, we recommend that you:

- **Cite the original papers** for every model, explainer, and metric you use,
  not just XAI4TSC.
- **State which implementation you used** (XAI4TSC's re-implementation versus a
  wrapped library) and its version.
- **Validate against a reference implementation** where one is available, and
  report any discrepancies.
- **Interpret scores in context** — compare methods under identical settings
  rather than reading absolute values as ground truth.

If you find a discrepancy between an XAI4TSC implementation and the method it is
based on, please open an issue or a pull request (see
[CONTRIBUTING.md](https://github.com/TimeXAI-group/XAI4TSC/blob/main/CONTRIBUTING.md)).
Corrections and clarifications are very welcome.
