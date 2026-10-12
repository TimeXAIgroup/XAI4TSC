# Third-party licenses

XAI4TSC is released under the [MIT License](LICENSE). This file records the
third-party code adapted into it, and reproduces the notices those licenses
require. Everything listed here is MIT licensed and therefore compatible with
this project's license.

Methods reimplemented from a published description rather than from source code
are cited in the relevant module's docstring and are not listed here, since no
third-party code was copied.

---

## multilabel-ts-xai-benchmark

Adapted in `src/xai4tsc/data/datasets.py`, as the generator and partition of the
`Synth_Multilabel_Freq` synthetic dataset
(`SynthMultilabelFreqDataset`). Derived from `data/benchmark.py`,
`generate_dataset.py` and `utils.py` of:

- Source: <https://github.com/TimeXAIgroup/multilabel-ts-xai-benchmark>
- Paper: J. Fischer, N. Gumpfer, T. Papenbrock, and J. Hannig, "A Synthetic
  Multi-Label Time Series Classification Dataset with Time-Frequency
  Ground-Truth Explanation for Evaluating Explainable Artificial Intelligence
  Methods", SSRN preprint 7389286.
- Dataset record: <https://doi.org/10.5281/zenodo.20744119>

The generator is ported rather than reimplemented so that its defaults reproduce
the published dataset bit-for-bit; the data itself is not redistributed here.

```
MIT License

Copyright (c) 2026, TimeXAIgroup

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
