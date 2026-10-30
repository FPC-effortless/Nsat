# DEEP-FIX001: Validate all temporal split boundaries

The split API rejects missing timestamps, empty/reversed train-validation windows and validation/test overlap. Valid strict temporal splits preserve train/validation/gap/test labels. This constructed configuration regression is not evidence that the existing data was contaminated.

Validation: Three pandas regression tests passed.

These are source/correctness diagnostics, not real-data capability evidence. No model training, GPU quota, protected holdout, merge or historical score rewrite. Portfolio audit: FPC-effortless/nsf-vibe PR3. Draft for source review.
