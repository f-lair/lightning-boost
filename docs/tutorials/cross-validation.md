# Cross-Validation Tutorial

In this tutorial, we present Lightning-Boost's capabilities for cross-validation.

Cross-validation is an effective technique to mitigate the selection bias one creates when training their model of one part of the dataset and evaluating on a different part.
It is usually applied in model selection, such that the model is not optimized for the specific data split selected, but generalizes well across different possible splits.
Currently, Lightning-Boost provides support for basic k-fold cross-validation, but can be extended by custom implementations, e.g., for time-series problems.

Given a working implementation in the Lightning-Boost framework (e.g., our [MNIST example](mnist.md)), we have almost everything it takes to perform cross-validation.
In fact, we only need to specify the following arguments in the CLI:

`--num_folds`

:   The number of folds (*k*) in k-fold cross-validation.


`--fold_index`

:   The fold index in k-fold cross-validation (starting at 0).


`--fold_seed`

:   The random number generator seed used to permute the data items identically across several runs with different fold indices. 
    Otherwise, folds would potentially overlap.


In addition, it is useful, but not necessary to specify the arguments `--log_name` and `--log_version` as well.
They represent the names of directories, where logs are saved during a run.
A recommended disignation is an experiment-specific name for `--log_name` and `fold_{k}` for `--log_version`, where `{k}` is to be replaced by the fold index.

For example, given a working `config.yaml`, we can perform 5-fold cross-validation and train our model on the second fold with seed 42 and log name *cv_test* using the following command:

```
$ python main.py fit --config=config.yaml \
    --num_folds=5 --fold_index=1 --fold_seed=42 \
    --log_name=cv_test --log_version=fold_1
```

When *k* models have been fitted to the respective dataset folds, we can either evaluate them on the hold-out validation splits or the test dataset using the subcommands `validate` and `test` instead of `fit`, respectively.

Last, if one needs to implement their custom routines to generate folds in cross-validation, they might want to override the methods `determine_cv_indices()`, `determine_fold_len()`, and `get_cv_train_val_split()`.
For details, we refer to the [documentation of the datamodule class](../../reference/lightning_boost/data/datamodules/base_datamodule/).