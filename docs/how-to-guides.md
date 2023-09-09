---
hide:
  - navigation
---

# How-to Guides

  In this section, you can quickly look up how to implement a specific base class in Lightning-Boost and how to use its features.


## Directory Structure

  The recommended directory structure of Lightning-Boost is as follows:

  ```
  .
  ├── data/
  │   ├── datamodules/
  │   └── datasets/
  ├── models/
  ├── modules/
  │   ├── loss/
  │   ├── metrics/
  │   ├── preprocessing/
  │   └── trainable/
  └── systems/
  ```

  The purpose of most directories should be self-explanatory from their names.
  In particular, we differentiate between `loss functions` and `metrics`, depending on their inclusion in the computation of gradients, as well as `models` and `systems`, depending on their usage as functions with well-defined inputs and outputs, or instances that manage the entire training process for one or more such models (see [Explanation/Models vs. Systems](explanation.md#models-vs-systems)).
  Components in the `preprocessing` directory should also not contribute to the computation of gradients, but transform the input and target data before entering the model(s).
  Modules in the `trainable` directory, by contrast, are intended to be repetitive lower-level building blocks of models.


## Dataset

  **Base class:** `lightning_boost.data.datasets.BaseDataset`

### Mandatory methods

  - `__init__(self, root: str, download: bool = False, transform: Optional[BaseTransform] = None, **kwargs) -> None`:
      - Call super-class method first.
      - Load dataset from disk, stored at `self.path`.
  - `get_item(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]`: 
      - Return input and target data (as dictionaries) at index.
  - `__len__(self) -> int`: 
      - Return dataset size.

### Optional methods

  - `download(self) -> None`: 
      - Download dataset from the internet.
      - Store on disk at `self.path`.


## Transform

  **Base class:** `lightning_boost.modules.preprocessing.BaseTransform`

### Mandatory methods

  - `def __call__(self, inputs: Dict[str, Any], targets: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]`:
      - Perform transforms on elements in inputs and targets dictionaries.
      - Return two dictionaries again, keys can vary from received dictionaries.

### Implementation tipps

  - Define a subclass of `BaseTransform` for each subclass of `BaseDataset`.
  - To implement a composition of multiple transforms, use the class `lightning_boost.modules.preprocessing.CompositeTransform`.
  - Embed it as attribute in another `BaseTransform` subclass, then invoke its `__call__()` method in the `__call__()` method of the latter.


## Collator

  **Base class:** `lightning_boost.modules.preprocessing.BaseCollator`

### Mandatory methods

  - `get_collate_fn(self) -> Dict[str, Callable[[List[Tensor]], Tensor]]`:
      - Return a dictionary of collate functions, one per key/data type in the transform's output.
      - Use pre-defined collate functions `pad_collate_nd()` (stacks n-dimensional, potentially differently shaped tensors along new dimension, using padding) and `flatten_collate()` (concatenates zero-dimensional tensors/scalars to a one-dimensional tensor), if possible.
      - For `pad_collate_nd()`, you can specify custom padding values as well as padding shapes through the initialization parameters `pad_val`, `pad_shape` and `pad_dim` of`BaseCollator`.

### Implementation tipps

  - Define a subclass of `BaseCollator` for each subclass of `BaseDataset`.


## Datamodule

  **Base class:** `lightning_boost.data.datamodules.BaseDatamodule`

### Mandatory methods

  - `get_collator(self, **kwargs) -> BaseCollator`:
      - Returns collator instance for the used dataset.
  - `get_dataset_type(self, **kwargs) -> Type[BaseDataset]`:
      - Returns type (not instance!) of the used dataset.
  - `get_transform(self, **kwargs) -> BaseTransform`:
      - Returns transform instance for the used dataset.
  - `get_train_test_split(self) -> Tuple[BaseDataset, BaseDataset]`:
      - Returns training-test split for the used dataset.
      - Use the attribute `test_ratio` to build the split based on pre-defined ratios.
      - Use the method `instantiate_dataset()` to get an instance of the used dataset without explicitly passing the parameters `root`, `download` and `transform` of `BaseDataset`.

### Optional methods

  - `get_train_val_split(self) -> Tuple[BaseDataset, BaseDataset]`:
      - Returns training-validation split for the used dataset.
      - By default, uses the attribute `test_ratio` to build the split based on pre-defined ratios.
  - `get_cv_train_val_split(self) -> Tuple[BaseDataset, BaseDataset]`:
      - Returns training-validation split for the used dataset, when performing cross-validation.
      - By default, splits dataset into k parts and selects (k-1)/k and 1/k as training and validation split, respectively.
  - `determine_cv_indices(self) -> None`:
      - Determines permutation of data indices for cross-validation.
      - By default, generates random permutation using the RNG seed `fold_seed`.
  - `determine_fold_len(self) -> None`:
      - Determines fold sizes for cross-validation.
      - By default, uses equisized folds.

## Model

  **Base class:** `lightning_boost.models.BaseModel`

### Mandatory methods

  - `__init__(self, name: str | None = None) -> None`:
      - Call super-class method first.
      - Define model's submodules.
      - Specify parameter `name` explicitly in a unique manner, if multiple instances of the same model class are to be used in the system. 
  - `forward(self, *args: Tensor) -> Tensor | Sequence[Tensor]`:
      - Perform forward pass by feeding input tensors into the model's submodules, return output tensor(s).


## System

  **Base class:** `lightning_boost.systems.BaseSystem`

### Mandatory methods

  - `step(self, inputs: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]`:
      - Extract input tensors from inputs dictionary.
      - Feed input tensors into model(s), receive prediction tensor(s).
      - Return dictionary of prediction tensor(s), where keys correspond to tasks (they must match the keys of the target data dictionary!).


## Loss functions and metrics

  Loss functions and metrics do not need to be implemented using a base class. 
  Instead, simply import them in `__init__.py` files in the corresponding directories `./modules/loss` and `./modules/metrics`.
  Custom loss functions and metrics can be implemented by subclassing ``torch.nn.Module` and `torchmetrics.Metric`, respectively.


## Main script

  First, import all components (usually, models, systems, datamodules, datasets, loss functions, and metrics) you want to make accessible via the command line interface.
  This can be simplified by creating `__init__.py` files in the corresponding directories.
  Also import `lightning_boost.cli.LightningBoostCLI`.

  Then, instantiate the `LightningBoostCLI` in the main function.


## Execution

### Command line interface

  Call the main script with one of the subcommands `fit`, `validate` or `test` and mandatory arguments `--data` and `--system`, where you pass the class names of the datamodule and the system to be used, respectively.
  Their non-default parameters need to be set subsequently.
  While `BaseDatamodule` has no non-default parameters, `BaseSystem` needs the following to be specified:

  - `models`: One or more subclasses of `BaseModel`.
  - `loss`: One or more subclasses of `lightning_boost.modules.loss.TaskLoss`.
  - `optimizer`: Subclass of `torch.optim.Optimizer`.

  Further optional parameters are:

  - `lr_scheduler`: Subclass of `torch.optim.LRScheduler`.
  - `lr_scheduling_policy`: Use `lightning_boost.modules.optim.LRSchedulingPolicy` (default), adapt parameters if needed.
  - `train_metrics`: One or more suclasses of `lightning_boost.modules.metrics.TaskMetric`.
  - `val_metrics`: One or more suclasses of `lightning_boost.modules.metrics.TaskMetric`.
  - `test_metrics`: One or more suclasses of `lightning_boost.modules.metrics.TaskMetric`.

  In the CLI, 
  
  - arguments are passed using the operator `=`, e.g, `--arg_x=...`.
  - initialization parameters are set using dot notation, e.g., `--arg_x=class_a --arg_x.param_1=...`.
  - list arguments can be passed using the operator `+=` and specifying the initialization parameters directly after, `--arg_x+=class_a --arg_x.param_1=... --arg_x+=class_b --arg_x.param_1=...`.

### YAML configuration

  Create a YAML configuration file by adding the flag `--print_config` and the pipe `> config.yaml` to the command.
  Default parameters can be removed from the configuration file to increase readability.
  Run with a configuration file using the argument `--config=config.yaml`.