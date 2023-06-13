---
hide:
  - navigation
---

# Explanation

  In this section, specific aspects of Lightning-Boost are explained in depth.


## Models vs. Systems

  In Lightning-Boost, there is a differentiation between a model and a system:

  - A *model* is a parametrized function. It has a well-defined signature, taking one or more tensors as input and producing one or more tensors as output.
  - A *system* is an instance that embeds one or more models, and manages the entire training process, given data through a datamodule.

  This clear logical separation has several advantages in terms of modularity and code structure:

  - A model can be exchanged by another model, if their signature is the same.
  - Multiple models can easily be combined in a multi-model system, e.g., for a multi-task pretraining-finetuning setup.
  - In general, code becomes clearer and easier to maintain.


## Task-oriented Data Processing

  In more complex scenarios, a model might have to solve a variety of tasks at once.

  Lightning-Boost allows tackling several tasks in one system through *task-oriented data processing*. This is enabled by the data structures used for input, target and prediction data, which are dictionaries. 
  Task-specific data is then logically assigned to a task by setting its key accordingly, making the code easy to understand.
  Furthermore, it allows the automatic processing of target and prediction data through loss functions and metrics within a specific task.