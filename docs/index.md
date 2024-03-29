---
hide:
  - navigation
---

# Welcome to :high_voltage: Lightning-Boost :rocket:

Lightning-Boost is an extension of the [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework to develop deep learning models in [PyTorch](https://pytorch.org/) even faster :dashing_away:.

PyTorch Lightning already saves its versed users a lot of time, as large chunks of the PyTorch code being necessary to train deep neural networks are actually boilerplate.
However, with parts of the structure and code still being shared by most not too exotic projects, there is potential for further optimization.

In essence, Lightning-Boost was born out of the need to not create a codebase from scratch for every deep learning project.
It provides three key features that help users to develop their models not only faster, but also in a more structured way:

### :octicons-command-palette-24:&ensp;Command Line Interface and Configuration Files

Powered by the [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html), Lightning-Boost unifies the configuration of deep learning models and their training process. This is accompanied by YAML-based configuration files that can be generated automatically and be used instead of mile-long parametrizations in run-scripts calls.
And the best: A single line of code in the run-script is sufficient, extensive `ArgumentParser` definitions are now a thing of the past.

### :octicons-stack-24:&ensp;Standardized Project Structure

Lightning-Boost also unifies the structure of a deep learning project in a highly modularized fashion. It provides a clear logical separation between a *model*, which takes a well-defined input and produces -- as a function -- an equally well-defined output, and a *system*, which operates on one or more models, given data, and manages the whole training process, [as already recommended by PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html). Moreover, Lightning-Boost suggests a directory structure that does not necessarily have to be used, but perfectly fits into this logical framework and helps to stay on top of things.

### :octicons-code-square-24:&ensp;Base Classes for Common Functionality

As both the management of the training process and datasets share some common functionality across projects, respectively, Lightning-Boost comes with two base classes for the corresponding concepts of a system and a datemodule. Analogously, further base classes are provided for models, datasets and many other components. When developing for a new project, users need to implement only a small number of methods that contain the exact functionality specific to their tasks.

## Installation

Simply install via pip:

```
pip install lightning-boost
```

## Getting started

For newcomers, it is recommended to work through the [tutorials](tutorials/index.md). Basic knowledge of PyTorch and PyTorch Lightning is assumed.