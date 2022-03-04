# Contributing to Iguanas

First of all, thank you for taking the time to contribute to the project! We love having you here :grinning:

## Table of Contents

- [Getting started](#getting-started)
- [Testing](#testing)
- [Submitting a change](#submitting-a-change)
- [Sphinx documentation process](#sphinx-documentation-process)
  - [Adding a class to an existing module](#adding-a-class-to-an-existing-module)
  - [Adding a new module](#adding-a-new-module)
  - [Updating the documentation](#updating-the-documentation)
- [Styleguide](#styleguide)
  - [Code](#code)
  - [Docstrings](#docstrings)
  - [Notebooks](#notebooks)
- [Raising a bug](#raising-a-bug)
- [Requesting a feature](#requesting-a-feature)
- [Code of Conduct](#code-of-conduct)
- [I need more help!](#i-need-more-help)

## Getting started

Before you can contribute, you need to create a virtual environment which Iguanas will be installed into:

```bash
cd <path-for-iguanas-virtual-environment>
python3 -m venv iguanas_dev
```

Then activate the virtual environment:

```bash
source <path-to-iguanas-virtual-environment>/bin/activate
```

Now `cd` to the location where you want to store the Iguanas repo and clone it using the command:

```bash
git clone https://github.com/paypal/Iguanas.git
```

`cd` to the `Iguanas` folder and install Iguanas in editable mode using the command:

```bash
pip install -e .
```

Then install the `dev` extras:

```bash
pip install .[dev]
```

If contributing to a module that utilises Spark, you need to install the `spark` extras:

```bash
pip install .[spark]
```

Once Iguanas has been installed in the virtual environment, create a jupyter kernel using the command:

```bash
python -m ipykernel install --user --name iguanas_dev
```

## Testing

You can run all unit, docstring and notebook tests for the **Spark and non-Spark modules** using the command:

```bash
sh <path-to-Iguanas-repo>/run_all_tests.sh iguanas_dev
```

You can run the tests for the **non-Spark modules only** using the command:

```bash
sh <path-to-Iguanas-repo>/run_tests_non_spark.sh iguanas_dev
```

You can run the tests for the **Spark modules only** using the command:

```bash
sh <path-to-Iguanas-repo>/run_tests_spark.sh iguanas_dev
```

## Submitting a change

You can submit a change by raising a [pull-request](https://github.com/paypal/Iguanas/pulls) and assigning the reviewer as James Laidler.

Please ensure that, before raising a pull-request:

- Your contribution conforms to the [Style guide](#styleguide).
- Your code has been profiled thouroughly to ensure runtime is optimised.
- Unit tests are added/extended.
- Unit test coverage is >95%.
- Docstrings are added in the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) format, using the same style as the existing docstrings.
- An example notebook is added, in the same style as the existing notebooks.
- The documentation has been updated by running the [Documentation update process](#sphinx-documentation-process).

## Documentation update process

To ensure that any changes or additions to Iguanas are reflected in the documentation, there are a few changes that need to be made to the files in the `run_doc` folder (located in the Iguanas parent directory). The changes that need to be made depend on whether a class has been added to an existing module, or a new module has been created:

### Adding a class to an existing module

If a class has been added to an existing module, the following `.rst` files need to be updated:

1. The class name must be added to the module's `.rst` file (under the `autosummary` Sphinx command) in the `run_doc/files/api` folder.
2. If the class requires it's own notebook, the name of this notebook (without the extension) must be added to the module's `index.rst` file (under the `toctree` Sphinx command) in the `run_doc/files/examples/<module-name>` folder.

### Adding a new module

If a new module has been added to Iguanas, the following `.rst` files need to be created:

1. An `.rst` file of the module's name must be added to the `run_doc/files/api` folder (e.g. if a new module named `new_module` was added to Iguanas, the file `new_module.rst` would be added). The `.rst` file should have the module's name as the header, a brief description of what the module does, then the classes listed under the `autosummary` command (see the `.rst` files in the `run_doc/files/api` folder for examples).
2. A folder of the module's name, containing a single `index.rst` file, must be added to the `run_doc/files/examples` folder. The `index.rst` file should have the module's name as the header, then the names of the example notebooks (without the extension) listed under the `toctree` command (see the `index.rst` files in the folders in the `run_doc/files/examples` folder for examples).
3. A line in the `autodoc.sh` file in the `run_doc` folder must be added (underneath the `Copy over example notebooks` header) to copy over the example notebooks of the new module into the module's folder in the `run_doc/files/examples` folder.

### Updating the documentation

Once the necessary changes have been made, the documentation can be updated by running the following command:

```bash
sh <path-to-Iguanas-repo>/run_doc/autodoc.sh
```

## Styleguide

### Code

Python code should follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) convention. It's recommended that you install [autopep8](https://pypi.org/project/autopep8/) before contributing, which will ensure that your code follows the PEP 8 convention.

### Docstrings

Docstrings should follow the [numpy](https://numpydoc.readthedocs.io/en/latest/format.html) format.

### Notebooks

Notebooks should follow the general structure:

- A brief introduction of the module, what it does and why it is used.
- Apply the module.
- A brief description of the outputs.

For example notebooks, see the [Examples](https://paypal.github.io/Iguanas/examples/index.html) section of the documentation.

## Raising a bug

You can raise a bug in the [Issues](https://github.com/paypal/Iguanas/issues) area of the repo.

## Requesting a feature

You can request a new feature in the [Issues](https://github.com/paypal/Iguanas/issues) area of the repo.

## Code of Conduct

You can find the Code of Conduct [here](https://github.com/paypal/Iguanas/blob/main/CODE_OF_CONDUCT.md).

## I need more help

If you have any other queries or questions, feel free to contact James Laidler:

- [Email](james.a.laidler@gmail.com)
- [Linkedin](https://www.linkedin.com/in/james-laidler-430571a7)
