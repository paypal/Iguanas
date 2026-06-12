Installation
============

Requirements
------------

* Python >= 3.10
* Polars >= 1.0.0

Installing from PyPI
--------------------

The easiest way to install iguanas is via pip:

.. code-block:: bash

    pip install iguanas

Installing from Source
----------------------

To install the latest development version:

.. code-block:: bash

    git clone https://github.com/paypal/iguanas.git
    cd iguanas
    pip install -e .

For development (includes testing dependencies):

.. code-block:: bash

    pip install -e ".[dev]"

Verifying Installation
----------------------

To verify that iguanas is installed correctly:

.. code-block:: python

    import iguanas
    print(iguanas.__version__)

You should see the version number printed without any errors.

Dependencies
------------

Core dependencies (installed automatically):

* **polars** - High-performance DataFrame library
* **pandas** - DataFrame library for interoperability
* **XGBoost** - For rule generation using gradient boosting models
* **joblib** - For parallel processing during rule generation
* **scikit-learn** - For model evaluation and metrics
* **pydantic** - For data validation and settings management

Optional dependencies for specific features:

Development extras:

* **pytest** - For running tests
* **pytest-cov** - For test coverage reporting
* **ruff** - For linting and code quality checks
* **mypy** - For static type checking
* **nbsphinx** - For building notebook-based documentation

Notebook extras:

* **ipykernel** - For running Jupyter kernels
* **jupyter** - For interactive notebook development

Install optional extras from source (editable):

.. code-block:: bash

    pip install -e ".[dev]"
    pip install -e ".[notebook]"
    pip install -e ".[all]"

Install optional extras from PyPI:

.. code-block:: bash

    pip install "iguanas[dev]"
    pip install "iguanas[notebook]"
    pip install "iguanas[all]"