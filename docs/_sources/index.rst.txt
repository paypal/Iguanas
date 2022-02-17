Welcome to the Iguanas documentation!
=====================================

.. image:: _static/iguanas_logo.png
   :width: 400px
   :height: 269px 
   :align: center

What is Iguanas?
----------------

Iguanas is a fast, flexible and modular Python package for:

* Generating new rules using a labelled dataset.
* Optimising existing rules using a labelled or unlabelled dataset.
* Combining rule sets and removing/filtering those which are unnecessary.
* Generating rule scores based on their performance.

It aims to help streamline the process for developing a deployment-ready rules-based system (RBS) for **binary classification use cases**.

What are rules and rule-based systems (RBS)?
--------------------------------------------

A **rule** is a set of conditions which, if met, trigger a certain response.

An example of a rule is a condition that captures a particular type of fraudulent behaviour for an e-commerce company:

*If the number of transactions made from a given email address in the past 4 hours is greater than 10, reject the transaction.*

An **RBS** leverages a number of these rules to provide a certain outcome.

For example, an e-commerce company might employ an RBS to *accept*, *reject* and *review* its transactions.

The advantages and disadvantages of an RBS
------------------------------------------

As with any approach, there are both advantages and disadvantages related to Rules-Based Systems:

**Advantages**

* Rules are intuitive, so the outcome given by the RBS is easy to understand.
* Rules-Based Systems are flexible, since rules can be quickly added to address new behaviour.
* Rules can be built using domain knowledge, which may capture behaviour an ML model would have missed.

**Disadvantages**

* If you don't have domain knowledge, you need to use a data-guided approach when generating rules, which can be challenging and time consuming.
* It's difficult to tweak existing rules to address new trends.
* Knowing which rules to include in the RBS to maximise its performance is very difficult.

The solution – Iguanas!
-----------------------

Iguanas addresses the disadvantages of an RBS for binary classification problems:

* Iguanas only requires a historic data set to generate rules - similar to the requirements of an ML model.
* Iguanas utilises an API that is familiar to most data scientists - Sklearn's fit/transform/predict methodology.
* Iguanas' Rule Optimisation module allows users to update the thresholds of current rules based on new trends.
* Iguanas' **LinearPipeline**, **ParallelPipeline** and **BayesSearchCV** classes allow users to identify the combination of rules that gives the best RBS performance.
* Iguanas has many other modules that streamline the process of generating an RBS.

Getting started
---------------

The :ref:`installation.index` section provides instructions on installing Iguanas on your system. Once installed, take a look at the :ref:`examples.index` section for how Iguanas can be used. The :ref:`api.index` section also provides documentation for the classes and methods in each module.

.. toctree::
    :maxdepth: 2
    :hidden:

    install/index
    user_guide/index
    api/index
    examples/index
    about_the_project/index