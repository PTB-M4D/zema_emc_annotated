# Installation

zema_emc_annotated is packaged with [poetry](https://python-poetry.org/) and thus 
supports automatic creation of a virtual environment and installation of all 
required dependencies with known-to-work versions.

All there is to do is to ensure there is an up-to-date version of poetry itself 
installed in your path. The commands differ slightly between Windows and Mac & Linux.

## Create a `venv` Python environment on Windows

In your Windows PowerShell execute the following to set up poetry in your user space:

```shell
PS C:> py -3 -m pip install --user poetry
Collecting poetry
[...]
Successfully installed cachecontrol-0.12.11 cleo-2.0.1 crashtest-0.4.1 distlib-0.3.6 dulwich-0.20.50 filelock-3.8.2 importlib-metadata-5.2.0 jaraco.classes-3.2.3 jsonschema-4.17.3 keyring-23.13.1 msgpack-1.0.4 pkginfo-1.9.2 platformdirs-2.6.0 poetry-1.3.1 poetry-core-1.4.0 poetry-plugin-export-1.2.0 rapidfuzz-2.13.7 shellingham-1.5.0 tomlkit-0.11.6 trove-classifiers-2022.12.22 virtualenv-20.17.1
```

Proceed to [the next step
](#install-zema_emc_annotated-via-pip).

## Create a `venv` Python environment on Mac & Linux

In your terminal execute the following to set up poetry in your user space:

```shell
$ python3 -m pip install --user poetry
Collecting poetry
[...]
Successfully installed cachecontrol-0.12.11 cleo-2.0.1 crashtest-0.4.1 distlib-0.3.6 dulwich-0.20.50 filelock-3.8.2 importlib-metadata-5.2.0 jaraco.classes-3.2.3 jsonschema-4.17.3 keyring-23.13.1 msgpack-1.0.4 pkginfo-1.9.2 platformdirs-2.6.0 poetry-1.3.1 poetry-core-1.4.0 poetry-plugin-export-1.2.0 rapidfuzz-2.13.7 shellingham-1.5.0 tomlkit-0.11.6 trove-classifiers-2022.12.22 virtualenv-20.17.1
```

Proceed to [the next step
](#install-zema_emc_annotated-via-pip).

## Install zema_emc_annotated via `pip`

Once you installed poetry, you can install zema_emc_annotated from your repository 
root via:

```shell
$ poetry install
Creating virtualenv zema-emc-annotated-j82TMThr-py3.10 in ~/.cache/pypoetry/virtualenvs
Installing dependencies from lock file

Package operations: 11 installs, 0 updates, 0 removals

  • Installing [...]
  • Installing pooch (1.6.0)
  • Installing tqdm (4.64.1)

Installing the current project: zema_emc_annotated (0.0.0)
```

That's it!

## Optional Jupyter Notebook dependencies

If you are familiar with Jupyter Notebooks, you find some examples in the _src/examples_
subfolder of the source code repository. To execute these you need additional 
dependencies which you get by appending `[examples]` to
zema_emc_annotated in the above installation command, 
e.g.

```shell
$ poetry install --with examples
Installing dependencies from lock file

Package operations: 75 installs, 0 updates, 0 removals
[...]
Installing the current project: zema_emc_annotated (0.0.0)
```

Then to start the Jupyter notebook server execute

```shell
$ poetry run jupyter notebook
[I 19:24:50.906 NotebookApp] [jupyter_nbextensions_configurator] enabled 0.6.1
[...]
```