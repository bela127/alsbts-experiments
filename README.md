## Installation:

### Recommended tools for python management
We recommend the following tools which do python version management and dependency management for you.

install pyenv dependencies:

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
```

install pyenv:

```bash
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

add it to bash, by adding the following to the bottom of the file  `~/.bashrc`, replace `USER`:

```bash
export PATH="/home/USER/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

install python version:

```bash
pyenv install 3.9.6
```

install poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

add it to bash, by adding the following to the bottom of the file  `~/.bashrc`:

```bash
export PATH="/home/USER/.local/bin:$PATH"
```

set poetry to use pyenv:

```bash
poetry config virtualenvs.prefer-active-python true
```

And make sure venv are created inside a project:

```bash
poetry config virtualenvs.in-project true
```

### Project installation
Clone the git repo, open a terminal inside the repo, install project dependencies:

```bash
poetry install
```

wait for all dependencies to install.

YOU HAVE FINISHED INSTALLATION

## Running Experiments:
Each file in `./alsbts/experiments` is one experiment, just execute them using the installed python environment.
You can use `./run.py` to run them all.

To produce the experiment result plots run the scripts in `./post_experiment_computation` and `./component_visualsation`.
For beautiful plots you may require a local latex installation, if this is not possible you should change the line `"text.usetex": True,` in `./post_experiment_computation/utils.py` to `FALSE`.