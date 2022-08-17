

#### Install package via poetry

The following creates a venv, and installs all dependencies but matlab

'''
poetry install
'''

then start the created venv:
'''
source ./.venv/bin/activate
'''


#### Install matlab

Under Linux:
you need to change the default path during the matlab installation to a path in your user home dir:
"/home/$USER/MATLAB/R2021b"

#### Verify Your Configuration

Before you install, verify your Python and MATLAB configurations.

    Check that your system has a supported version of Python and MATLAB R2014b or later. For more information, see Versions of Python Compatible with MATLAB Products by Release .
    
    To check that Python is installed on your system, run Python at the operating system prompt.
    
    Add the folder that contains the Python interpreter to your path, if it is not already there.
    
    Find the path to the MATLAB folder. Start MATLAB and type matlabroot in the command window. Copy the path returned by matlabroot.

#### Install the Engine API

To install the engine API, choose one of the following. You must call this python install command in the specified folder.

    At a Windows operating system prompt (you might need administrator privileges to execute these commands) —
    
    '''
    cd "$matlabroot\extern\engines\python"
    python setup.py install
    '''
    
    At a macOS or Linux operating system prompt (you might need administrator privileges to execute these commands) —
    
    '''
    cd "$matlabroot/extern/engines/python"
    python setup.py install
    '''
    
    or
    
    '''
    cd "$matlabroot/extern/engines/python"
    pip install -e ./
    '''

build the package:
    '''
    cd "$matlabroot/extern/engines/python"
    python setup.py build
    '''

add a pyproject.toml to the "$matlabroot/extern/engines/python/build/lib", with following content:

'''
[tool.poetry]
name = "matlab"
version = "1.0.0"
description = "pyproject pakage for matlab engin"
authors = ["bela127 <bhb127@outlook.de>"]
license = "MIT"

[tool.poetry.dependencies]
python = ">=3.8,<3.10"

[tool.poetry.dev-dependencies]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

'''

install all dependencies with poetry:
'''
poetry install
'''


### Install new packages after matlab engin installation
The matlab engin brakes poetry version parser, this is why every time a new package needs installation one has to uninstall *matlabengineforpython* install the package and reinstall it like described above in **Install the Engine API**.

To uninstall just run:
'''
pip remove matlabengineforpython
'''