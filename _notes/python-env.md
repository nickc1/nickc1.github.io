---
layout: post
title: "Python Dev Environment"
date: 2025-03-16
categories: python
---


This guide shows how I setup a Python development environment on macOS using Homebrew, pyenv, pipx, and Poetry.

### 1. Install Homebrew (Package Manager)

[Homebrew](https://brew.sh) simplifies the process of setting up a Python development environment on macOS by providing a straightforward way to install and manage tools and dependencies not included with the operating system. It saves you time by handling the installation of essential components like pyenv and pipx through simple commands, rather than dealing with manual downloads and configurations.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, follow any instructions Homebrew gives you about adding it to your PATH.

### 2. Install pyenv for Python Version Management

Pyenv allows you to easily install and switch between multiple Python versions on your Mac, giving you the flexibility to work with different Python environments for various projects. It ensures compatibility with project requirements and lets you test your code across different Python versions without the headache of managing installations manually.

```bash
# Install pyenv using Homebrew
brew install pyenv

# Add pyenv to your shell configuration (for Zsh)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc
```

### 3. Install Python Versions with pyenv

The Python official website ([python.org]()) is the best place to check for Python versions, as it lists all available releases including stable versions and their support status. You can also run pyenv install --list directly in your terminal to see all Python versions that pyenv can install, which helps you quickly identify current stable releases and legacy versions without leaving the command line.

```bash
# Install Python versions (change versions as needed)
pyenv install 3.11.5
pyenv install 3.10.13

# Set a global Python version
pyenv global 3.11.5

# Verify installation
python --version
```

### 4. Install pipx

Pipx enables you to install and run Python applications in isolated environments, keeping your global Python installation clean while still making the tools available system-wide. It's perfect for installing command-line tools like [Poetry](https://python-poetry.org) and [cookiecutter](https://github.com/cookiecutter/cookiecutter) without worrying about dependency conflicts between different applications.

```bash
# Install pipx for isolated application installation
brew install pipx
pipx ensurepath

# Install Poetry using pipx
pipx install poetry

# Verify Poetry installation
poetry --version
```

### 5. Install Poetry

```bash
# Install Poetry using pipx
pipx install poetry

# Verify Poetry installation
poetry --version
```


### 5. Configure Poetry (Optional)

```bash
# Configure Poetry to create virtual environments inside project directories
poetry config virtualenvs.in-project true
```

### 6. Install a Code Editor

```bash
# Install Visual Studio Code
brew install --cask visual-studio-code
```

Recommended VS Code extensions:
- Python
- Pylance
- Python Indent
- autoDocstring
- Python Test Explorer

### 7. Create a New Project with Poetry

```bash
# Navigate to your projects directory
cd ~/Projects

# Create a new directory for your project
mkdir my_project
cd my_project

# Optional: Specify Python version for this project
pyenv local 3.10.13  # If you want to use a different version than your global

# Initialize a new Poetry project
poetry init  # Interactive initialization
# OR
poetry new my_project  # Creates a new project with predefined structure
```

### 8. Add Dependencies with Poetry

```bash
# Add production dependencies
poetry add requests pandas matplotlib

# Add development dependencies
poetry add --group dev pytest pytest-cov black mypy flake8

# Activate the virtual environment
poetry shell
```

### 9. Version Control Setup

```bash
# Install Git
brew install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize Git repository
git init
echo "__pycache__/" > .gitignore
echo "*.pyc" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo ".coverage" >> .gitignore
echo "htmlcov/" >> .gitignore
echo ".venv/" >> .gitignore

# Initial commit
git add .
git commit -m "Initial project setup"
```



### Usage Tips

#### Switching Python Versions

```bash
# List installed Python versions
pyenv versions

# Switch global Python version
pyenv global 3.10.13

# Set project-specific Python version
cd my_project
pyenv local 3.9.13
```

#### Poetry Commands

```bash
# Create a new project
poetry new project-name

# Install dependencies
poetry install

# Add a dependency
poetry add package-name

# Update dependencies
poetry update

# Run a command within the virtual environment
poetry run python script.py

# Activate the virtual environment
poetry shell

# Build the package
poetry build

# Publish to PyPI
poetry publish
```

This setup provides a solid foundation for Python development on macOS with modern tools for dependency management and Python version control.

