---
layout: post
title: "Javascript Env"
date: 2024-03-16
description: "A short description of this note"
categories: [jekyll, notes]
---

# JavaScript Development Environment Setup for macOS

This guide shows how I setup a JavaScript development environment on macOS focusing on Node.js development.

## 1. Install Homebrew (Package Manager)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, follow any instructions Homebrew gives you about adding it to your PATH.

## 2. Install Node Version Manager (nvm)

nvm is the JavaScript equivalent of pyenv, allowing you to manage multiple Node.js versions:

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# Add nvm to your shell configuration (for Zsh)
echo 'export NVM_DIR="$HOME/.nvm"' >> ~/.zshrc
echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"' >> ~/.zshrc
echo '[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"' >> ~/.zshrc

# Reload your shell configuration
source ~/.zshrc
```

## 3. Install Node.js Versions with nvm

```bash
# Install the latest LTS version of Node.js
nvm install --lts

# Install a specific version if needed
nvm install 18.17.1

# Set a default Node.js version
nvm alias default lts/*

# Verify installation
node --version
npm --version
```

## 4. Install Package Managers

npm comes with Node.js, but you might want to install Yarn or pnpm:

```bash
# Install Yarn
npm install -g yarn

# OR install pnpm
npm install -g pnpm
```

## 5. Install Useful Global Packages

```bash
# Install useful global developer tools
npm install -g typescript
npm install -g eslint
npm install -g prettier
npm install -g http-server
npm install -g vite
```

## 6. Install a Code Editor

```bash
# Install Visual Studio Code
brew install --cask visual-studio-code
```

Recommended VS Code extensions for JavaScript:
- ESLint
- Prettier
- JavaScript and TypeScript Nightly
- ES7+ React/Redux/React-Native snippets
- vscode-styled-components
- Import Cost
- Path Intellisense
- Vite (for Vite project integration)

## 7. Create a New JavaScript Project

### For a basic Node.js project:

```bash
# Create a project directory
mkdir my-node-project
cd my-node-project

# Initialize npm
npm init -y

# OR initialize with Yarn
yarn init -y
```

### For a front-end project with Vite:

```bash
# Create a new Vite project (will prompt for template)
npm create vite@latest my-project

# OR specify template directly
npm create vite@latest my-react-app -- --template react
npm create vite@latest my-vanilla-app -- --template vanilla
npm create vite@latest my-typescript-app -- --template vanilla-ts

# Navigate to project directory and install dependencies
cd my-project
npm install
```

## 8. Add Development Dependencies

```bash
# Navigate to your project
cd my-project

# Add ESLint and Prettier for code quality
npm install --save-dev eslint prettier eslint-config-prettier eslint-plugin-prettier

# Add TypeScript (if using)
npm install --save-dev typescript @types/node

# Add testing libraries (for Vite projects)
npm install --save-dev vitest @testing-library/react jsdom
```

## 9. Configure ESLint and Prettier

Create `.eslintrc.js`:

```js
module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:prettier/recommended',
  ],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  rules: {
    // Your custom rules
  },
};
```

Create `.prettierrc`:

```json
{
  "singleQuote": true,
  "trailingComma": "es5",
  "printWidth": 100,
  "tabWidth": 2,
  "semi": true
}
```

## 10. Version Control Setup

```bash
# Install Git (if not already installed)
brew install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Initialize Git repository
git init

# Create .gitignore
echo "node_modules/" > .gitignore
echo "dist/" >> .gitignore
echo "build/" >> .gitignore
echo ".env" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "coverage/" >> .gitignore
echo ".idea/" >> .gitignore
echo ".vscode/" >> .gitignore

# Initial commit
git add .
git commit -m "Initial project setup"
```

## Usage Tips

### Switching Node.js Versions with nvm

```bash
# List installed Node.js versions
nvm ls

# List available versions to install
nvm ls-remote

# Switch Node.js version
nvm use 16.20.0

# Set project-specific Node.js version
echo "16.20.0" > .nvmrc
nvm use  # Uses version from .nvmrc
```

### npm Commands

```bash
# Install dependencies
npm install

# Install a dependency
npm install package-name

# Install a dev dependency
npm install --save-dev package-name

# Run a script defined in package.json
npm run script-name

# Start development server (if configured)
npm start

# Build for production
npm run build

# Run tests (with Vitest for Vite projects)
npm run test
```

### Yarn Commands

```bash
# Install dependencies
yarn

# Install a dependency
yarn add package-name

# Install a dev dependency
yarn add --dev package-name

# Run a script defined in package.json
yarn script-name

# Start development server (if configured)
yarn start

# Build for production
yarn build

# Run tests
yarn test
```

This setup provides a solid foundation for JavaScript development on macOS with modern tools for dependency management and Node.js version control.