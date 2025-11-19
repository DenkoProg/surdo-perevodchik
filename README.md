# Python Repo Template

## ⚙️ Installation

### 🔧 Set Up the Python Environment

#### 1. Clone the repository

```bash
git clone REPO_NAME
cd REPO_NAME
```

#### 2. Install `uv` — A fast Python package manager

📖 [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)

#### 3. Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

Alternatively, you can use the predefined Makefile command:

```bash
make install
```
This will set up the virtual environment, install dependencies, and configure pre-commit hooks automatically.

#### 4. Install dependencies (choose ONE path)

##### 4.1 Reproduce exact versions (use uv.lock)

```bash
# Usage environment (pinned, reproducible)
uv sync --locked

# Development environment (pinned + dev extras)
uv sync --locked --extra dev
```

- Uses the checked-in uv.lock exactly; no re-resolution.
- Ideal for CI and deterministic installs.

##### 4.2 Resolve fresh compatible versions (from pyproject.toml)

```bash
# Usage environment (resolve now and write/update uv.lock)
uv sync

# Development environment (resolve + dev extras)
uv sync --extra dev
```

- Resolves to the latest compatible versions and writes/updates uv.lock.
- Ideal when you want newer dependency versions locally.

##### 4.3 pip-style installs (do NOT enforce the lockfile)

```bash
# Usage only
uv pip install .

# Development (editable) install
uv pip install -e .[dev]
```

> These behave like regular pip installs and ignore uv.lock.
