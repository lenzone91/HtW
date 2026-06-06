Hello leWorld!
tets grrr



# HTW installation guide

## 1. Clone the high-level repository

```powershell
git clone <HTW_REPO_URL>
cd HtW
```

## 2. Initialize submodules

```powershell
git submodule update --init --recursive
```

If the submodule already exists but is outdated:

```powershell
git submodule update --remote --recursive
```

## 3. Create a Python 3.12 environment

eb_jepa requires Python 3.12.*, so use Python 3.12.

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Check:

```powershell
python --version
```

Expected:

Python 3.12.x

## 4. Install the high-level HTW package safely

Install HtW without dependencies:

```powershell
python -m pip install -e . --no-deps
```

Reason: HtW is the high-level project wrapper and should not override dependency versions.

## 5. Install eb_jepa as the dependency provider
```powershell
cd eb_jepa
python -m pip install -e .
cd ..
```

Reason: eb_jepa owns the strict dependency constraints, including torch.

## 6. Verify the installation

```powershell
python -c "import Octave; print('HtW OK:', Octave.__file__)"
python -c "from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig; print('eb_jepa OK')"
```

Expected:

HtW OK: ...
eb_jepa OK

## 7. Optional: install as Jupyter kernel

```powershell
python -m pip install ipykernel
python -m ipykernel install --user --name htw312 --display-name "Python (.venv312 HTW)"
```

Then select:

Python (.venv312 HTW)

inside Jupyter or VS Code.

Important dependency rule

Do not install HtW with dependencies unless explicitly needed.

Use:

```powershell
python -m pip install -e . --no-deps
```

not:

```powershell
python -m pip install -e .
```

The root HtW/pyproject.toml should keep:

dependencies = []

so that dependency resolution is controlled by eb_jepa.