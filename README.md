# ---- Data (don't commit the Kaggle dataset) ----
data/
!data/.gitkeep

# ---- Models / experiment artifacts ----
models/
!models/.gitkeep
mlruns/
mlartifacts/
wandb/
*.pkl
*.joblib
*.pt
*.ckpt

# ---- Python ----
__pycache__/
*.py[cod]
.venv/
venv/
env/
.ipynb_checkpoints/

# ---- Kaggle credentials (NEVER commit) ----
kaggle.json

# ---- OS / editors ----
.DS_Store
.idea/
.vscode/