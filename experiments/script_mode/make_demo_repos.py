"""The two scratch script-mode repos of docs/RESULTS.md section 8, written by Claude for a first check.

    python experiments/script_mode/make_demo_repos.py <folder>

Creates <folder>/script-file-demo (argparse script, joblib.dump) and <folder>/script-mlflow-demo
(sys.argv script, mlflow.sklearn.log_model), each a git repo with deterministic data. Their gold
answers are tests/gold/script_file_demo.json and tests/gold/script_mlflow_demo.json.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

FILE_SCRIPT = '''import argparse
import joblib
import pandas as pd
from sklearn.linear_model import Ridge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="training CSV")
    p.add_argument("--out", required=True, help="where to write the model")
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()
    df = pd.read_csv(args.data)
    model = Ridge(alpha=args.alpha).fit(df.drop(columns=["price"]), df["price"])
    joblib.dump(model, args.out)


if __name__ == "__main__":
    main()
'''
MLFLOW_SCRIPT = '''import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    with mlflow.start_run():
        model = Ridge(alpha=alpha).fit(df.drop(columns=["price"]), df["price"])
        mlflow.log_metric("mse", mean_squared_error(df["price"], model.predict(df.drop(columns=["price"]))))
        mlflow.sklearn.log_model(model, "ridge")
'''


def rows(n, seed):
    g = np.random.default_rng(seed)
    a, b, grp = g.normal(size=n), g.normal(size=n), g.integers(0, 3, n)
    y = 3 * a - 2 * b + g.normal(scale=0.3, size=n) + 10
    return "a,b,g,price\n" + "".join(f"{x:.4f},{z:.4f},{k},{t:.3f}\n" for x, z, k, t in zip(a, b, grp, y))


def main(folder: Path) -> None:
    for name, script in (("script-file-demo", FILE_SCRIPT), ("script-mlflow-demo", MLFLOW_SCRIPT)):
        r = folder / name
        (r / "data/processed").mkdir(parents=True)
        (r / "scripts").mkdir()
        (r / "data/processed/train.csv").write_text(rows(500, 1))
        (r / "data/processed/test.csv").write_text(rows(150, 2))
        (r / "requirements.txt").write_text("scikit-learn==1.5.2\npandas==2.2.3\nnumpy==1.26.4\njoblib==1.4.2\n")
        (r / ".python-version").write_text("3.11\n")
        (r / "scripts/train.py").write_text(script)
        subprocess.run("git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init",
                       shell=True, cwd=r, check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
