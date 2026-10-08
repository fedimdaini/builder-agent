"""Static knowledge: import name -> pip distribution, and framework categories.

These are facts about the Python ecosystem, not team decisions, so they live here
rather than in contracts.yaml.
"""

IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
    "attr": "attrs",
    "Crypto": "pycryptodome",
    "docx": "python-docx",
    "serial": "pyserial",
    "pytorch_lightning": "pytorch-lightning",
    "sentence_transformers": "sentence-transformers",
    "google.protobuf": "protobuf",
    "Levenshtein": "python-levenshtein",
    "multipart": "python-multipart",
    "pandas_profiling": "pandas-profiling",
    "umap": "umap-learn",
    "tensorflow_hub": "tensorflow-hub",
    "IPython": "ipython",
}

# pip name -> category. Detected from imports and from declared dependencies.
FRAMEWORKS = {
    # ml
    "scikit-learn": "ml", "xgboost": "ml", "lightgbm": "ml", "catboost": "ml",
    "torch": "ml", "tensorflow": "ml", "keras": "ml", "jax": "ml", "flax": "ml",
    "transformers": "ml", "statsmodels": "ml", "prophet": "ml",
    "pytorch-lightning": "ml", "lightning": "ml", "optuna": "ml",
    "sentence-transformers": "ml",
    # serving
    "flask": "serving", "fastapi": "serving", "gunicorn": "serving", "uvicorn": "serving",
    "streamlit": "serving", "gradio": "serving", "bentoml": "serving",
    # tracking
    "mlflow": "tracking", "wandb": "tracking", "dvc": "tracking", "neptune": "tracking",
    # data
    "pandas": "data", "numpy": "data", "polars": "data", "pyspark": "data",
    "dask": "data", "pyarrow": "data", "geopandas": "data", "pandera": "data",
}
