# Outlier Explorer

Interactive outlier analysis tool built with Streamlit.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

Features

- Upload CSV / Excel / Parquet files
- Automatic detection of numeric columns
- IQR / z-score based scoring per column, combined into a global outlier score
- Interactive filters, ranking, and preview with highlighted outliers
- Export filtered results to Excel or Parquet

Notes

- For very large datasets consider enabling Dask (toggle in UI) and installing dask accordingly.
- The sample data `sample_data.csv` is included for quick testing.
