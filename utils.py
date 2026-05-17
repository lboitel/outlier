import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_tabular(file_obj, filename, use_dask=False):
    """Read a tabular file into a pandas DataFrame.

    If `use_dask` is True and `file_obj` is a filesystem path (str/Path), attempt
    to use dask for lazy reading. For uploaded file-like objects, Dask-read is
    not used because dask typically needs a path or fsspec URL.
    """
    name = filename.lower()
    # If use_dask and file is a path string, attempt dask
    try:
        if use_dask and isinstance(file_obj, (str,)):
            import dask.dataframe as dd
            if name.endswith('.parquet'):
                ddf = dd.read_parquet(file_obj)
            else:
                ddf = dd.read_csv(file_obj)
            return ddf.compute()
    except Exception:
        # fallback to pandas reading
        pass

    if name.endswith('.parquet'):
        return pd.read_parquet(file_obj)
    if name.endswith('.csv') or name.endswith('.txt'):
        return pd.read_csv(file_obj)
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(file_obj)
    # Fallback try CSV
    return pd.read_csv(file_obj)


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='results')
    bio.seek(0)
    return bio.read()

