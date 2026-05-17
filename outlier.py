import numpy as np
import pandas as pd
from scipy import stats


def detect_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def iqr_scores(series: pd.Series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return pd.Series(0.0, index=series.index)
    score = ((series - q3).clip(lower=0) / iqr) + ((q1 - series).clip(lower=0) / iqr)
    return score.fillna(0.0)


def zscore_scores(series: pd.Series):
    if series.std(ddof=0) == 0 or pd.isna(series.std(ddof=0)):
        return pd.Series(0.0, index=series.index)
    return pd.Series(np.abs(stats.zscore(series.fillna(series.mean()))), index=series.index).fillna(0.0)


def categorical_freq_score(series: pd.Series):
    # rare categories get higher score
    freq = series.fillna('__MISSING__').value_counts(normalize=True)
    inv = freq.reindex(series.fillna('__MISSING__')).fillna(0).apply(lambda x: 1.0 - x)
    return inv


def normalize_df(df_scores: pd.DataFrame):
    # normalize each column to 0..1
    norm = df_scores.copy()
    for c in norm.columns:
        col = norm[c]
        if col.max() == col.min():
            norm[c] = 0.0
        else:
            norm[c] = (col - col.min()) / (col.max() - col.min())
    return norm


def compute_outlier_scores(df: pd.DataFrame, method='iqr', numeric_cols=None, categorical_cols=None, weight_numeric=1.0, weight_categorical=0.5):
    if numeric_cols is None:
        numeric_cols = detect_numeric_columns(df)
    if categorical_cols is None:
        categorical_cols = detect_categorical_columns(df)

    scores = pd.DataFrame(index=df.index)

    # numeric
    for c in numeric_cols:
        try:
            if method == 'iqr':
                scores[c] = iqr_scores(df[c].astype(float))
            else:
                scores[c] = zscore_scores(df[c].astype(float))
        except Exception:
            scores[c] = 0.0

    # categorical
    for c in categorical_cols:
        try:
            scores[c] = categorical_freq_score(df[c])
        except Exception:
            scores[c] = 0.0

    if scores.shape[1] == 0:
        scores['combined'] = 0.0
        return scores

    norm = normalize_df(scores)

    # combine with simple weighting: numeric columns together and categorical together
    num_mask = [c for c in norm.columns if c in numeric_cols]
    cat_mask = [c for c in norm.columns if c in categorical_cols]

    numeric_comb = norm[num_mask].sum(axis=1) if len(num_mask) > 0 else 0.0
    categorical_comb = norm[cat_mask].sum(axis=1) if len(cat_mask) > 0 else 0.0

    combined = weight_numeric * numeric_comb + weight_categorical * categorical_comb
    # normalize combined to 0..1
    if isinstance(combined, (float, int)):
        combined = pd.Series(float(combined), index=df.index)
    if combined.max() == combined.min():
        combined = pd.Series(0.0, index=df.index)
    else:
        combined = (combined - combined.min()) / (combined.max() - combined.min())

    norm['combined'] = combined
    return norm


def label_outliers(df: pd.DataFrame, score_series: pd.Series = None, threshold: float = None, top_n: int = None):
    # Return boolean Series marking outliers.
    if score_series is None and '_outlier_score' in df.columns:
        score_series = df['_outlier_score']
    if score_series is None:
        raise ValueError('score_series required or `_outlier_score` column present')
    if top_n is not None:
        thr = score_series.sort_values(ascending=False).iloc[top_n-1] if len(score_series) >= top_n and top_n > 0 else score_series.max() + 1
        return score_series >= thr
    if threshold is not None:
        return score_series >= threshold
    # default: mark rows above 95th percentile
    thr = score_series.quantile(0.95)
    return score_series >= thr

