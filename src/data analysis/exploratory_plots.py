"""
Exploratory plotting helpers for Alzheimer disease dataset.

Place your plotting functions here. This script:
- loads the raw CSV path from config.RAW_DATA_PATH
- provides helpers to show dataset info and sample plots
- is intentionally minimal so you can add your own plots

Usage:
    python "data analysis/exploratory_plots.py"

"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

# Make sure project root is on sys.path so local imports (like `config`) work
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_PATH

DATA_PATH = Path(RAW_DATA_PATH)

# Control flags (can be toggled via CLI). Default: do not show interactive windows.
SHOW_PLOTS = False
SAVE_PLOTS = True
# When True, suppress informational console prints about saved files
QUIET = False

# Layout constants (centralized sizing to make tuning easier)
# width in inches per column for combined grids
COLUMN_WIDTH = 4.5
# width for single plot figures
SINGLE_PLOT_WIDTH = 11

# Base output folder for all generated artifacts (inside the data analysis folder)
OUTPUT_DIR = PROJECT_ROOT / 'data analysis' / 'output_exploratory_plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns that uniquely identify patients / doctors and should never be plotted
IDENTIFIER_COLS = ('PatientID', 'DoctorInCharge')

# Human-friendly display names for columns (used in titles/labels)
DISPLAY_NAME_MAP = {
    'PatientID': 'Patient ID',
    'Age': 'Age',
    'Gender': 'Gender',
    'Ethnicity': 'Ethnicity',
    'EducationLevel': 'Education Level',
    'BMI': 'BMI',
    'Smoking': 'Smoking',
    'AlcoholConsumption': 'Alcohol Consumption',
    'PhysicalActivity': 'Physical Activity',
    'DietQuality': 'Diet Quality',
    'SleepQuality': 'Sleep Quality',
    'FamilyHistoryAlzheimers': "Family History of Alzheimer's",
    'CardiovascularDisease': 'Cardiovascular Disease',
    'Diabetes': 'Diabetes',
    'Depression': 'Depression',
    'HeadInjury': 'Head Injury',
    'Hypertension': 'Hypertension',
    'SystolicBP': 'Systolic BP',
    'DiastolicBP': 'Diastolic BP',
    'CholesterolTotal': 'Cholesterol Total',
    'CholesterolLDL': 'Cholesterol LDL',
    'CholesterolHDL': 'Cholesterol HDL',
    'CholesterolTriglycerides': 'Cholesterol Triglycerides',
    'MMSE': 'MMSE',
    'FunctionalAssessment': 'Functional Assessment',
    'MemoryComplaints': 'Memory Complaints',
    'BehavioralProblems': 'Behavioral Problems',
    'ADL': 'ADL',
    'Confusion': 'Confusion',
    'Disorientation': 'Disorientation',
    'PersonalityChanges': 'Personality Changes',
    'DifficultyCompletingTasks': 'Difficulty Completing Tasks',
    'Forgetfulness': 'Forgetfulness',
    'Diagnosis': 'Diagnosis',
    'DoctorInCharge': 'Doctor In Charge'
}
from config import RAW_DATA_PATH

DATA_PATH = Path(RAW_DATA_PATH)

# Control flags (can be toggled via CLI). Default: do not show interactive windows.
SHOW_PLOTS = False
SAVE_PLOTS = True

# Layout constants (centralized sizing to make tuning easier)
# width in inches per column for combined grids
COLUMN_WIDTH = 4.5
# width for single plot figures
SINGLE_PLOT_WIDTH = 11

# Base output folder for all generated artifacts (inside the data analysis folder)
OUTPUT_DIR = PROJECT_ROOT / 'data analysis' / 'output_exploratory_plots'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns that uniquely identify patients / doctors and should never be plotted
IDENTIFIER_COLS = ('PatientID', 'DoctorInCharge')


def load_data(sample_rows: int | None = 1000) -> pd.DataFrame:
    """Load the raw dataset. By default, reads first `sample_rows` rows for quick plotting.

    Set sample_rows=None to load the full dataset.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    if sample_rows is None:
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, nrows=sample_rows)

    # Drop identifier columns for safety (PatientID, DoctorInCharge)
    df = _drop_identifier_columns(df)
    return df

def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with identifier columns removed (no-op if not present)."""
    if df is None:
        return df
    # use errors='ignore' to be safe if columns don't exist
    return df.drop(columns=[c for c in IDENTIFIER_COLS if c in df.columns], errors='ignore')


def show_basic_info(df: pd.DataFrame, to_console: bool = True, out_path: Path | str | None = None) -> None:
    """Print basic info to console and/or save to a text file.

    Args:
        df: dataframe to summarize
        to_console: if True, print the formatted report to stdout
        out_path: optional path to write the same formatted report to (uses save_basic_info_to_file)
    """
    content = format_basic_info(df)
    if to_console:
        print(content)
    if out_path is not None:
        save_basic_info_to_file(df, out_path=out_path)

def format_basic_info(df: pd.DataFrame) -> str:
    """Return a nicely formatted string with dataset basic information."""
    lines = []
    lines.append('Dataset basic information')
    lines.append('=' * 40)
    lines.append(f'Shape: {df.shape[0]} rows x {df.shape[1]} columns')
    lines.append('')
    lines.append('Columns:')
    for col in df.columns:
        lines.append(f' - {col}')
    lines.append('')
    lines.append('Column types and non-null counts:')
    dtypes = df.dtypes
    nonnull = df.count()
    max_col_len = max(len(c) for c in df.columns)
    lines.append(f"{ 'Column'.ljust(max_col_len) }  Type        Non-Null Count")
    lines.append('-' * (max_col_len + 30))
    for col in df.columns:
        lines.append(f"{col.ljust(max_col_len)}  {str(dtypes[col]).ljust(10)}  {nonnull[col]}")
    lines.append('')
    lines.append('Descriptive statistics for numeric columns:')
    lines.append('-' * 40)
    try:
        stats = df.describe()
        lines.append(stats.to_string())
    except Exception:
        lines.append('(could not compute describe())')
    return "\n".join(lines)

def save_basic_info_to_file(df: pd.DataFrame, out_path: Path | str | None = None) -> Path:
    """Save basic dataset info to a text file. Returns the path written to."""
    if out_path is None:
        out_path = OUTPUT_DIR / 'data_analysis_report.txt'
    out_path = Path(out_path)
    # create parent dir if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = format_basic_info(df)
    out_path.write_text(content, encoding='utf-8')
    return out_path


def save_figure(fig: plt.Figure, plot_name: str, subfolder: str | None = None) -> Path:
    """Save a matplotlib figure into the data analysis folder.

    If subfolder is provided, the file will be written to PROJECT_ROOT/'data analysis'/subfolder/<plot_name>.png
    Otherwise it will be saved to PROJECT_ROOT/'data analysis'/plot_name/plot_name.png (legacy behavior).
    Returns the Path written to.
    """
    safe_name = str(plot_name).replace(' ', '_')
    # If a specific subfolder is requested, place the file there.
    # Otherwise write directly into OUTPUT_DIR to avoid creating a nested
    # folder named after the file (which led to duplicated names).
    if subfolder:
        folder = OUTPUT_DIR / str(subfolder)
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{safe_name}.png"
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"{safe_name}.png"
    try:
        fig.savefig(out, bbox_inches='tight', dpi=150)
    except Exception:
        # fallback to saving via pyplot
        plt.savefig(out, bbox_inches='tight', dpi=150)
    return out


def plot_distributions(attributes: list[str], df: pd.DataFrame, combine: bool = False, per_row: int = 5) -> list[Path]:
    """Create and save one histogram per attribute into data analysis/distribution/.

    If an attribute is missing from df, read the raw CSV to try to access it (so PatientID/DoctorInCharge can be included).
    If combine=True, create a single grid figure with all attributes and save it to
    data analysis/distribution/distribution_all.png. In that mode the function
    returns a list with the written Path (or empty list if saving disabled).
    Otherwise it returns a list of Paths written for individual attribute figures.
    """
    written = []
    # filter out identifiers from requested attributes early
    attributes = [a for a in attributes if a not in IDENTIFIER_COLS]
    if combine:
        # prepare source df (use raw if attributes not present)
        src = df
        missing = [a for a in attributes if a not in src.columns]
        if missing:
            raw_df = pd.read_csv(DATA_PATH)
            src = raw_df

        # ensure identifiers are removed
        src = _drop_identifier_columns(src)

        # Exclude explicit categorical/binary columns from the combined grid.
        # These columns are either identifiers, booleans, or flags (0/1) and
        # look cluttered when shown in a big histogram grid.
        EXCLUDE_FROM_COMBINED = {
            'Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
            'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
            'PersonalityChanges', 'Disorientation', 'Confusion', 'Diagnosis',
            'Forgetfulness', 'MemoryComplaints', 'DifficultyCompletingTasks',
            'BehavioralProblems'
        }
        # Start with attributes that exist in the source and are not in explicit exclude set
        attrs_present = [a for a in attributes if a in src.columns and a not in EXCLUDE_FROM_COMBINED]
        # Further drop obvious binary flags (values only 0/1 or booleans)
        filtered_attrs = []
        for a in attrs_present:
            try:
                vals = pd.unique(src[a].dropna())
            except Exception:
                vals = []
            # if all non-null values are a subset of {0,1,True,False} skip
            if len(vals) > 0 and set(vals).issubset({0, 1, True, False}):
                continue
            filtered_attrs.append(a)
        attributes = filtered_attrs

        n = len([a for a in attributes if a in src.columns])
        if n == 0:
            return []
        rows = math.ceil(n / per_row)
        # size per subplot: width per column from COLUMN_WIDTH, height ~2.5in
        fig, axes = plt.subplots(rows, per_row, figsize=(per_row * COLUMN_WIDTH, rows * 2.5))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        idx = 0
        for attr in attributes:
            if attr not in src.columns:
                continue
            ax = axes_flat[idx]
            try:
                sns.histplot(src[attr].dropna(), bins=20, kde=True, color='skyblue', ax=ax)
            except Exception:
                vals = src[attr].value_counts()
                sns.barplot(x=vals.values, y=vals.index, ax=ax)
            ax.set_title(attr, fontsize=8, pad=6)
            # remove small x-axis labels under each subplot for a cleaner grid
            ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.set_ylabel('')
            idx += 1

        # hide any unused axes
        for j in range(idx, len(axes_flat)):
            try:
                axes_flat[j].set_visible(False)
            except Exception:
                pass

        plt.tight_layout()
        if SAVE_PLOTS:
            out = save_figure(fig, 'distribution_all', subfolder='distribution')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)
        return written
    # try to load full raw if needed
    raw_df = None
    for attr in attributes:
        # skip identifier attributes explicitly
        if attr in IDENTIFIER_COLS:
            continue
        if attr in df.columns:
            src = df
        else:
            if raw_df is None:
                raw_df = pd.read_csv(DATA_PATH)
            src = raw_df
        if attr not in src.columns:
            # skip silently if attribute doesn't exist anywhere
            continue
        # drop identifier columns from the source to be safe
        src = _drop_identifier_columns(src)
        # slightly larger figure for better readability
        fig, ax = plt.subplots(figsize=(SINGLE_PLOT_WIDTH, 5))
        try:
            # match style of plot_age_distribution: use seaborn histplot with KDE
            sns.histplot(src[attr].dropna(), kde=True, ax=ax)
        except Exception:
            # if histogram fails (e.g., non-numeric), attempt value_counts barplot
            vals = src[attr].value_counts()
            sns.barplot(x=vals.values, y=vals.index, ax=ax)
        # place the title above the subplot and add a small pad
        ax.set_title(f'{attr} distribution', pad=6)
        # show x-axis tick labels for saved individual distribution figures
        display = DISPLAY_NAME_MAP.get(attr, attr)
        ax.set_xlabel(display)
        ax.tick_params(axis='x', labelsize=8)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
        ax.set_ylabel('')
        if SAVE_PLOTS:
            out = save_figure(fig, attr, subfolder='distribution')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)
    return written


def plot_missing_values(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(SINGLE_PLOT_WIDTH, 4))
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    sns.barplot(x=missing.values, y=missing.index, ax=ax)
    ax.set_xlabel('% missing')
    ax.set_title('Missing values by column')
    if SAVE_PLOTS:
        save_figure(fig, 'missing_values')
    if SHOW_PLOTS:
        fig.show()
    return fig


def plot_duplicate_values(df: pd.DataFrame) -> plt.Figure:
    """Plot counts of duplicate values per column.

    For each column this computes: duplicate_count = total_rows - nunique(dropna=True).
    Columns with zero duplicate entries are omitted from the plot.
    """
    # remove identifier columns from consideration
    df = _drop_identifier_columns(df)
    total = len(df)
    # compute duplicate counts per column (number of non-unique entries)
    dup_counts = {}
    for c in df.columns:
        try:
            nunq = df[c].nunique(dropna=True)
        except Exception:
            nunq = 0
        dup = max(0, total - nunq)
        dup_counts[c] = dup

    dup_series = pd.Series(dup_counts)
    dup_series = dup_series[dup_series > 0].sort_values(ascending=False)

    # choose a height proportional to number of rows so labels don't overlap
    n = len(dup_series)
    fig_h = max(3, 0.35 * n)
    fig_w = max(SINGLE_PLOT_WIDTH, SINGLE_PLOT_WIDTH * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if dup_series.empty:
        ax.text(0.5, 0.5, 'No duplicate values found', ha='center', va='center', fontsize=12)
        ax.set_axis_off()
    else:
        # horizontal bars, largest first
        # use a matplotlib horizontal bar chart with a Blues colormap mapped to counts
        vals = dup_series.values.astype(float)
        cmap = plt.cm.Blues
        if vals.max() > vals.min():
            norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
        else:
            norm_vals = np.full_like(vals, 0.5)
        # shift and scale so very small values are visible (avoid pure-white)
        colors = [cmap(0.3 + 0.7 * v) for v in norm_vals]
        ax.barh(dup_series.index, dup_series.values, color=colors)
        ax.set_xlabel('Number of duplicate entries')
        ax.set_title('Duplicate values by column')
        ax.invert_yaxis()

        # annotate bars with their values
        for p in ax.patches:
            width = p.get_width()
            if width is None:
                continue
            ax.text(width + max(1, fig_w * 2), p.get_y() + p.get_height() / 2,
                    f'{int(width)}', va='center', fontsize=9)

        # increase left margin so long y labels are visible
        fig.subplots_adjust(left=0.30)

    if SAVE_PLOTS:
        save_figure(fig, 'duplicate_values')
    if SHOW_PLOTS:
        fig.show()
    return fig


def plot_feature_target_grid(df: pd.DataFrame, target_col: str | None = None, per_row: int = 6, sample_limit: int | None = 2000) -> plt.Figure:
    """Create a grid of small scatter/regression plots: each numeric feature vs the target column.

    - Detects a sensible default target: 'Diagnosis' if present, otherwise the last numeric column.
    - `per_row` controls how many subplots horizontally (original snippet used 9).
    - `sample_limit` reduces plotted points for speed/clarity; set to None to use full data.
    """
    if target_col is None:
        if 'Diagnosis' in df.columns:
            target = 'Diagnosis'
        else:
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if not num_cols:
                raise ValueError('No numeric columns available to plot')
            target = num_cols[-1]
    else:
        target = target_col

    # ensure identifier columns removed
    df = _drop_identifier_columns(df)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    features = [c for c in numeric_cols if c != target]

    n_features = len(features)
    if n_features == 0:
        raise ValueError('No features to plot against target')

    # take a sample for plotting speed if requested
    if sample_limit is not None and sample_limit > 0 and len(df) > sample_limit:
        df_plot = df.sample(sample_limit, random_state=0)
    else:
        df_plot = df

    rows = math.ceil(n_features / per_row)
    # widen columns to match combined distribution grid per-column width
    fig = plt.figure(figsize=(per_row * COLUMN_WIDTH, max(2.5, rows * 2.5)))

    counter = 1
    for i, feat in enumerate(features):
        ax = fig.add_subplot(rows, per_row, counter)
        # prefer a small regression plot, fall back to scatter if values not suitable
        try:
            sns.regplot(x=df_plot[feat], y=df_plot[target], scatter_kws={'s': 10}, line_kws={'color': 'red'}, ax=ax)
        except Exception:
            sns.scatterplot(x=df_plot[feat], y=df_plot[target], s=10, ax=ax)
        # move subplot title up slightly to avoid overlap and hide x-axis tick labels
        ax.set_title(feat, fontsize=8, pad=6)
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.set_ylabel('')
        counter += 1

    # larger title and moved down slightly to avoid overlap with subplots
    fig.suptitle(f'Features vs {target}', fontsize=20, y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if SAVE_PLOTS:
        save_figure(fig, 'features_vs_target')
    if SHOW_PLOTS:
        fig.show()
    return fig


def plot_features_vs_diagnosis_lineplots(df: pd.DataFrame, target: str = 'Diagnosis', per_row: int = 6) -> plt.Figure:
    """Plot a lineplot of each column vs the Diagnosis column (one subplot per feature)."""
    df = _drop_identifier_columns(df)
    cols = [c for c in df.columns if c != target]
    n = len(cols)
    rows = math.ceil(n / per_row)
    # widen columns to match other grids
    fig = plt.figure(figsize=(per_row * COLUMN_WIDTH, max(3, rows * 3)))
    for i, col in enumerate(cols):
        ax = fig.add_subplot(rows, per_row, i + 1)
        try:
            sns.lineplot(x=df[col], y=df[target], data=df, ax=ax)
        except Exception:
            # fallback to scatter if lineplot fails
            sns.scatterplot(x=df[col], y=df[target], data=df, s=10, ax=ax)
        # hide x-axis labels and place title above
        ax.set_title(col, fontsize=8, pad=6)
        ax.set_xlabel('')
        ax.set_xticklabels([])
    plt.tight_layout()
    if SAVE_PLOTS:
        save_figure(fig, 'features_vs_diagnosis_lineplots')
    if SHOW_PLOTS:
        fig.show()
    return fig


def plot_selected_correlation_heatmap(df: pd.DataFrame, corr_columns: list[str] | None = None) -> plt.Figure:
    """Plot a correlation heatmap for the requested columns.

    If `corr_columns` is None, the function defaults to all numeric columns in `df`
    except those listed in `IDENTIFIER_COLS`.
    """
    # If no columns provided, default to all numeric non-identifier columns
    if corr_columns is None:
        corr_columns = [c for c in df.select_dtypes(include='number').columns if c not in IDENTIFIER_COLS]
    else:
        # filter out identifier columns from requested corr_columns
        corr_columns = [c for c in corr_columns if c not in IDENTIFIER_COLS and c in df.columns]

    if not corr_columns:
        raise ValueError('No columns available for correlation heatmap after filtering identifiers')

    corr = df[corr_columns].corr()
    sns.set_palette("husl")
    # scale figure size with number of columns so annotations remain readable
    n = len(corr_columns)
    # width/height scales: 0.5 inch per variable, with reasonable minimums
    fig_w = max(10, n * 0.5)
    fig_h = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # larger annotation font and tick labels for readability
    annot_kws = {"fontsize": 8 + (0 if n < 20 else 0)}
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5, ax=ax, annot_kws=annot_kws)
    ax.set_title('Correlation Heatmap', fontsize=16)
    # increase tick label font sizes
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    if SAVE_PLOTS:
        save_figure(fig, 'correlation_heatmap')
    if SHOW_PLOTS:
        fig.show()
    return fig


def plot_pies(
    df: pd.DataFrame,
    target: str | None = 'ALL',
    categorical_columns: list[str] | None = None,
    bins_map: dict | None = None,
    per_row: int = 3,
    max_categories: int = 10,
) -> list[Path]:
    """Unified pie plotting helper.

    Behavior:
    - If `target` is a column name (and exists in `df`), produce a single pie for that column.
    - If `target` is None or 'ALL', produce two separate grid figures:
        1) categorical pies: one pie per categorical/low-cardinality column (or
           the explicit `categorical_columns` list if provided)
        2) binned pies: numeric columns binned according to `bins_map` (or
           sensible defaults if not provided)

    Returns a list of Paths written (may be empty if SAVE_PLOTS is False).
    """
    sns.set_palette("husl")
    df = _drop_identifier_columns(df)
    written: list[Path] = []

    # Single-column pie requested
    if target is not None and target != 'ALL':
        if target not in df.columns:
            raise ValueError(f'Column not found: {target}')
        counts = df[target].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        vals = list(counts.index)
        if set([v for v in vals if pd.notna(v)]).issubset({0, 1, True, False}):
            label_map = {1: 'Yes', True: 'Yes', 0: 'No', False: 'No'}
            labels = [f"{label_map.get(v, str(v))} ({counts[v]})" for v in vals]
        else:
            labels = [f"{str(v)} ({counts[v]})" for v in vals]
        ax.pie(counts.values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 10})
        display = DISPLAY_NAME_MAP.get(target, target)
        ax.set_title(f'{display} Proportion (n={int(counts.sum())})', fontsize=12)
        if SAVE_PLOTS:
            out = save_figure(fig, f'{target}_pie', subfolder='pie')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)
        return written

    # ---------- CATEGORICAL / LOW-CARDINALITY PIES ----------
    # Determine categorical columns if not provided
    if categorical_columns is None:
        # candidate columns: object / category dtype OR low-cardinality (<= max_categories)
        candidates = []
        for c in df.columns:
            if c in IDENTIFIER_COLS:
                continue
            try:
                nunq = df[c].nunique(dropna=True)
            except Exception:
                nunq = 0
            if df[c].dtype.name in ('object', 'category') or (1 < nunq <= max_categories):
                candidates.append(c)
        categorical_columns = candidates
    else:
        categorical_columns = [c for c in categorical_columns if c in df.columns and c not in IDENTIFIER_COLS]

    if categorical_columns:
        pies = categorical_columns
        rows = math.ceil(len(pies) / per_row)
        fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 5, rows * 5))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, col in enumerate(pies):
            ax = axes_flat[i]
            counts = df[col].value_counts(dropna=False)
            vals = list(counts.index)
            if set([v for v in vals if pd.notna(v)]).issubset({0, 1, True, False}):
                label_map = {1: 'Yes', True: 'Yes', 0: 'No', False: 'No'}
                labels = [f"{label_map.get(v, str(v))} ({counts[v]})" for v in vals]
            else:
                labels = [f"{str(v)} ({counts[v]})" for v in vals]
            ax.pie(counts.values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 9})
            display = DISPLAY_NAME_MAP.get(col, col)
            ax.set_title(f'{display} (n={int(counts.sum())})', fontsize=10)
        for j in range(len(pies), len(axes_flat)):
            try:
                axes_flat[j].set_visible(False)
            except Exception:
                pass
        plt.tight_layout()
        if SAVE_PLOTS:
            out = save_figure(fig, 'pie_categorical', subfolder='pie')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)

    # ---------- BINNED NUMERICAL PIES ----------
    if bins_map is None:
        bins_map = {
            'Age': [50, 60, 70, 80, 120],
            'BMI': [0, 18.5, 25, 30, 100],
            'SystolicBP': [0, 120, 130, 140, 300],
            'DiastolicBP': [0, 80, 90, 100, 200],
            'CholesterolTotal': [0, 200, 240, 1000],
            'MMSE': [0, 18, 24, 30]
        }
    available = [c for c in bins_map.keys() if c in df.columns]
    if available:
        rows = math.ceil(len(available) / per_row)
        fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 5, rows * 5))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, col in enumerate(available):
            ax = axes_flat[i]
            edges = bins_map[col]
            try:
                binned = pd.cut(df[col], bins=edges, include_lowest=True)
                counts = binned.value_counts(dropna=False).sort_index()
                labels = [f"{str(interval)} ({counts[interval]})" for interval in counts.index]
                ax.pie(counts.values, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 8})
                display = DISPLAY_NAME_MAP.get(col, col)
                ax.set_title(f'{display} (n={int(counts.sum())})', fontsize=10)
            except Exception:
                ax.set_visible(False)
        for j in range(len(available), len(axes_flat)):
            try:
                axes_flat[j].set_visible(False)
            except Exception:
                pass
        plt.tight_layout()
        if SAVE_PLOTS:
            out = save_figure(fig, 'pie_binned', subfolder='pie')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)

    return written


def plot_pair_scatter(
    df: pd.DataFrame,
    x: str | None = None,
    y: str | None = None,
    pairs: list[tuple[str, str]] | None = None,
    per_row: int = 2,
    sample_limit: int | None = 2000,
) -> list[Path] | plt.Figure:
    """Create scatter plots for numeric attribute pairs.

    Usage modes:
    - If both `x` and `y` are provided, create a single scatter of `x` vs `y`.
    - If `pairs` is provided (list of (x,y)), produce a grid of scatter plots.
      If neither pairs nor (x,y) provided, a sensible default set of pairs is used.

    The function will drop identifier columns and sample the dataframe for speed.
    Returns saved Path(s) if SAVE_PLOTS is True, otherwise returns the Figure(s).
    """
    sns.set_palette("husl")
    df = _drop_identifier_columns(df)

    # determine sample for plotting
    if sample_limit is not None and sample_limit > 0 and len(df) > sample_limit:
        df_plot = df.sample(sample_limit, random_state=0)
    else:
        df_plot = df

    # If explicit single pair provided
    if x and y:
        if x not in df_plot.columns or y not in df_plot.columns:
            raise ValueError(f'Columns not found in dataframe: {x}, {y}')
        fig, ax = plt.subplots(figsize=(SINGLE_PLOT_WIDTH, 6))
        try:
            sns.scatterplot(data=df_plot, x=x, y=y, hue='Diagnosis' if 'Diagnosis' in df_plot.columns else None, s=20, ax=ax)
        except Exception:
            sns.scatterplot(data=df_plot, x=x, y=y, s=20, ax=ax)
        ax.set_title(f'{DISPLAY_NAME_MAP.get(y,y)} vs {DISPLAY_NAME_MAP.get(x,x)}')
        ax.set_xlabel(DISPLAY_NAME_MAP.get(x, x))
        ax.set_ylabel(DISPLAY_NAME_MAP.get(y, y))
        plt.tight_layout()
        if SAVE_PLOTS:
            out = save_figure(fig, f'scatter_{x}_vs_{y}')
            if SHOW_PLOTS:
                fig.show()
            plt.close(fig)
            return [out]
        else:
            if SHOW_PLOTS:
                fig.show()
            return fig

    # Default pairs if none provided (user-requested comprehensive list)
    if pairs is None:
        pairs = [
            # Blood Pressure & Body Metrics
            ('BMI', 'SystolicBP'),
            ('BMI', 'DiastolicBP'),
            ('Age', 'SystolicBP'),
            ('PhysicalActivity', 'BMI'),
            ('SystolicBP', 'DiastolicBP'),
            # Cholesterol & Body Metrics
            ('Age', 'CholesterolTotal'),
            ('BMI', 'CholesterolLDL'),
            ('BMI', 'CholesterolTriglycerides'),
            ('PhysicalActivity', 'CholesterolHDL'),
            ('CholesterolLDL', 'CholesterolHDL'),
            # Cognitive & Lifestyle Metrics
            ('Age', 'MMSE'),
            ('SleepQuality', 'MMSE'),
            ('PhysicalActivity', 'MMSE'),
            ('DietQuality', 'MMSE'),
            ('AlcoholConsumption', 'SleepQuality'),
            # General Health & Lifestyle
            ('PhysicalActivity', 'SleepQuality'),
            ('DietQuality', 'BMI'),
            ('Age', 'PhysicalActivity'),
            ('AlcoholConsumption', 'SystolicBP')
        ]

    # filter pairs to those present in data
    valid_pairs = [(a, b) for (a, b) in pairs if a in df_plot.columns and b in df_plot.columns]
    if not valid_pairs:
        raise ValueError('No valid column pairs found in dataframe for scatter plotting')

    written: list[Path] = []
    # Create and save each pair as a separate figure in its own folder
    for a, b in valid_pairs:
        fig, ax = plt.subplots(figsize=(SINGLE_PLOT_WIDTH, 6))
        try:
            sns.scatterplot(data=df_plot, x=a, y=b, hue='Diagnosis' if 'Diagnosis' in df_plot.columns else None, s=20, ax=ax)
        except Exception:
            sns.scatterplot(data=df_plot, x=a, y=b, s=20, ax=ax)
        ax.set_title(f'{DISPLAY_NAME_MAP.get(b,b)} vs {DISPLAY_NAME_MAP.get(a,a)}')
        ax.set_xlabel(DISPLAY_NAME_MAP.get(a, a))
        ax.set_ylabel(DISPLAY_NAME_MAP.get(b, b))
        plt.tight_layout()
        safe_name = f"{a}_vs_{b}".replace(' ', '_')
        if SAVE_PLOTS:
            out = save_figure(fig, safe_name, subfolder=f'scatter/{safe_name}')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)

    return written


def plot_mmse_box_by_diagnosis(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    per_row: int = 2,
    sample_limit: int | None = 2000,
) -> list[Path]:
    """Create box plots for requested categorical vs numeric pairs.

    Default pairs (from user request) include cognitive, cardiovascular, and lifestyle comparisons.
    Each pair is saved as its own PNG under OUTPUT_DIR/box/<x>_vs_<y>/<x>_vs_<y>.png
    Returns a list of written Paths.
    """
    sns.set_palette("husl")
    df = _drop_identifier_columns(df)

    # sample for speed
    if sample_limit is not None and sample_limit > 0 and len(df) > sample_limit:
        df_plot = df.sample(sample_limit, random_state=0)
    else:
        df_plot = df

    # default pairs requested by user
    if pairs is None:
        pairs = [
            # Cognitive & Clinical
            ('Diagnosis', 'MMSE'),
            ('EducationLevel', 'MMSE'),
            ('FamilyHistoryAlzheimers', 'MMSE'),
            ('Diagnosis', 'FunctionalAssessment'),
            # Cardiovascular vs Demographics & Lifestyle
            ('Hypertension', 'SystolicBP'),
            ('Smoking', 'SystolicBP'),
            ('CardiovascularDisease', 'CholesterolLDL'),
            ('Gender', 'CholesterolTotal'),
            ('Diabetes', 'BMI'),
            ('Smoking', 'BMI'),
            # General Health & Behavioral & Lifestyle
            ('Diagnosis', 'Age'),
            ('Depression', 'SleepQuality'),
            ('Diagnosis', 'PhysicalActivity'),
            ('EducationLevel', 'DietQuality')
        ]

    written: list[Path] = []
    valid_pairs = [(x, y) for (x, y) in pairs if x in df_plot.columns and y in df_plot.columns]
    if not valid_pairs:
        return []

    for x, y in valid_pairs:
        fig, ax = plt.subplots(figsize=(SINGLE_PLOT_WIDTH, 6))
        try:
            # avoid passing `palette` without `hue` (deprecated); use a single color
            sns.boxplot(data=df_plot, x=x, y=y, color='skyblue', ax=ax)
        except Exception:
            try:
                sns.violinplot(data=df_plot, x=x, y=y, ax=ax)
            except Exception:
                plt.close(fig)
                continue

        display_x = DISPLAY_NAME_MAP.get(x, x)
        display_y = DISPLAY_NAME_MAP.get(y, y)
        ax.set_title(f'{display_y} by {display_x}', fontsize=12)
        ax.set_xlabel(display_x)
        ax.set_ylabel(display_y)
        plt.tight_layout()

        safe_name = f"{x}_vs_{y}".replace(' ', '_')
        if SAVE_PLOTS:
            out = save_figure(fig, safe_name, subfolder=f'box/{safe_name}')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)

    return written


def plot_violin_box_pairs(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    sample_limit: int | None = 2000,
) -> list[Path]:
    """Create violin or box plots for requested categorical vs numeric pairs.

    Each pair is saved to OUTPUT_DIR/violin/<x>_vs_<y>/<x>_vs_<y>.png
    Returns list of written Paths.
    """
    sns.set_palette("husl")
    df = _drop_identifier_columns(df)

    # sample for speed
    if sample_limit is not None and sample_limit > 0 and len(df) > sample_limit:
        df_plot = df.sample(sample_limit, random_state=0)
    else:
        df_plot = df

    # default pairs as requested
    if pairs is None:
        pairs = [
            # Cognitive & Functional
            ('Diagnosis', 'MMSE'),
            ('MemoryComplaints', 'MMSE'),
            ('Diagnosis', 'FunctionalAssessment'),
            ('DifficultyCompletingTasks', 'ADL'),
            # Cardiovascular & Metabolic
            ('Hypertension', 'SystolicBP'),
            ('Diabetes', 'BMI'),
            ('CardiovascularDisease', 'CholesterolTotal'),
            ('Smoking', 'CholesterolTriglycerides'),
            # General Health & Lifestyle
            ('Diagnosis', 'Age'),
            ('Depression', 'SleepQuality'),
            ('EducationLevel', 'DietQuality'),
            ('Gender', 'PhysicalActivity')
        ]

    written: list[Path] = []
    valid_pairs = [(x, y) for (x, y) in pairs if x in df_plot.columns and y in df_plot.columns]
    if not valid_pairs:
        return []

    for x, y in valid_pairs:
        fig, ax = plt.subplots(figsize=(SINGLE_PLOT_WIDTH, 6))
        try:
            sns.violinplot(data=df_plot, x=x, y=y, inner='box', cut=0, ax=ax)
        except Exception:
            try:
                sns.boxplot(data=df_plot, x=x, y=y, color='skyblue', ax=ax)
            except Exception:
                plt.close(fig)
                continue

        display_x = DISPLAY_NAME_MAP.get(x, x)
        display_y = DISPLAY_NAME_MAP.get(y, y)
        ax.set_title(f'{display_y} by {display_x}', fontsize=12)
        ax.set_xlabel(display_x)
        ax.set_ylabel(display_y)
        plt.tight_layout()

        safe_name = f"{x}_vs_{y}".replace(' ', '_')
        if SAVE_PLOTS:
            out = save_figure(fig, safe_name, subfolder=f'violin/{safe_name}')
            written.append(out)
        if SHOW_PLOTS:
            fig.show()
        plt.close(fig)

    return written


def plot_histograms(column_list: list[str], fig_title: str, df: pd.DataFrame, per_row: int = 5) -> plt.Figure:
    """Plot histograms for the provided numeric columns from df in a single figure.

    By default arranges 6 plots per row. The figure is sized using `COLUMN_WIDTH`
    so it aligns with other grid plots in this module.
    """
    # filter out any identifier columns from the provided list
    column_list = [c for c in column_list if c not in IDENTIFIER_COLS]
    # determine binary-like columns (only 0/1/True/False present) and
    # exclude them from the combined grid so they don't dominate scales.
    binary_cols = []
    for c in list(column_list):
        try:
            unique_vals = set(pd.Series(df[c].dropna().unique()).tolist())
        except Exception:
            unique_vals = set()
        # treat boolean-like and strict {0,1} as binary
        if unique_vals.issubset({0, 1, True, False}) and len(unique_vals) <= 2:
            binary_cols.append(c)
            # keep them out of the combined grid but keep in the list for per-column saves
            column_list.remove(c)
    n_cols = per_row  # subplots per row (default 5)
    n_rows = int(math.ceil(len(column_list) / n_cols)) if column_list else 0
    # width: increase per-column width so each subplot is wider on the page
    # multiply COLUMN_WIDTH by a scale factor to make each subplot noticeably larger
    PER_COLUMN_WIDTH_SCALE = 1.8
    fig_w = max(12, COLUMN_WIDTH * n_cols * PER_COLUMN_WIDTH_SCALE)
    # increase per-row height so each subplot is larger (4.5 inches per row)
    fig_h = max(5, 4.5 * max(1, n_rows))
    sns.set_palette("husl")

    # create grid of subplots so spacing can be controlled more predictably
    if n_rows == 0:
        fig, axes = plt.subplots(figsize=(fig_w, fig_h))
        axes = [axes]
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(column_list):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=20, color='pink', edgecolor='black')
        ax.set_title(f'{col} Distribution', fontsize=10, pad=6)
        # show x-axis tick labels so numeric ranges are visible to the user
        # use a smaller font and rotate if labels overlap
        ax.set_xlabel(DISPLAY_NAME_MAP.get(col, col))
        ax.tick_params(axis='x', labelsize=8)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
        ax.set_ylabel('Number of Patients')

    # hide any unused axes
    for j in range(len(column_list), len(axes)):
        try:
            axes[j].set_visible(False)
        except Exception:
            pass

    # reduce vertical whitespace and slightly tighten overall layout
    plt.subplots_adjust(hspace=0.35, wspace=0.35)
    plt.tight_layout()
    # place the suptitle slightly above the tight layout
    fig.suptitle(fig_title, fontsize=16, y=1.02)
    if SAVE_PLOTS:
        # save combined figure (all histograms in one image)
        combined_out = save_figure(fig, fig_title, subfolder='histograms')
        # also save each column separately (like the distribution helper)
        # save per-column for both the columns in the combined grid and the binary columns we excluded
        for col in (column_list + binary_cols):
            safe_name = str(col).replace(' ', '_')
            try:
                f2, ax2 = plt.subplots(figsize=(SINGLE_PLOT_WIDTH, 5))
                try:
                    ax2.hist(df[col].dropna(), bins=20, color='pink', edgecolor='black')
                except Exception:
                    vals = df[col].value_counts()
                    sns.barplot(x=vals.values, y=vals.index, ax=ax2)
                display = DISPLAY_NAME_MAP.get(col, col)
                ax2.set_title(f'{display} Distribution', fontsize=12, pad=6)
                # show x-axis tick labels for per-column saved histograms
                ax2.set_xlabel(display)
                ax2.tick_params(axis='x', labelsize=9)
                for label in ax2.get_xticklabels():
                    label.set_rotation(30)
                ax2.set_ylabel('Number of Patients')
                save_figure(f2, safe_name, subfolder='histograms')
            except Exception:
                try:
                    plt.close(f2)
                except Exception:
                    pass
            finally:
                try:
                    plt.close(f2)
                except Exception:
                    pass
    if SHOW_PLOTS:
        fig.show()
    return fig


def main():
    df = load_data(sample_rows=None)  # load full file by default for EDA
    # Write dataset basic info to a report file instead of printing to console
    report_path = save_basic_info_to_file(df)
    if not QUIET:
        print(f'Dataset report written to: {report_path}')

    # Basic plots
    try:
        # Use the unified distribution plotting helper for Age
        written_age = plot_distributions(['Age'], df)
        # If interactive display requested, show any open pyplot figures
        if SHOW_PLOTS:
            plt.show()
    except Exception as exc:
        print('age distribution plot failed:', exc)

    try:
        f2 = plot_missing_values(df)
        f2.tight_layout()
        if SHOW_PLOTS:
            f2.show()
    except Exception as exc:
        print('missing values plot failed:', exc)

    # Duplicate values plot (how many non-unique entries per column)
    try:
        fdup = plot_duplicate_values(df)
        fdup.tight_layout()
        if SHOW_PLOTS:
            fdup.show()
    except Exception as exc:
        print('duplicate values plot failed:', exc)

    # Correlation / feature vs target grid (samples up to 2000 rows for speed)
    try:
        plot_feature_target_grid(df, target_col='Diagnosis', per_row=6, sample_limit=2000)
    except Exception as exc:
        print('Could not create feature-target grid:', exc)

    # Feature vs Diagnosis lineplots
    try:
        plot_features_vs_diagnosis_lineplots(df, target='Diagnosis', per_row=6)
    except Exception as exc:
        print('features vs diagnosis plots failed:', exc)

    # Heatmap and pie
    try:
        # Plot correlation heatmap for all numeric features except identifiers
        plot_selected_correlation_heatmap(df)
    except Exception as exc:
        print('correlation heatmap failed:', exc)

    try:
        # produce pies: categorical and binned numerical (uses loaded df)
        plot_pies(df, target='ALL')
    except Exception as exc:
        print('pie plotting failed:', exc)

    # Histograms (all numeric features merged into a single image)
    try:
        # Get all numeric columns and plot them together in one figure
        columns = df.select_dtypes(include='number').columns.tolist()
        if columns:
            plot_histograms(columns, "Numeric Feature Distributions", df, per_row=5)
    except Exception as exc:
        print('histograms failed:', exc)

    # Save individual distributions for the requested attributes
    try:
        attributes = [
            'PatientID','Age','Gender','Ethnicity','EducationLevel','BMI','Smoking','AlcoholConsumption','PhysicalActivity','DietQuality','SleepQuality','FamilyHistoryAlzheimers','CardiovascularDisease','Diabetes','Depression','HeadInjury','Hypertension','SystolicBP','DiastolicBP','CholesterolTotal','CholesterolLDL','CholesterolHDL','CholesterolTriglycerides','MMSE','FunctionalAssessment','MemoryComplaints','BehavioralProblems','ADL','Confusion','Disorientation','PersonalityChanges','DifficultyCompletingTasks','Forgetfulness','Diagnosis','DoctorInCharge'
        ]
        # remove any identifier columns from the list to avoid plotting them
        attributes = [a for a in attributes if a not in IDENTIFIER_COLS]
        written = plot_distributions(attributes, df)
        # per-user request: do not print per-plot distribution messages here
    except Exception as exc:
        print('distribution plots failed:', exc)
        
    # Combined figure with all distributions in a grid
    try:
        written_all = plot_distributions(attributes, df, combine=True, per_row=5)
        # per-user request: do not print combined distribution message here
    except Exception as exc:
        print('combined distribution plot failed:', exc)

    # Scatter / box / violin
    try:
        plot_pair_scatter(df)
    except Exception as exc:
        print('age vs mmse scatter failed:', exc)

    try:
        plot_mmse_box_by_diagnosis(df)
    except Exception as exc:
        print('mmse boxplot failed:', exc)

    try:
        plot_violin_box_pairs(df)
    except Exception as exc:
        print('mmse violin plot failed:', exc)

    # Final summary message: print once that all plots were saved (if saving enabled)
    if SAVE_PLOTS and not QUIET:
        print(f'All plots saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Exploratory plotting helpers')
    parser.add_argument('--report-only', action='store_true', help='Write dataset info to a text file and exit')
    parser.add_argument('--report-out', type=str, default=None, help='Path to write the report text file')
    parser.add_argument('--no-show', action='store_true', help='Do not show interactive plot windows')
    parser.add_argument('--no-save', action='store_true', help='Do not save plot files')
    parser.add_argument('--quiet', action='store_true', help='Suppress informational console output about saved files')
    args = parser.parse_args()

    if args.report_only:
        df = load_data(sample_rows=None)
        out = save_basic_info_to_file(df, out_path=args.report_out)
        print(f'Wrote dataset report to: {out}')
    else:
        # set show/save flags
        if args.no_show:
            SHOW_PLOTS = False
        if args.no_save:
            SAVE_PLOTS = False
        if args.quiet:
            QUIET = True
        main()
