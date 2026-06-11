import json
import logging
import os

import numpy as np
import pandas as pd

from ..base.utils import to_native_types as _canonical_to_native_types

logger = logging.getLogger(__name__)


class Exporter:
    """
    Export data/results in various formats: DataFrame/CSV, JSON, dictionary, and
    NumPy array. Provides both display (to console) and save (to file) helpers,
    with automatic conversion of NumPy types to native Python types.

    Safety
    ------
    By default the Exporter does **not** overwrite existing files: ``save_*`` /
    ``export_*`` raise :class:`FileExistsError` if the target already exists.
    Pass ``overwrite=True`` (per call) or construct with ``overwrite=True`` to
    allow clobbering. This protects previously-saved results from silent loss
    when a study is re-run.
    """

    def __init__(self, output_dir="exported_results", overwrite=False):
        """
        Parameters
        ----------
        output_dir : str
            Directory to save exported files (created if missing).
        overwrite : bool
            Default overwrite policy for all ``save_*`` / ``export_*`` calls.
            ``False`` (default) refuses to clobber existing files.
        """
        self.output_dir = output_dir
        self.overwrite = bool(overwrite)
        os.makedirs(self.output_dir, exist_ok=True)

    # --- path resolution + overwrite guard ---------------------------------
    def _resolve_path(self, file_name, overwrite=None):
        """Join ``file_name`` to ``output_dir`` and enforce the overwrite policy.

        Ensures the output directory exists at save time (not just at
        construction) and raises :class:`FileExistsError` if the target exists
        and overwriting is not permitted.
        """
        allow = self.overwrite if overwrite is None else bool(overwrite)
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, file_name)
        if os.path.exists(file_path) and not allow:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {file_path}. "
                f"Pass overwrite=True (or construct Exporter(overwrite=True)) to allow."
            )
        return file_path

    # --- NumPy → Python conversion helper ----------------------------------
    @staticmethod
    def to_native_types(obj):
        """
        Recursively convert NumPy scalar types in dicts/lists/tuples to native
        Python types. Thin wrapper over :func:`quends.base.utils.to_native_types`.
        """
        return _canonical_to_native_types(obj)

    # --- conversion helpers ------------------------------------------------
    def to_dataframe(self, data):
        """Convert input data to a pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        else:
            try:
                return pd.DataFrame(data)
            except Exception as e:  # noqa: BLE001 - re-raised as a clear ValueError
                raise ValueError("Cannot convert data to DataFrame: " + str(e))

    def to_dictionary(self, data):
        """Convert input data to a dictionary with native Python types."""
        if isinstance(data, dict):
            d = data
        elif isinstance(data, pd.DataFrame):
            d = data.to_dict(orient="list")
        elif isinstance(data, np.ndarray):
            d = {"data": data.tolist()}
        else:
            raise ValueError("Cannot convert data to dictionary.")
        return self.to_native_types(d)

    def to_numpy(self, data):
        """Convert input data to a NumPy array."""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, dict):
            return pd.DataFrame(data).values
        else:
            raise ValueError("Cannot convert data to NumPy array.")

    def to_json(self, data):
        """Convert input data to a JSON string with native Python types."""
        if isinstance(data, pd.DataFrame):
            d = data.to_dict(orient="list")
        elif isinstance(data, dict):
            d = data
        elif isinstance(data, np.ndarray):
            d = data.tolist()
        else:
            raise ValueError("Cannot convert data to JSON.")
        d = self.to_native_types(d)
        return json.dumps(d, indent=2)

    # --- display functions -------------------------------------------------
    def display_dataframe(self, data, head=None):
        """Display data as a DataFrame."""
        df = self.to_dataframe(data)
        print(df.head(head) if head is not None else df)

    def display_dictionary(self, data):
        """Display data as a (native-typed) dictionary."""
        print(self.to_dictionary(data))

    def display_numpy(self, data):
        """Display data as a NumPy array."""
        print(self.to_numpy(data))

    def display_json(self, data):
        """Display data as a (native-typed) JSON string."""
        print(self.to_json(data))

    # --- save functions ----------------------------------------------------
    def save_dataframe(self, data, file_name="dataframe.csv", *, overwrite=None):
        """Save data as a CSV file. Returns the written path."""
        df = self.to_dataframe(data)
        file_path = self._resolve_path(file_name, overwrite)
        df.to_csv(file_path, index=False)
        logger.info("DataFrame saved to %s", file_path)
        return file_path

    def save_dictionary(self, data, file_name="data_dictionary.json", *, overwrite=None):
        """Save data as a JSON dictionary file. Returns the written path."""
        d = self.to_dictionary(data)
        file_path = self._resolve_path(file_name, overwrite)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        logger.info("Dictionary saved to %s", file_path)
        return file_path

    def save_numpy(self, data, file_name="data.npy", *, overwrite=None):
        """Save data as a ``.npy`` file. Returns the written path."""
        arr = self.to_numpy(data)
        file_path = self._resolve_path(file_name, overwrite)
        np.save(file_path, arr)
        logger.info("NumPy array saved to %s", file_path)
        return file_path

    def save_json(self, data, file_name="data.json", *, overwrite=None):
        """Save data as a JSON file. Returns the written path."""
        j = self.to_json(data)
        file_path = self._resolve_path(file_name, overwrite)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(j)
        logger.info("JSON saved to %s", file_path)
        return file_path

    def export_figure(self, fig, filename="figure.png", dpi=300, *, overwrite=None):
        """Save a Matplotlib figure. Returns the written path."""
        file_path = self._resolve_path(filename, overwrite)
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        logger.info("Figure exported to %s", file_path)
        return file_path

    def save_results(self, results, name, *, overwrite=None, metadata=None):
        """Save a results object as CSV (if tabular) or JSON, plus a provenance
        sidecar ``<name>.meta.json``.

        Parameters
        ----------
        results : DataFrame | dict | ndarray
            The results to persist.
        name : str
            Base name (without extension). A data file and a ``<name>.meta.json``
            sidecar are written.
        overwrite : bool, optional
            Per-call overwrite policy (defaults to the instance policy).
        metadata : dict, optional
            Provenance to record in the sidecar (source file, variable, trim
            parameters, etc.). ``schema_version`` is added automatically.

        Returns
        -------
        (data_path, meta_path)
        """
        if isinstance(results, pd.DataFrame):
            data_path = self.save_dataframe(results, f"{name}.csv", overwrite=overwrite)
        else:
            data_path = self.save_json(results, f"{name}.json", overwrite=overwrite)
        sidecar = {"schema_version": "1.0", "data_file": os.path.basename(data_path)}
        if metadata:
            sidecar.update(self.to_native_types(dict(metadata)))
        meta_path = self._resolve_path(f"{name}.meta.json", overwrite)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        logger.info("Results + provenance saved to %s (+ %s)", data_path, meta_path)
        return data_path, meta_path

    def export_dataframe(self, data, filename="dataframe.csv", *, overwrite=None):
        """Deprecated alias for :meth:`save_dataframe` (kept for compatibility)."""
        return self.save_dataframe(data, file_name=filename, overwrite=overwrite)
