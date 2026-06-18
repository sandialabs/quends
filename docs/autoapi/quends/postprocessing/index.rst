quends.postprocessing
=====================

.. py:module:: quends.postprocessing

.. autoapi-nested-parse::

   Postprocessing utilities for QUENDS: exporting results and plotting.



Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/quends/postprocessing/exporter/index
   /autoapi/quends/postprocessing/loader/index
   /autoapi/quends/postprocessing/plotter/index
   /autoapi/quends/postprocessing/writer/index


Classes
-------

.. autoapisummary::

   quends.postprocessing.Exporter
   quends.postprocessing.Plotter


Package Contents
----------------

.. py:class:: Exporter(output_dir='exported_results', overwrite=False)

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


   .. py:attribute:: output_dir
      :value: 'exported_results'



   .. py:attribute:: overwrite
      :value: False



   .. py:method:: to_native_types(obj)
      :staticmethod:


      Recursively convert NumPy scalar types in dicts/lists/tuples to native
      Python types. Thin wrapper over :func:`quends.base.utils.to_native_types`.



   .. py:method:: to_dataframe(data)

      Convert input data to a pandas DataFrame.



   .. py:method:: to_dictionary(data)

      Convert input data to a dictionary with native Python types.



   .. py:method:: to_numpy(data)

      Convert input data to a NumPy array.



   .. py:method:: to_json(data)

      Convert input data to a JSON string with native Python types.



   .. py:method:: display_dataframe(data, head=None)

      Display data as a DataFrame.



   .. py:method:: display_dictionary(data)

      Display data as a (native-typed) dictionary.



   .. py:method:: display_numpy(data)

      Display data as a NumPy array.



   .. py:method:: display_json(data)

      Display data as a (native-typed) JSON string.



   .. py:method:: save_dataframe(data, file_name='dataframe.csv', *, overwrite=None)

      Save data as a CSV file. Returns the written path.



   .. py:method:: save_dictionary(data, file_name='data_dictionary.json', *, overwrite=None)

      Save data as a JSON dictionary file. Returns the written path.



   .. py:method:: save_numpy(data, file_name='data.npy', *, overwrite=None)

      Save data as a ``.npy`` file. Returns the written path.



   .. py:method:: save_json(data, file_name='data.json', *, overwrite=None)

      Save data as a JSON file. Returns the written path.



   .. py:method:: export_figure(fig, filename='figure.png', dpi=300, *, overwrite=None)

      Save a Matplotlib figure. Returns the written path.



   .. py:method:: save_results(results, name, *, overwrite=None, metadata=None)

      Save a results object as CSV (if tabular) or JSON, plus a provenance
      sidecar ``<name>.meta.json``.

      :Parameters: * **results** (*DataFrame | dict | ndarray*) -- The results to persist.
                   * **name** (*str*) -- Base name (without extension). A data file and a ``<name>.meta.json``
                     sidecar are written.
                   * **overwrite** (*bool, optional*) -- Per-call overwrite policy (defaults to the instance policy).
                   * **metadata** (*dict, optional*) -- Provenance to record in the sidecar (source file, variable, trim
                     parameters, etc.). ``schema_version`` is added automatically.

      :returns: *(data_path, meta_path)*



   .. py:method:: export_dataframe(data, filename='dataframe.csv', *, overwrite=None)

      Deprecated alias for :meth:`save_dataframe` (kept for compatibility).



.. py:class:: Plotter(output_dir='results_figures')

   Plotting utilities for DataStream and Ensemble objects.

   Uniform plotting contract
   --------------------------
   Every public plotting method accepts the same output-control keywords and
   returns the Matplotlib objects instead of forcing display:

   * ``save`` (bool)        — write the figure to disk.
   * ``show`` (bool)        — call ``plt.show()`` (default **False**; scripts/CI safe).
   * ``filename`` (str)     — output filename (single-figure methods).
   * ``output_dir`` (str)   — per-call output directory (defaults to the ctor dir).
   * ``overwrite`` (bool)   — allow clobbering an existing file (default False).
   * ``dpi`` (int)          — raster resolution when saving.

   Return value:

   * single-figure methods return ``(fig, axes)``;
   * multi-figure methods (one figure per variable/member) return a list of
     ``(fig, axes)`` tuples.

   Regenerate-from-CSV
   -------------------
   The stats/trim-heavy methods accept optional *precomputed* arguments
   (``stats=``, ``ss_starts=``, ``means=``, ``avg_df=``, ``acf_values=``). When
   supplied, the method skips the expensive computation and only renders — so a
   figure can be rebuilt from saved results without recomputing statistics.


   .. py:attribute:: output_dir
      :value: 'results_figures'



   .. py:method:: format_dataset_name(dataset_name)
      :staticmethod:


      Return a title-cased, space-separated version of a dataset name.



   .. py:method:: trace_plot(data, variables_to_plot=None, *, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Plot raw time-series traces. Returns a list of (fig, axes).



   .. py:method:: trace_plot_with_mean(data, variables_to_plot=None, window_size=None, *, stats=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Plot each trace with block-mean + 95% CI overlaid.

      Pass ``stats={column: {"mean": .., "confidence_interval": (lo, hi)}}`` to
      skip ``compute_statistics`` (regenerate-from-saved). Returns list of (fig, axes).



   .. py:method:: ensemble_trace_plot(data, variables_to_plot=None, *, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Overlay traces from all members, one figure per variable. Returns list of (fig, axes).



   .. py:method:: ensemble_trace_plot_with_mean(data, variables_to_plot=None, window_size=None, *, means=None, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Overlay member traces with per-member block mean.

      Pass ``means={dataset_name: {var: mean}}`` to skip ``compute_statistics``.
      Returns list of (fig, axes).



   .. py:method:: steady_state_automatic_plot(data, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, *, ss_starts=None, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Auto-detect steady-state start per variable and annotate.

      Pass ``ss_starts={column: ss_start_time}`` to skip trimming
      (regenerate-from-saved). Returns list of (fig, axes).



   .. py:method:: steady_state_plot(data, variables_to_plot=None, steady_state_start=None, *, save=False, show=False, output_dir=None, overwrite=False, dpi=150)

      Annotate steady state from a user-supplied start (float or {var: float}).

      Returns list of (fig, axes).



   .. py:method:: plot_acf(data, alpha=0.05, column=None, ax=None, *, acf_values=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Plot the autocorrelation function.

      Pass ``acf_values`` (precomputed array) to skip ``acf()``. If ``ax`` is
      given the stem is drawn into it (no save/show); otherwise a new figure is
      created and returned as (fig, ax).



   .. py:method:: plot_acf_ensemble(ensemble_obj, alpha=0.05, column=None, *, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      ACF grid, one subplot per member. Returns (fig, axes).



   .. py:method:: ensemble_steady_state_automatic_plot(ensemble_obj, variables_to_plot=None, batch_size=10, start_time=0.0, method='std', threshold=None, robust=True, *, ss_starts=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Auto-detect steady state per member, one subplot each. Returns (fig, axes).

      Pass ``ss_starts={member_index: {var: ss_start}}`` to skip trimming.



   .. py:method:: ensemble_steady_state_plot(ensemble_obj, variables_to_plot=None, steady_state_start=None, *, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Annotate each member with a user-supplied SS start. Returns (fig, axes).



   .. py:method:: plot_ensemble(ensemble_obj, variables_to_plot=None, *, avg_df=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Members + ensemble average, 2-column grid. Returns (fig, axes).

      Pass ``avg_df`` (a precomputed average DataFrame) to skip
      ``compute_average_ensemble``.



   .. py:method:: plot_ensemble_with_average(ensemble_obj, variables_to_plot=None, *, avg_df=None, condensed_legend=False, y_range=None, save=False, show=False, filename=None, output_dir=None, overwrite=False, dpi=150)

      Members + ensemble average with optional condensed legend / y-range.

      Pass ``avg_df`` to skip ``compute_average_ensemble``. Returns (fig, axes).



