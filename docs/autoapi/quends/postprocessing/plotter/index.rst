quends.postprocessing.plotter
=============================

.. py:module:: quends.postprocessing.plotter


Attributes
----------

.. autoapisummary::

   quends.postprocessing.plotter.logger


Classes
-------

.. autoapisummary::

   quends.postprocessing.plotter.Plotter


Module Contents
---------------

.. py:data:: logger

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



