quends.base.utils
=================

.. py:module:: quends.base.utils


Attributes
----------

.. autoapisummary::

   quends.base.utils.SCHEMA_VERSION


Classes
-------

.. autoapisummary::

   quends.base.utils.StatsResult


Functions
---------

.. autoapisummary::

   quends.base.utils.power_law_model
   quends.base.utils.confidence_multiplier
   quends.base.utils.to_native_types
   quends.base.utils.stationarity_results
   quends.base.utils.stationarity_value
   quends.base.utils.autotune_blocks


Module Contents
---------------

.. py:data:: SCHEMA_VERSION
   :value: '1.0'


.. py:class:: StatsResult(results=None, metadata=None)

   Bases: :py:obj:`dict`


   A ``{column: stats}`` mapping with an attached ``.metadata`` dict.

   Subclasses :class:`dict`, so it behaves exactly like the historical
   ``{column: {...}}`` return value — ``res[col]["mean"]``, ``res == plain_dict``,
   iteration, ``.get`` all work unchanged. It only *adds* a ``.metadata``
   attribute (run-level info such as estimator, sample counts, schema_version),
   so existing callers keep working while new code can read provenance.


   .. py:attribute:: metadata


.. py:function:: power_law_model(n, A, p)

.. py:function:: confidence_multiplier(confidence_level = 0.95, method = 'normal', dof = None)

   Return the multiplier ``z`` such that ``CI = mean ± z × SE``.

   :Parameters: * **confidence_level** (*float*) -- Two-sided confidence level in (0, 1).  Default ``0.95``.
                * **method** (*{"normal", "t"}*) -- ``"normal"`` — standard-normal quantile.
                  ``"t"`` — Student's *t* quantile (requires *dof*).
                * **dof** (*int or None*) -- Degrees of freedom for the *t* distribution.  Required when
                  ``method="t"``.  Ignored otherwise.

   :returns: *float* -- The CI multiplier.

   .. rubric:: Notes

   For the historical default ``(method="normal", confidence_level=0.95)``,
   this function returns the literal value ``1.96`` to preserve byte-for-byte
   backward compatibility with previously-stored results.  For all other
   parameter values, the quantile is computed exactly via :mod:`scipy.stats`.


.. py:function:: to_native_types(obj)

   Recursively convert NumPy scalar and array types in nested structures to native Python types.

   This function walks through dictionaries, lists, tuples, NumPy scalars, and arrays,
   converting them into Python built-ins:

   - NumPy scalar → Python int or float
   - NumPy array  → Python list (recursively)

   :Parameters: **obj** (*any*) -- The object to convert. Supported container types are dict, list, tuple,
                NumPy ndarray/scalar. Other types are returned unchanged.

   :returns: *any* -- A new object mirroring the input structure but with all NumPy data types replaced
             by their native Python equivalents.


.. py:function:: stationarity_results(result)

   Normalize supported stationarity return schemas to a plain results mapping.

   DataStream currently returns ``{column: bool}``, while richer callers may
   use ``{"results": {column: bool}, "metadata": ...}``.  Keeping this small
   adapter at call sites prevents schema drift from turning into indexing bugs.


.. py:function:: stationarity_value(result, column, default=False)

   Extract one column's stationarity boolean from any supported schema.


.. py:function:: autotune_blocks(x, window_size=None, method = 'non-overlapping', alpha = 0.05, lag_set=(5, 10), B_min = 15, min_blocks = 2, max_iter = 25, w_min = 5, c0 = 2.0)

   Shared core helper: autotune block window until block means are independent.

   This is the single implementation used by *both* ``DataStream.compute_statistics``
   and the ensemble statistics pipeline (Techniques 1 and 2).  All other
   window-autotune helpers in this package are thin wrappers around this function.

   Algorithm
   ---------
   If *window_size* is ``None`` (autotune path):

   1. Estimate tau_int from the raw-series ACF via Geyer positive-pair
      truncation.
   2. Seed the starting window: ``w0 = max(w_min, ceil(c0 * tau_int))``,
      soft-capped to ``n // B_min`` so that at least *B_min* blocks are
      available at the start of the search.
   3. Iterate: compute non-overlapping block means, run
      ``_ljung_box_pass(lag_set, alpha)`` (lags are capped to ``n_blocks-1``
      and deduplicated), advance ``w += 1``.
   4. Return on the first window that passes (status ``"independent"``).
   5. If the loop exhausts *max_iter* or the block count drops below
      *min_blocks*, return the window with the best LB p-value seen
      (status ``"best_p"``).
   6. If no valid blocks exist at all, return status ``"too_few_blocks"``.

   If *window_size* is provided (user-window path):

   * Use the given window directly without searching.
   * Still run the LB diagnostic for informational purposes.
   * Always set status ``"user_window"``; SE should be computed via Geyer ESS.

   :Parameters: * **x** (*array-like*) -- Raw series values.  NaNs and Infs are removed before processing.
                * **window_size** (*int or None*) -- User-supplied window size.  ``None`` triggers autotuning.
                * **method** (*{"non-overlapping", "sliding"}*) -- Block type.  Independence testing is designed for
                  ``"non-overlapping"``; Ljung-Box on sliding means may be misleading.
                * **alpha** (*float*) -- Ljung-Box significance level.  Default ``0.05``.
                * **lag_set** (*tuple of int*) -- Lags passed to ``_ljung_box_pass``.  Pass condition: ``p > alpha``
                  for **all** tested lags.  Lags are capped to ``n_blocks - 1`` and
                  deduplicated automatically.  Default ``(5, 10)``.
                * **B_min** (*int*) -- Soft starting-window cap.  The seed window is capped so that at
                  least *B_min* blocks are available at the start of the search.
                  Matches the ``B_min`` parameter of
                  ``DataStream._autotune_window_size``.  Default ``15``.
                * **min_blocks** (*int*) -- Hard stop.  The loop terminates immediately (without a LB test) if
                  the block count drops below *min_blocks*.  The best window found so
                  far is returned.  Default ``2`` (matches the old DataStream
                  behaviour of continuing until fewer than 2 blocks remain).
                * **max_iter** (*int*) -- Maximum window-increment iterations.  Default ``25``.
                * **w_min** (*int*) -- Minimum allowed window size.  Default ``5``.
                * **c0** (*float*) -- Multiplier for tau_int: ``w0 = ceil(c0 * tau_int)``.  Default ``2.0``.

   :returns: *dict* -- Keys:

             ``blocks`` : np.ndarray
                 Final block-mean values (may be empty if no valid blocks).
             ``window_size`` : int
                 Chosen window size.
             ``n_blocks`` : int
                 Number of block means.
             ``independence_status`` : str
                 ``"independent"``, ``"best_p"``, ``"user_window"``, or
                 ``"too_few_blocks"``.
             ``independent`` : bool
                 ``True`` iff status is ``"independent"``.
             ``ljungbox_lags`` : list[int]
                 Lags actually tested at the chosen window.
             ``ljungbox_pvalues`` : list[float]
                 P-values at each tested lag.
             ``best_pvalue`` : float
                 Minimum p-value across lags at the chosen window, or NaN.
             ``tau_int`` : float
                 Estimated tau_int (NaN for ``user_window`` path).
             ``initial_window`` : int
                 Seed window before iteration starts.
             ``iterations`` : int
                 Loop iterations used.
             ``autotuned`` : bool
                 ``False`` for ``user_window`` path.
             ``warning`` : str or None
                 Diagnostic message, or ``None`` when autotuning succeeded.


