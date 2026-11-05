quends.postprocessing.exporter
==============================

.. py:module:: quends.postprocessing.exporter


Attributes
----------

.. autoapisummary::

   quends.postprocessing.exporter.logger


Classes
-------

.. autoapisummary::

   quends.postprocessing.exporter.Exporter


Module Contents
---------------

.. py:data:: logger

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



