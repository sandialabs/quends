# Tutorial Examples

This directory contains runnable QUENDS tutorials.

- `scripts/`: Python scripts used by the documentation gallery.
- `notebooks/`: Jupyter notebooks for interactive walkthroughs.

Both formats use the shared datasets in `examples/data`. Notebook paths are
written relative to this directory layout, so run notebooks with
`examples/tutorial/notebooks` as the working directory. The scripts derive the
data directory from their own file location and can be run from any working
directory.

The old `examples/notebooks` and `examples/tutorial/data` trees were removed to
avoid duplicated data and stale paths.
