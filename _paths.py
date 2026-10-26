"""Filesystem path resolution shared across SDBench.

This is a low-level, dependency-free leaf module (stdlib only) so it can be imported by
both ``config`` and ``case_data`` without creating import cycles. It resolves two things
from the environment, on every call (not cached at import time) so values populated from
a ``.env`` file loaded after import are honoured:

``sdbench_data_path()`` — root directory for SDBench data (case datasets and costing
data), controlled by ``SDBENCH_DATA_PATH`` (default: ``data`` relative to the CWD).

``local_overrides_base()`` — base directory for user-supplied overrides (custom prompts,
diagnostic agents, and a ``config.py``), controlled by ``SDBENCH_LOCAL_DIR``:

* unset            → the current working directory (the default for direct repo use);
* a path           → that directory (e.g. when running from another project);
* ``disable``      → no local overrides at all (also accepts ``none`` / ``off``); only
                     the packaged built-ins under ``sdbench/`` are used.

The expected layout under the local-override base directory is::

    <base>/prompts/<prompt_id>/<version>/prompt.yaml   # prompt overrides
    <base>/diagnostic_agents/*.py                      # agent overrides
    <base>/config.py                                   # config override
"""

import os
import pathlib

_DATA_PATH_ENV = "SDBENCH_DATA_PATH"
_LOCAL_DIR_ENV = "SDBENCH_LOCAL_DIR"
_DISABLE_VALUES = {"disable", "none", "off"}


def sdbench_data_path() -> str:
    """Root directory for SDBench data (case datasets and costing data).

    Controlled by ``SDBENCH_DATA_PATH``; defaults to ``data`` relative to the current
    working directory.
    """
    return os.environ.get(_DATA_PATH_ENV, "data")


def local_overrides_base() -> pathlib.Path | None:
    """Return the base directory for local overrides, or ``None`` if disabled.

    Returns ``None`` when ``SDBENCH_LOCAL_DIR`` is set to a disabling sentinel
    (``disable`` / ``none`` / ``off``). When unset, returns the current working
    directory. Otherwise returns the configured directory (resolved).
    """
    value = os.environ.get(_LOCAL_DIR_ENV)
    if value is None or value == "":
        return pathlib.Path.cwd()
    if value.strip().lower() in _DISABLE_VALUES:
        return None
    return pathlib.Path(value).expanduser().resolve()
