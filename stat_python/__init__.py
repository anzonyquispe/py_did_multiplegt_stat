"""
did_multiplegt_stat: Python implementation of de Chaisemartin & D'Haultfeuille (2024)
Difference-in-Differences estimator for staggered adoption designs.

This package provides both a functional API (backward compatible) and a
scikit-learn style class API for DiD estimation.

Example usage (class API):
    >>> from stat_python import DIDMultiplegtStat
    >>> model = DIDMultiplegtStat(estimator=['aoss', 'waoss'])
    >>> model.fit(df, Y='outcome', ID='unit_id', Time='time', D='treatment')
    >>> model.summary()
    >>> model.plot()

Example usage (functional API):
    >>> from stat_python import did_multiplegt_stat
    >>> results = did_multiplegt_stat(df, Y='outcome', ID='unit_id',
    ...                               Time='time', D='treatment')
"""

from .estimator import DIDMultiplegtStat
from ._core import did_multiplegt_stat, summary_did_multiplegt_stat, print_did_multiplegt_stat

__version__ = "0.1.0"
__all__ = [
    "DIDMultiplegtStat",
    "did_multiplegt_stat",
    "summary_did_multiplegt_stat",
    "print_did_multiplegt_stat",
]
