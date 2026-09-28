"""Errors shared by the feature builders."""


class InsufficientData(ValueError):
    """Not enough point-in-time history to build a snapshot."""
