"""Top-level package for shared RL algorithm utilities."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rl-algs")
except PackageNotFoundError:  # pragma: no cover - when running from source
    __version__ = "0.0.0"

__all__ = ["__version__"]
