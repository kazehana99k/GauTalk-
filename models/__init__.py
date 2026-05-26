"""Namespace package for project + Spectre models."""

from pkgutil import extend_path
from pathlib import Path

__path__ = extend_path(__path__, __name__)

SPECTRE_MODELS = Path(__file__).resolve().parents[1] / "EMOCA" / "external" / "spectre" / "src" / "models"
if SPECTRE_MODELS.exists():
    spectre_path = str(SPECTRE_MODELS)
    if spectre_path not in __path__:
        __path__.append(spectre_path)








