# -*- coding: utf-8 -*-
"""Entry point for the h4 Toolkit custom nodes package.

Responsibilities handled here:
- Dependency validation and on-demand installation for Python packages.
- Optional update notification (no automatic upgrades).
- Import of the node classes with defensive error handling.
- Colourised startup table so toolkit status is front-and-centre in the console.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import metadata
from typing import Dict, List, Tuple

try:
    from colorama import Fore, Style, init as colorama_init
except ImportError:  # pragma: no cover - bootstrap path
    class _BlankColour:  # type: ignore
        RESET_ALL = ""

        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = Style = _BlankColour()  # type: ignore

    def colorama_init(*_args, **_kwargs):  # type: ignore
        return None

colorama_init(autoreset=True)

DEPENDENCIES: Dict[str, str] = {
    "colorama": "0.4.6",
    "numpy": "1.24.0",
    "torch": "1.13.0",
}

TOOLKIT_NAME = "h4_ToolKit"
TOOLKIT_VERSION = "1.0.0"


def _emit(message: str, colour: str = Fore.LIGHTWHITE_EX) -> None:
    print(f"{colour}[{TOOLKIT_NAME}] {message}{Style.RESET_ALL}")


def _ensure_dependency(name: str, minimum_version: str) -> None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        _emit(f"Installing missing dependency: {name}", Fore.YELLOW)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f"{name}>={minimum_version}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - installation failures logged
            _emit(f"Failed to install {name}: {exc.stderr.decode(errors='ignore')}", Fore.RED)
            return
    else:
        installed_version = metadata.version(name)
        if _needs_update(installed_version, minimum_version):
            _emit(
                f"Dependency {name} is below recommended version {minimum_version} (installed {installed_version}).",
                Fore.YELLOW,
            )
    _notify_if_newer(name)


def _needs_update(installed: str, minimum: str) -> bool:
    return _normalise_version(installed) < _normalise_version(minimum)


def _normalise_version(version: str) -> List[int]:
    parts = []
    for fragment in version.replace("-", ".").split('.'):
        if fragment.isdigit():
            parts.append(int(fragment))
        else:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return parts[:3]


def _notify_if_newer(package_name: str) -> None:
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package_name}/json", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            latest_version = payload["info"]["version"]
    except (urllib.error.URLError, KeyError, TimeoutError, ValueError):  # pragma: no cover - network best effort
        return
    try:
        installed_version = metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return
    if _normalise_version(latest_version) > _normalise_version(installed_version):
        _emit(
            f"Update available for {package_name}: installed {installed_version}, latest {latest_version}",
            Fore.CYAN,
        )


def ensure_dependencies() -> None:
    for package, minimum_version in DEPENDENCIES.items():
        _ensure_dependency(package, minimum_version)


@dataclass
class NodeStatus:
    name: str
    version: str
    status: str
    colour: str
    symbol: str


STATUS_COLOURS = {
    "ok": (Fore.GREEN, "✔"),
    "error": (Fore.RED, "✘"),
}


def _split_display_entry(display_value: str) -> Tuple[str, str]:
    """Derive human-friendly name and version from a display string."""

    name = display_value.strip()
    version = ""
    if " (v" in display_value and display_value.strip().endswith(")"):
        base, suffix = display_value.rsplit(" (v", 1)
        name = base.strip()
        version = suffix[:-1].strip()
    elif " v" in display_value:
        base, suffix = display_value.rsplit(" v", 1)
        name = base.strip()
        version = suffix.strip()
    if not version:
        version = TOOLKIT_VERSION
    return name, version


def _render_status_table(status_rows: List[NodeStatus]) -> None:
    def _measure(getter) -> int:
        return max((len(getter(row)) for row in status_rows), default=0)

    name_width = max(len("Node"), _measure(lambda row: row.name))
    version_width = max(len("Version"), _measure(lambda row: row.version))
    status_width = max(len("Status"), _measure(lambda row: row.status))
    marker_width = max(len("Mark"), _measure(lambda row: row.symbol))

    table_width = name_width + version_width + status_width + marker_width + 13  # pipes + spacing
    title = f" {TOOLKIT_NAME} v{TOOLKIT_VERSION} ".center(table_width - 2, "=")
    top_border = f"+{title}+"
    separator = (
        f"+{'-' * (name_width + 2)}+{'-' * (version_width + 2)}+"
        f"{'-' * (status_width + 2)}+{'-' * (marker_width + 2)}+"
    )
    _emit(top_border, Fore.LIGHTMAGENTA_EX)
    _emit(
        f"| {'Node':<{name_width}} | {'Version':<{version_width}} | {'Status':<{status_width}} | {'Mark':<{marker_width}} |",
        Fore.LIGHTWHITE_EX,
    )
    _emit(separator, Fore.LIGHTWHITE_EX)
    for row in status_rows:
        colour, symbol = row.colour, row.symbol
        payload = (
            f"| {row.name:<{name_width}} | {row.version:<{version_width}} | {row.status:<{status_width}} | {symbol:<{marker_width}} |"
        )
        _emit(payload, colour)
    bottom_border = f"+{'=' * (table_width - 2)}+"
    _emit(bottom_border, Fore.LIGHTBLUE_EX)


ensure_dependencies()

status_log: List[NodeStatus] = []
_exported_class_names: List[str] = []

try:
    from . import nodes as _nodes  # noqa: F401

    NODE_CLASS_MAPPINGS = _nodes.NODE_CLASS_MAPPINGS  # noqa: F401
    NODE_DISPLAY_NAME_MAPPINGS = _nodes.NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401
    NODE_TOOLTIP_MAPPINGS = _nodes.NODE_TOOLTIP_MAPPINGS  # noqa: F401

    ok_colour, ok_symbol = STATUS_COLOURS["ok"]

    for class_key, class_obj in NODE_CLASS_MAPPINGS.items():
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(class_key, class_obj.__name__)
        name, version = _split_display_entry(display_name)
        status_log.append(NodeStatus(name, version, "Loaded", ok_colour, ok_symbol))
        globals()[class_obj.__name__] = class_obj
        _exported_class_names.append(class_obj.__name__)
except Exception as exc:  # pragma: no cover - import failures should not kill startup
    err_colour, err_symbol = STATUS_COLOURS["error"]
    status_log.append(NodeStatus("h4 : The Engine (Simple Sampler+Plot)", "-", f"Failed: {exc}", err_colour, err_symbol))
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    NODE_TOOLTIP_MAPPINGS = {}

_render_status_table(status_log)

WEB_DIRECTORY = "js"

__all__ = _exported_class_names + [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_TOOLTIP_MAPPINGS",
    "WEB_DIRECTORY",
]
