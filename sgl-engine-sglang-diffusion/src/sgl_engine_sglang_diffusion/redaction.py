from __future__ import annotations

from collections.abc import Mapping


_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "KEY")
_REDACTED = "<redacted>"


def _contains_secret_marker(value: str) -> bool:
    normalized = value.upper().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_MARKERS)


def redact_argv(argv: list[str]) -> list[str]:
    """Redact values attached to credential-looking command-line options."""
    redacted: list[str] = []
    hide_next = False
    for argument in argv:
        if hide_next:
            redacted.append(_REDACTED)
            hide_next = False
            continue
        if "=" in argument:
            key, _ = argument.split("=", 1)
            redacted.append(
                f"{key}={_REDACTED}" if _contains_secret_marker(key) else argument
            )
            continue
        redacted.append(argument)
        if argument.startswith("-") and _contains_secret_marker(argument):
            hide_next = True
    return redacted


def redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return receipt-safe environment overrides."""
    return {
        key: _REDACTED if _contains_secret_marker(key) else value
        for key, value in sorted(environment.items())
    }
