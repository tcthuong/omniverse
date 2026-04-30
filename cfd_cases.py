"""Shared OpenFOAM case discovery helpers."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CfdCase:
    omega: float  # rad/s
    path: Path

    @property
    def rpm(self) -> float:
        """Rotation speed in RPM (user-facing unit)."""
        return self.omega * 30.0 / math.pi


def resolve_input_dir(input_dir: str | Path = "input", base_dir: Path | None = None) -> Path:
    """Resolve an input directory from cwd first, then relative to base_dir."""
    path = Path(input_dir).expanduser()
    candidates = [path] if path.is_absolute() else [path]
    if not path.is_absolute() and base_dir is not None:
        candidates.append(Path(base_dir) / path)

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        checked.append(str(candidate))
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(f"input directory not found; checked: {checked}")


def read_omega(case_dir: Path) -> float:
    """Read rotation speed from MRFProperties first, then from the folder name."""
    mrf = case_dir / "constant" / "MRFProperties"
    if mrf.exists():
        match = re.search(
            r"omega\s+constant\s+([0-9.eE+-]+)",
            mrf.read_text(errors="ignore"),
        )
        if match:
            return float(match.group(1))

    match = re.search(r"([0-9.]+)\s*RAD", case_dir.name, re.IGNORECASE)
    if match:
        return float(match.group(1))

    raise ValueError(f"cannot determine omega for {case_dir}")


def discover_cases(input_dir: str | Path = "input", base_dir: Path | None = None) -> list[CfdCase]:
    """Return OpenFOAM case folders under input_dir sorted by omega."""
    root = resolve_input_dir(input_dir, base_dir=base_dir)
    cases = []
    seen = set()

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir() or not (case_dir / "case.foam").exists():
            continue

        omega = read_omega(case_dir)
        if omega in seen:
            raise ValueError(f"duplicate omega {omega} in {root}")
        seen.add(omega)
        cases.append(CfdCase(omega=omega, path=case_dir))

    if not cases:
        raise FileNotFoundError(f"no OpenFOAM case folders found in {root}")

    return sorted(cases, key=lambda case: case.omega)


def case_map_by_omega(input_dir: str | Path = "input", base_dir: Path | None = None) -> dict[float, Path]:
    return {case.omega: case.path for case in discover_cases(input_dir, base_dir=base_dir)}
