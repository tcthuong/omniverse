# Auto Discover CFD Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Python CFD data scripts discover OpenFOAM cases from `input/` automatically instead of hardcoding individual case folders.

**Architecture:** Add one shared `cfd_cases.py` helper for input resolution, omega parsing, OpenFOAM case discovery, and template case selection. Update data scripts to use the helper while preserving their existing CLI defaults and output behavior.

**Tech Stack:** Python 3.14, `unittest`, NumPy, PyVista/OpenFOAM readers, existing project scripts.

---

### Task 1: Shared Case Discovery Helper

**Files:**
- Create: `tests/test_cfd_cases.py`
- Create: `cfd_cases.py`

- [ ] **Step 1: Write the failing test**

```python
import tempfile
import unittest
from pathlib import Path

from cfd_cases import discover_cases, read_omega, resolve_input_dir


class CfdCasesTests(unittest.TestCase):
    def test_read_omega_prefers_mrfproperties_over_folder_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "Case-999RAD-SOLUTION_FIELDS"
            (case_dir / "constant").mkdir(parents=True)
            (case_dir / "constant" / "MRFProperties").write_text("omega constant 123.5;")

            self.assertEqual(read_omega(case_dir), 123.5)

    def test_discover_cases_returns_sorted_case_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, omega in [
                ("Case-350RAD-SOLUTION_FIELDS", "350.0"),
                ("Case-150RAD-SOLUTION_FIELDS", "150.0"),
            ]:
                case_dir = root / name
                (case_dir / "constant").mkdir(parents=True)
                (case_dir / "case.foam").write_text("")
                (case_dir / "constant" / "MRFProperties").write_text(f"omega constant {omega};")

            cases = discover_cases(root)

            self.assertEqual([case.omega for case in cases], [150.0, 350.0])
            self.assertEqual([case.path.name for case in cases], [
                "Case-150RAD-SOLUTION_FIELDS",
                "Case-350RAD-SOLUTION_FIELDS",
            ])

    def test_resolve_input_dir_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"

            with self.assertRaises(FileNotFoundError):
                resolve_input_dir(missing)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_cfd_cases -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cfd_cases'`.

- [ ] **Step 3: Implement minimal helper**

Create `cfd_cases.py` with a frozen `CfdCase` dataclass, `resolve_input_dir`, `read_omega`, `discover_cases`, and `case_map_by_omega`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\venv\Scripts\python.exe -m unittest tests.test_cfd_cases -v`
Expected: PASS.

### Task 2: Replace Script-Level Discovery

**Files:**
- Modify: `train.py`
- Modify: `evaluate.py`
- Modify: `compare.py`
- Modify: `viz.py`
- Modify: `export_streamlines.py`

- [ ] **Step 1: Update imports**

Import shared helpers in each script that reads OpenFOAM case folders.

- [ ] **Step 2: Replace hardcoded cases and duplicate discovery**

Use `resolve_input_dir` and `discover_cases` in `train.py`, `evaluate.py`, `compare.py`, and `viz.py`. Keep `export_streamlines.py` prediction loading unchanged because it reads generated arrays, not raw OpenFOAM cases.

- [ ] **Step 3: Run syntax verification**

Run: `.\venv\Scripts\python.exe -m py_compile cfd_cases.py train.py evaluate.py viz.py export_streamlines.py compare.py omniverse_ui.py`
Expected: exit code 0.

### Task 3: Environment Verification

**Files:**
- No file changes expected.

- [ ] **Step 1: Verify installed packages import**

Run a Python import check for `torch`, `torch_scatter`, `scipy`, `pyvista`, `physicsnemo`, `torch_geometric`, `matplotlib`, and `pxr`.
Expected: all import successfully.

- [ ] **Step 2: Verify script CLIs expose input defaults**

Run: `.\venv\Scripts\python.exe train.py --write-default-config tmp_train_config.json`
Expected: config writes and includes `"input": "input"`.

Delete `tmp_train_config.json` after verification.
