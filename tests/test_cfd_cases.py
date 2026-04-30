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
            self.assertEqual(
                [case.path.name for case in cases],
                ["Case-150RAD-SOLUTION_FIELDS", "Case-350RAD-SOLUTION_FIELDS"],
            )

    def test_resolve_input_dir_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"

            with self.assertRaises(FileNotFoundError):
                resolve_input_dir(missing)


if __name__ == "__main__":
    unittest.main()
