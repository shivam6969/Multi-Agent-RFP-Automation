from __future__ import annotations

import unittest
from pathlib import Path

import settings


class SettingsTests(unittest.TestCase):
    def test_data_files_exist(self) -> None:
        for path in (
            settings.BU_SMALL_CHUNKS_PATH,
            settings.BU_LARGE_CHUNKS_PATH,
            settings.BU_JSON_PATH,
            settings.RFP_PDF_PATH,
        ):
            self.assertTrue(Path(path).is_file(), f"missing data file: {path}")

    def test_paths_do_not_depend_on_working_directory(self) -> None:
        self.assertTrue(Path(settings.BU_JSON_PATH).is_absolute())

    def test_no_api_keys_in_source(self) -> None:
        source = Path(settings.__file__).read_text(encoding="utf-8")
        self.assertNotIn("sk-or-", source)
        self.assertNotIn("AIza", source)


if __name__ == "__main__":
    unittest.main()
