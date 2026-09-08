import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photo_sorter.app_paths import application_dir
from photo_sorter.engine import PhotoSorter


class GpuPackagingTests(unittest.TestCase):
    def test_gpu_edition_shares_parent_data_only_in_designated_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for relative, expected in [
                ("GPU版/MidnaUdon EventPhotoSorter_GPU.exe", root),
                ("MidnaUdon EventPhotoSorter.exe", root),
                ("standalone/MidnaUdon EventPhotoSorter_GPU.exe", root / "standalone"),
                ("GPU版/LinkPhotoSorter_GPU.exe", root),
                ("LinkPhotoSorter.exe", root),
                ("standalone/LinkPhotoSorter_GPU.exe", root / "standalone"),
            ]:
                with patch.object(sys, "frozen", True, create=True), patch.object(sys, "executable", str(root / relative)):
                    self.assertEqual(application_dir(), expected)

    def test_device_summary_does_not_import_torch_for_api_only(self):
        sorter = PhotoSorter.__new__(PhotoSorter)
        self.assertEqual(sorter._local_devices(), {})


if __name__ == "__main__":
    unittest.main()
