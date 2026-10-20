"""Russian names in ZIP archives packed without the UTF-8 flag and without the Unicode Path field."""
import io
import tempfile
import unittest
from pathlib import Path

from dxaqc.api import analyze
from dxaqc.inputs import entry_name, input_directory
from test_release_cli import SyntheticPredictor, synthetic_dicom, zip_bytes


def zip_with_raw_name(raw_name: bytes, content: bytes) -> bytes:
    """ZIP, где имя записано сырыми байтами без флага UTF-8 и без поля 0x7075 (как у zip на macOS или Windows)."""
    import zipfile
    placeholder = b'N' * len(raw_name)
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(placeholder.decode('ascii'), content)
    data = out.getvalue()
    assert data.count(placeholder) == 2, 'имя должно встретиться в локальном заголовке и в каталоге'
    return data.replace(placeholder, raw_name)


class ZipNameEncodingTests(unittest.TestCase):
    NAME = 'Для теста/CR000000_ППОБ.dcm'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dicom = synthetic_dicom(self.root / 'src.dcm').read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def _entry(self, data):
        import zipfile
        return zipfile.ZipFile(io.BytesIO(data)).infolist()[0]

    def test_utf8_bytes_without_flag_are_recovered(self):
        entry = self._entry(zip_with_raw_name(self.NAME.encode('utf-8'), self.dicom))
        self.assertFalse(entry.flag_bits & 0x800)
        self.assertNotEqual(entry.filename, self.NAME)          # без исправления были бы кракозябры
        self.assertEqual(entry_name(entry), self.NAME)

    def test_cp866_bytes_without_flag_are_recovered(self):
        entry = self._entry(zip_with_raw_name(self.NAME.encode('cp866'), self.dicom))
        self.assertEqual(entry_name(entry), self.NAME)

    def test_archive_with_utf8_flag_is_untouched(self):
        data = zip_bytes([(self.NAME, self.dicom)])             # zipfile сам ставит флаг UTF-8
        entry = self._entry(data)
        self.assertTrue(entry.flag_bits & 0x800)
        self.assertEqual(entry_name(entry), self.NAME)

    def test_ascii_name_is_untouched(self):
        entry = self._entry(zip_with_raw_name(b'study/CR000001.dcm', self.dicom))
        self.assertEqual(entry_name(entry), 'study/CR000001.dcm')

    def test_report_path_keeps_russian_name(self):
        archive = self.root / 'in.zip'
        archive.write_bytes(zip_with_raw_name(self.NAME.encode('utf-8'), self.dicom))
        with input_directory(archive) as folder:
            self.assertTrue((folder / self.NAME).is_file())
        rows, _ = analyze(archive, predictor=SyntheticPredictor())
        self.assertEqual([r['path_to_study'] for r in rows], [self.NAME])


if __name__ == '__main__':
    unittest.main()
