"""Provenance must be checkout-independent: line endings and linked model directories may not change keys or hashes."""
import subprocess, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as pv  # noqa: E402


def test_crlf_and_lf_text_hash_equal(tmp_path):
    lf, crlf = tmp_path / 'lf.sv', tmp_path / 'crlf.sv'
    lf.write_bytes(b'module m;\nendmodule\n')
    crlf.write_bytes(b'module m;\r\nendmodule\r\n')
    assert pv.sha256_file(lf) == pv.sha256_file(crlf)


def test_text_content_change_still_detected(tmp_path):
    a, b = tmp_path / 'a.sv', tmp_path / 'b.sv'
    a.write_bytes(b'assign x = 1;\r\n')
    b.write_bytes(b'assign x = 2;\r\n')
    assert pv.sha256_file(a) != pv.sha256_file(b)


def test_binary_hashed_raw(tmp_path):
    a, b = tmp_path / 'a.bin', tmp_path / 'b.bin'
    a.write_bytes(b'\0\x01\r\n\x02')
    b.write_bytes(b'\0\x01\n\x02')
    assert pv.sha256_file(a) != pv.sha256_file(b)


def test_linked_dir_keeps_repo_relative_key(tmp_path):
    target = tmp_path / 'models_target'
    target.mkdir()
    (target / 'w.bin').write_bytes(b'\0weights')
    link = pv.ROOT / 'build' / '_provenance_link_test'
    link.parent.mkdir(exist_ok=True)
    if link.exists():
        link.rmdir()
    made = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], capture_output=True) if sys.platform == 'win32' else None
    if made is None or made.returncode != 0:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip('cannot create a directory link here')
    try:
        assert pv.rel(link / 'w.bin') == 'build/_provenance_link_test/w.bin'
        assert pv.expand(['build/_provenance_link_test/w.bin']) == ['build/_provenance_link_test/w.bin']
    finally:
        link.rmdir() if sys.platform == 'win32' else link.unlink()   # removes the link only, never the target's files
