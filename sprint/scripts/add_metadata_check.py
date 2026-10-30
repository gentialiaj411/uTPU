from pathlib import Path
p=Path('sprint/scripts/engineering_driver.py');s=p.read_text();s=s.replace("if not rc:\n  os.environ['UTPU_SIM']", "if not rc:\n  rc=run([sys.executable,'-m','pytest','sprint/scripts/test_compact_metadata.py','-q'],'metadata_python_exactness',180)\n if not rc:\n  os.environ['UTPU_SIM']")
p.write_text(s)
