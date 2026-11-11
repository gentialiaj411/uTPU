import shutil,os
from pathlib import Path
root=Path('/tmp/utpu-perf');assert root.resolve()==Path('/tmp/utpu-perf')
active=[]
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:active.append((proc/'exe').resolve())
  except OSError:pass
for child in root.iterdir():
 if child.is_dir() and child.resolve().parent==root.resolve() and child.name.endswith('_obj'):
  if not any(child.resolve() in exe.parents for exe in active):shutil.rmtree(child)
print('SUMMARY cleaned completed owned build directories')
