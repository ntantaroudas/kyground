"""List open generation batches (no output file yet): batch kind n lines lang grounding."""
import json, sys
from pathlib import Path
b = Path('runs/session/batches')
ms = [json.loads(l) for l in (b / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
kind = sys.argv[1] if len(sys.argv) > 1 else None
for m in ms:
    if (b / f"{m['batch']}.out.jsonl").exists():
        continue
    if kind and m['kind'] != kind:
        continue
    head = (b / f"{m['batch']}.md").read_text(encoding='utf-8').split('## SYSTEM PROMPT', 1)[1][:3000]
    lang = 'en' if 'Interface language: English' in head else 'el'
    g = ('full' if 'KNOWLEDGE_BASE' in head else 'rag' if 'CONTEXT' in head else 'rewrite' if m['kind'] == 'rewrite'
         else 'tools' if 'kythera' in head else 'none')
    print(m['batch'], m['kind'], len(m['ids']), m['lines'], lang, g)
