import collections
import json
import os
import sys

BIN = int(os.environ.get('BIN', 100000))
for L in sys.argv[1:]:
  print('==', os.path.basename(L))
  try:
    s = [json.loads(l) for l in open(L + '/scores.jsonl')]
    b = collections.defaultdict(lambda: [0, 0])
    for x in s:
      k = x['step'] // BIN
      b[k][0] += 1
      b[k][1] += x['episode/score'] > 0
    print('episodes', len(s), 'last step', s[-1]['step'])
    print('  success/episodes per', BIN, 'steps:',
          {k * BIN // 1000: f'{v[1]}/{v[0]}' for k, v in sorted(b.items())})
  except Exception as e:
    print('  scores:', e)
  try:
    m = [json.loads(l) for l in open(L + '/metrics.jsonl')]
    r = [r for r in m if 'fps/policy' in r][-1]
    print('  step', r['step'], {k: round(v, 4) for k, v in r.items()
           if k.startswith(('train/qimag', 'fps', 'train/rand'))})
  except Exception as e:
    print('  metrics:', e)
  q = L + '/qstats.jsonl'
  if os.path.exists(q):
    lines = open(q).read().splitlines()
    if lines:
      print('  qstats', lines[-1])
  out = L + '.out'
  if os.path.exists(out):
    t = open(out).read()
    if 'Traceback' in t:
      print('  TRACEBACK in', out)
