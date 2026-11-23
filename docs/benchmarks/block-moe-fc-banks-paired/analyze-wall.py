import json,pathlib,statistics
p=pathlib.Path('/tmp/nxrt-bank-paired');r=json.loads((p/'runs.json').read_text());s=json.loads((p/'summary.json').read_text())
for c in s:
 for proc in c['processes']:
  row=next(x for x in r if x['shape']==c['shape'] and x['rotation']==proc['rotation']);a={x['arm']:x for x in row['arms']}
  for label in ['banked','null']:
   proc[f'{label}_wall_over_before']=statistics.median(x/y for x,y in zip(a[label]['wall_batch_ms'],a['before']['wall_batch_ms']))
 c['median_banked_wall_ratio']=statistics.median(x['banked_wall_over_before'] for x in c['processes']);c['median_null_wall_ratio']=statistics.median(x['null_wall_over_before'] for x in c['processes'])
(p/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
