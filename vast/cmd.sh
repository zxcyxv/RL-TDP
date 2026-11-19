# id: 5
# vast.ai 인스턴스에서 한 번 실행할 명령. id를 바꿔 푸시하면 runner.sh가 실행한다.
python3 -c "
import json; d=json.load(open('/workspace/runs/stage1_full.json')); s=json.dumps(d,separators=(',',':'))
print('CHUNKS', (len(s)+299)//300, len(s))
[print('C%04d %s' % (i, s[i*300:(i+1)*300])) for i in range((len(s)+299)//300)]
"
