#!/bin/bash
# vast.ai 인스턴스 명령 실행기. onstart가 /workspace/runner.sh로 복사해 띄운다.
# 이 레포(공개)의 작업 브랜치를 주기적으로 받아, vast/cmd.sh의 "# id:" 값이 바뀌면 그 파일을 한 번 실행한다.
# 출력은 컨테이너 로그로 보낸다 (vast.ai 로그 API로 읽음). 각 줄 앞에 "RLTDP#<id> "를 붙이고,
# 끝에 "RLTDP#<id> __exit <코드>"를 쓴다. 같은 내용은 /workspace/cmdlogs/<id>.log에도 남는다.
B=claude/busy-goldberg-392gtg
CTL=/workspace/ctl
LOGD=/workspace/cmdlogs
OUT=/proc/1/fd/1
( echo "RLTDP#runner probe" > "$OUT" ) 2>/dev/null || OUT=/dev/stdout
mkdir -p "$LOGD"
last=$(cat /workspace/.last_id 2>/dev/null || echo none)
echo "RLTDP#runner start last=$last out=$OUT" > "$OUT"
while true; do
  if git -C "$CTL" fetch -q origin "$B" 2>/dev/null && git -C "$CTL" reset -q --hard FETCH_HEAD; then
    id=$(sed -n 's/^# id: *//p' "$CTL/vast/cmd.sh" 2>/dev/null | head -1 | tr -cd 'A-Za-z0-9._-')
    if [ -n "$id" ] && [ "$id" != "$last" ]; then
      last=$id
      echo "$id" > /workspace/.last_id
      cp "$CTL/vast/cmd.sh" "$LOGD/$id.sh"
      ( cd /workspace && { bash "$LOGD/$id.sh"; echo "__exit $?"; } 2>&1 \
          | tee "$LOGD/$id.log" | sed -u "s/^/RLTDP#$id /" > "$OUT" ) &
    fi
  else
    echo "RLTDP#runner fetch failed $(date -u +%H:%M:%S)" > "$OUT"
  fi
  sleep 20
done
