# 실시간 배드민턴 VAR 엔진

앱 프로젝트 안에 직접 넣는판독 엔진입니다.

```text
휴대폰 A/B 카메라
  → TrackNetV3: 프레임별 셔틀콕 x, y, confidence
  → json_runtime.py 또는 LiveVarSession
  → IN / OUT / UNKNOWN + 최근 1초 TOUCH 후보 JSON
  → 앱 화면 표시
```

 `TOUCH`는 셔틀콕 궤적이 급격히 바뀌었을 때만 나오는 추정 판정입니다.

## 앱에 넣을 코드

| 파일 | 앱 개발자가 쓰는 목적 |
|---|---|
| `realtime_engine.py` | 세션 상태, 코트 보정, IN/OUT·터치 후보 판정 핵심 |
| `json_runtime.py` | JSON Lines 입력/출력으로 엔진을 계속 실행하는 브리지 |
| `tracknet_worker.py` | 공식 TrackNetV3 체크포인트 자동 다운로드·동영상 청크 추론 |
| `model_runtime.py` | TrackNet 체크포인트 캐시와 추론 실행 |
| `var_core.py` | 좌표 변환, 낙하·궤적 이벤트 계산 |

앱이 Python 코드를 직접 호출할 수 있으면 `LiveVarSession`을 import하면 됩니다. 다른 언어로 만든 앱이면 `json_runtime.py`를 자식 프로세스로 한 번 실행하고, 표준입력·표준출력으로 한 줄씩 JSON을 주고받으면 됩니다.

## 설치

GPU 서버/로컬 백엔드에 CUDA 버전에 맞는 PyTorch를 먼저 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\\Scripts\\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126  # 환경에 맞게 변경
pip install -r requirements.txt
```

TrackNetV3의 공개 가중치는 처음 `video_chunk`를 처리할 때 `.model-cache/`에 자동 저장됩니다. 공식 TrackNetV3 저장소는 코드와 체크포인트를 MIT 라이선스로 공개합니다. [TrackNetV3 공식 저장소](https://github.com/qaz812345/TrackNetV3)

## 실시간 권장 구조

앱은 두 카메라의 프레임 시간을 맞추고 같은 시각의 TrackNet 결과를 `frame_pair`로 엔진에 전달합니다. 이 경로가 진짜 실시간 구현에 적합합니다.

```text
각 카메라 프레임 → TrackNet 추론 → 동일 시각 좌표 매칭 → frame_pair JSON → verdict JSON
```

TrackNet의 추론 결과 하나는 다음 형식입니다.

```json
{"x": 945.2, "y": 412.8, "confidence": 0.94}
```

## JSON Lines 계약

`python json_runtime.py`를 시작한 뒤, 앱은 아래처럼 한 줄을 보내고 한 줄의 응답을 받습니다.

### 1. 경기 세션 시작

```json
{
  "action": "start",
  "session_id": "game-001",
  "camera_a_points": [[164,774],[1758,774],[1432,202],[408,202]],
  "camera_b_points": [[153,743],[1771,743],[1490,221],[374,221]],
  "court": {"width_m": 6.1, "length_m": 13.4}
}
```

네 점 순서는 항상 아래와 같습니다.

```text
가까운 왼쪽 → 가까운 오른쪽 → 먼 오른쪽 → 먼 왼쪽
```

응답:

```json
{"type":"session_started","session_id":"game-001","model_input":"synchronized TrackNet shuttlecock coordinates"}
```

### 2. 실시간 프레임 쌍 전달

```json
{
  "action": "frame_pair",
  "session_id": "game-001",
  "time_ms": 48500,
  "camera_a": {"x": 945.2, "y": 412.8, "confidence": 0.94},
  "camera_b": {"x": 1005.1, "y": 396.4, "confidence": 0.91}
}
```

응답:

```json
{
  "type": "verdict",
  "session_id": "game-001",
  "time_ms": 48500,
  "in_out": "OUT",
  "touch_last_1s": true,
  "tracking": {"camera_a_detected": true, "camera_b_detected": true, "confidence": 0.91},
  "new_events": []
}
```

`in_out`은 낙하 판정이 확정되기 전에는 `PENDING`, 라인 근처·증거 부족이면 `UNKNOWN`입니다. `new_events`에 새 `LANDING` 또는 `TOUCH` 후보가 들어오면 그 시점에 앱 화면을 갱신하면 됩니다.

### 3. 세션 종료

```json
{"action":"end","session_id":"game-001"}
```

종료 응답에는 해당 세션의 이벤트 목록이 포함됩니다.

## 동영상 청크 방식: 빠른 프로토타입용

앱에서 TrackNet 프레임 추론을 바로 연결하기 전에는 `video_chunk`를 쓸 수 있습니다. 앱이 두 카메라의 0.5~1초짜리 로컬 MP4 청크를 만들고 다음 JSON을 보내면, 엔진이 TrackNet을 자동 실행합니다.

```json
{
  "action": "video_chunk",
  "session_id": "game-001",
  "chunk_start_ms": 48000,
  "camera_a_video_path": "C:/app-cache/a_48000.mp4",
  "camera_b_video_path": "C:/app-cache/b_48000.mp4"
}
```

이 방식은 TrackNet 프로세스가 청크마다 시작되므로 진짜 실시간용으로는 비효율적입니다. 최종 앱은 모델을 메모리에 한 번만 올린 뒤 `frame_pair` 경로를 사용해야 합니다. 청크 방식은 촬영·입력·출력 연결을 빠르게 시험하는 용도입니다.

## Python에서 직접 쓰기

```python
from realtime_engine import LiveVarSession

session = LiveVarSession(
    "game-001",
    [[164,774], [1758,774], [1432,202], [408,202]],
    [[153,743], [1771,743], [1490,221], [374,221]],
)

result = session.ingest_frame_pair(
    48500,
    {"x": 945.2, "y": 412.8, "confidence": 0.94},
    {"x": 1005.1, "y": 396.4, "confidence": 0.91},
)
```

## 개발용 코트 클릭 도구

테스트 영상에서 네 점 좌표를 구할 때 사용합니다. 실제 앱에서는 사용자가 화면에서 누른 픽셀 좌표를 위 JSON에 바로 넣으면 됩니다.

```bash
python select_court_points.py --config config.json --camera a
python select_court_points.py --config config.json --camera b
```


