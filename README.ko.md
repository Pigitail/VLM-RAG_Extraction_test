# vlm-rag-extraction

[English](README.md)

VLM으로 도면 PDF, 매뉴얼 슬라이드(PPTX), 사양서(DOCX)를 RAG 청크로 만드는 기법 모음.

사내 RAG 적재 파이프라인에서 쓰던 기법을 일반 명칭·합성 문서 기준으로 재구현.
원본 데이터 없이 실행·테스트·측정 가능.

## 기법

| 문서 | 문제 | VLM 역할 |
|---|---|---|
| 도면 PDF ([문서](docs/techniques/drawing.md)) | 표제란 글자가 작고 이미지로만 있는 경우도 있음, 텍스트 레이어 순서도 뒤섞임 | 겹치는 타일 + 확대 crop으로 표제란 판독. 텍스트 레이어는 힌트로만 전달 |
| 매뉴얼 슬라이드 PPTX ([문서](docs/techniques/slides.md)) | 의미는 스크린샷에 있는데 텍스트 추출로는 주변 글만 나옴 | 슬라이드 전체 렌더 기준으로 그림별 역할 + 설명 + 짧은 영/한 캡션. 전체 OCR 안 함 |
| 사양서 DOCX ([문서](docs/techniques/docx.md)) | 청크에 실제 페이지 번호 필요, 긴 표가 페이지를 넘어감 | 페이지별 그림 설명. 여러 페이지 표는 "어느 행이 몇 페이지"만 답함 |
| 공통 ([문서](docs/techniques/common.md)) | 문서 유형마다 같은 처리 반복 | - |

공통 부분: `retrieve_content` / `llm_content` / `filtering_content` 3종 청크 포맷, 검색 요약 prefix(선택),
VLM backend(OpenAI, Azure OpenAI, vLLM 같은 OpenAI 호환 서버), 동시 호출 제한, retry, JSON 복구,
응답 캐시, 토큰 집계, record/replay.

## 원칙

- 파일에 있는 텍스트(PDF 텍스트 레이어, DOCX/PPTX XML)는 청크에 그대로 사용. 모델 출력은 옆에 추가만,
  대체 없음
- VLM 담당은 이 텍스트로 안 되는 부분: 그림 내용, 도면 표제란(텍스트 레이어 없음·순서 뒤섞임),
  표 행의 페이지 배정
- VLM 응답은 사용 전 검증. 잘못된 응답은 정해진 fallback으로 처리하고 청크에 기록 (`provenance.fallbacks`).
  도면은 표제란을 못 읽으면(vision 응답 없음, 항목 3개 미만) 파일 실패 처리
- `Chunk.page` = 원본 파일 페이지(슬라이드) 번호. 병합·가상 페이지, 번호 재부여 없음
- 이미지 설명은 `retrieve_content`에 포함 (metadata는 임베딩 대상이 아니라 검색에 안 걸림)
- 실패 파일은 `<out>.errors.jsonl`에 기록, 나머지 배치는 계속 진행

## 결과

합성 문서(도면 3개 / 4장, 슬라이드 1개 / 6장, DOCX 1개 / 4페이지), Azure OpenAI `gpt-6-luna`,
reasoning effort `low` (DOCX 표 배정만 `medium`), 2026-11-19 1회 실행.
기준선 = 같은 파이프라인에서 VLM 응답만 빈 값으로 처리 (모델 호출 없음).

| 항목 | VLM | 기준선 |
|---|---|---|
| 도면 표제란 항목 | 65/68 | 0/68 |
| 표제란이 이미지로만 있는 도면 (항목) | 17/17 | 0/17 |
| 슬라이드 그림 역할 (메인 스크린샷 / 확대 crop / 아이콘 등) | 7/7 | 0/7 |
| DOCX 표 행 페이지 배정 | 44/44 | 44/44 |
| DOCX 그림 내용이 청크 텍스트에 들어간 비율 | 6/8 | 2/8 |
| 검색 Recall@1 (BM25, 질의 18개, 페이지 단위) | 83.3% | 38.9% |
| 검색 Recall@5 | 100% | 83.3% |
| 검색 MRR | 0.917 | 0.563 |

토큰: 도면 입력 333,491 / 출력 6,900 (3개 파일, 9회 호출), 슬라이드 22,046 / 1,289 (4회),
DOCX 31,456 / 746 (4회).

해석 시 참고:

- 합성 문서 소량, 1회 실행 결과. 항목·질의 하나로 수치가 몇 %p씩 움직임
- 도면 65/68이 이 데이터의 최대치. 한 도면의 2장째는 1장째 표제란을 그대로 씀
  (기본값 `title_block_pages="first"`) → 3개 항목은 설계상 다름
- 기준선 도면은 표제란 gate에서 실패 (빈 응답). 검색 평가에서는 모델 출력만 뺀 도면 청크
  (파일명, 텍스트 레이어, 거기서 뽑은 항목)로 대체 → 검색 차이 = 모델 출력 차이
- 슬라이드 "꼭 나올 내용" 1/6. 정답이 스크린샷 속 문자열(`V-2864`, `Overdue orders: 4`)인데 프롬프트가
  화면 받아 적기 대신 화면 설명을 요구 (전체 OCR 금지) → 대부분 문자열 그대로는 안 나옴.
  잘못 읽은 게 아니라 이 규칙의 비용
- DOCX 놓친 2건은 표현 차이 ("carry hook" → "hanging loop", "green status ring" → "green collar").
  문자열 포함 여부로 채점
- 합성 DOCX는 페이지 경계 표시가 정확해서 기준선도 행 배정을 다 맞춤. 44/44는 VLM 경로가 망가뜨리지
  않는다는 뜻이지, 틀린 경계 표시를 고친다는 뜻은 아님

전체 표와 지표 정의: [examples/recordings/gpt-6-luna/results.md](examples/recordings/gpt-6-luna/results.md),
[docs/evaluation.md](docs/evaluation.md)

## 실행

Python 3.10+. 슬라이드·DOCX는 페이지 렌더에 LibreOffice 필요 (replay 때도 필요, 도면은 불필요).

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 합성 문서 + 정답 생성
python -m vlmrag.synthetic --out work/synthetic

# API 키 없이 위 결과 재현 (녹화된 모델 응답 사용)
python -m vlmrag.evaluation --replay-dir examples/recordings/gpt-6-luna --out-dir out/eval-replay
```

replay는 이미지 픽셀이 아니라 요청 텍스트와 이미지 라벨로 응답을 찾음. pypdfium2나 LibreOffice 버전이
다르면 요청이 달라질 수 있고, 그때는 추측하지 않고 miss로 보고함.

모델 사용:

```bash
cp .env.example .env            # provider, model, key 입력
vlmrag run drawing work/synthetic/drawing --out out/drawing.jsonl
vlmrag run slides  work/synthetic/slides  --out out/slides.jsonl
vlmrag run docx    work/synthetic/docx    --out out/docx.jsonl
```

`--record FILE`로 모델 응답 저장, `--replay FILE`로 네트워크 없이 재실행,
`--provider null`로 빈 응답 기준 전체 흐름(fallback 경로) 확인 (도면은 이때 표제란 gate에서 실패하는 게 정상).

데모: `python examples/drawing_demo.py --provider null` (`slides_demo.py`, `docx_demo.py`도 동일).
`--provider null` 없이 실행하면 설정된 모델 호출. `drawing_demo.py --simulate`는 정답 파일로 응답.

## 시행착오

- **이미지 크기 상한.** 원본은 확대 crop에 상한 없음 (A3 메타데이터 표 전체 이미지가 약 280MP).
  `gpt-6-luna`는 이 요청을 거부 (이미지 patch 한도 초과). 같은 도면 3개로 따로 돌려본 결과 4MP가 8MP와
  같은 항목 수, 16MP보다 2개 더 맞힘. 입력 토큰은 16MP의 28% → 기본값 4MP
  (`VLMRAG_DRAWING_MAX_IMAGE_PIXELS`, 0 = 상한 없음)

  | 상한 | 항목 | 입력 토큰 (도면 3개) |
  |---|---|---|
  | 없음 | 거부 | - |
  | 16MP | 63/68 | 1,189,152 |
  | 8MP | 65/68 | 643,750 |
  | 4MP | 65/68 | 333,334 |

- **요약 prefix 부작용.** 숫자 표에 모델 요약을 붙였더니 원본에 없는 숫자(반올림·조합)가 들어가고
  임베딩 모델 8,192 토큰 한도를 상당 부분 차지 (실제 표 행이 밀려남) → 문서 유형별 선택
  (`VLMRAG_RETRIEVAL_SUMMARY`), 위 결과에는 미적용
- **복사 말고 배정.** 여러 페이지 표는 모델이 셀 내용 없이 페이지별 행 번호만 답함. 틀린 답은 검증
  (모든 행 정확히 1번, 페이지 범위, 페이지 순서)으로 쉽게 걸러지고 통째로 버림. 표 내용은 모델이 바꿀 수 없음
- **모듈 레벨 `multiprocessing.Semaphore`** 는 `fork`에서만 프로세스 간 공유됨. `spawn` / `forkserver`
  에서는 프로세스마다 따로 생김 → 제한은 `VLMClient`가 갖고, VLM 호출은 thread로 처리
- **LibreOffice** 병렬 실행 시 호출마다 별도 프로필 필요, timeout 시 프로세스 트리 전체 종료,
  Windows에서는 프로필 경로가 길면 죽음 → 짧은 경로 사용

## 구조

```
src/vlmrag/
  techniques/{drawing,slides,docx}/   기법 3종
  vlm/                                요청 타입, backend, client, JSON 복구, 캐시, 토큰 집계
  render/                             PDF 렌더 (pypdfium2), 타일, LibreOffice 변환
  augment/                            검색 요약 prefix (선택)
  synthetic/                          합성 문서 + 정답
  evaluation/                         지표, BM25, record/replay 실행기
examples/                             데모, 녹화된 모델 응답
configs/                              표제란 스키마·도메인 용어 예시
docs/                                 기법 설명, 평가 방법
scripts/release_check.py              공개 전 검사 (키, 내부 주소, 이메일, 파일 형식)
```

## 라이선스

MIT
