# dataset.py
import json
import time
import random
from pathlib import Path
from typing import Sequence, Dict, Any, List, Optional
from urllib.parse import urlparse
from collections import defaultdict

import requests
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from tqdm.auto import tqdm

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36"
)

# 도메인별 마지막 요청 시각 기록 (rate limiting 용)
_DOMAIN_STATE: Dict[str, Dict[str, float]] = defaultdict(
    lambda: {"last_ts": 0.0}
)


def throttled_get(
    session: requests.Session,
    url: str,
    min_interval: float = 0.5,
    timeout: float = 15.0,
) -> requests.Response:
    """
    같은 도메인에 대해 요청 간 최소 간격(min_interval)을 보장하는 GET.
    너무 자주 치는 걸 막기 위한 클라이언트 측 rate limiting.
    """
    host = urlparse(url).netloc
    state = _DOMAIN_STATE[host]
    now = time.monotonic()
    elapsed = now - state["last_ts"]

    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    resp = session.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    state["last_ts"] = time.monotonic()
    return resp


def load_artpedia_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class ArtpediaLocalImageDataset(Dataset):
    """
    - Artpedia JSON에서 URL을 읽어서 cache_dir에 이미지를 다운로드.
    - prepare_images()를 통해 URL -> 로컬 파일로 캐시한 뒤,
      실제 학습에서는 로컬 경로만 사용.

    __getitem__:
      - "image_path": str  (로컬 이미지 파일 경로)
      - "text_v": str      (visual sentence)
      - "text_c": str      (contextual sentence)
    """

    def __init__(
        self,
        raw_data: Dict[str, Any],
        cache_dir: str | Path,
        valid_splits: Sequence[str] = ("train",),
        min_len_visual: int = 1,
        min_len_contextual: int = 1,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 원본 엔트리 (다운로드 전)
        self.entries: List[Dict[str, Any]] = []

        for _id, entry in raw_data.items():
            split = entry.get("split", "train")
            if split not in valid_splits:
                continue

            img_url = entry.get("img_url")
            vis = entry.get("visual_sentences") or []
            ctx = entry.get("contextual_sentences") or []

            if not img_url:
                continue
            if len(vis) < min_len_visual:
                continue
            if len(ctx) < min_len_contextual:
                continue

            self.entries.append(
                {
                    "id": _id,
                    "split": split,
                    "img_url": img_url,
                    "visual_sentences": vis,
                    "contextual_sentences": ctx,
                }
            )

        # 실제로 사용할 샘플 리스트 (prepare_images 후 채워짐)
        self.samples: List[Dict[str, Any]] = []
        print(
            f"[ArtpediaLocalImageDataset] splits={valid_splits} "
            f"raw usable entries: {len(self.entries)}"
        )

# 내부 유틸
    def _local_path_for(self, entry: Dict[str, Any]) -> Path:
        """
        id + URL 확장자를 이용해 캐시 파일 이름 생성.
        """
        url = entry["img_url"]
        sample_id = entry["id"]

        # URL에서 확장자 추출
        suffix = Path(url).suffix
        if not suffix:
            suffix = ".jpg"
        filename = f"{sample_id}{suffix}"
        return self.cache_dir / filename

    def _download_one(
        self,
        entry: Dict[str, Any],
        max_retries: int = 3,
        base_sleep: float = 0.5,
    ) -> Optional[Path]:
        """
        개별 URL 다운로드 + 간단한 검증.
        429(Too Many Requests) 발생 시 점진적 대기 후 재시도.
        """
        out_path = self._local_path_for(entry)
        if out_path.is_file():
            return out_path

        url = entry["img_url"]

        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=15,
                )
                resp.raise_for_status()

                # 파일 저장
                with out_path.open("wb") as f:
                    f.write(resp.content)

                # 간단한 이미지 검증
                try:
                    with Image.open(out_path) as img:
                        img.verify()
                except UnidentifiedImageError:
                    print(f"[download_one] Invalid image file for id={entry['id']} url={url}")
                    out_path.unlink(missing_ok=True)
                    return None

                return out_path

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                print(
                    f"[download_one] HTTPError id={entry['id']} "
                    f"status={status} url={url} attempt={attempt+1}/{max_retries}"
                )
                # 429는 rate limit → 조금 더 길게 기다렸다가 재시도 권장 
                if status == 429:
                    wait = (attempt + 1) * 5.0
                    print(f"  -> 429 Too Many Requests, sleep {wait:.1f}s then retry")
                    time.sleep(wait)
                    continue
            except Exception as e:
                print(
                    f"[download_one] ERROR id={entry['id']} url={url} "
                    f"attempt={attempt+1}/{max_retries}: {repr(e)}"
                )

            # 그 외 에러들: 짧게 쉬고 재시도
            time.sleep(base_sleep)

        print(f"[download_one] Failed to download id={entry['id']} url={url}")
        return None

# 공개 API: 이미지 캐시 준비
    def prepare_images(
        self,
        download: bool = True,
        max_retries: int = 3,
        base_sleep: float = 0.5,
        rank: int = 0,
    ):
        """
        - download=True: URL에서 이미지 다운 + 캐시 생성
        - download=False: 이미 캐시된 파일만 기준으로 인덱스 재구성

        분산 학습에서:
          - rank==0: download=True로 한 번만 실제 다운로드
          - 나머지 rank: download=False로 로컬 캐시만 사용
        """
        if download:
            print(f"[Rank {rank}] Downloading & caching images to {self.cache_dir} ...")
            for entry in tqdm(self.entries, disable=(rank != 0)):
                self._download_one(
                    entry,
                    max_retries=max_retries,
                    base_sleep=base_sleep,
                )

        # 캐시 디렉토리 기준으로 실제 사용 가능한 샘플만 뽑기
        self.samples = []
        for entry in self.entries:
            local_path = self._local_path_for(entry)
            if local_path.is_file():
                self.samples.append(
                    {
                        "id": entry["id"],
                        "img_path": str(local_path),
                        "visual_sentences": entry["visual_sentences"],
                        "contextual_sentences": entry["contextual_sentences"],
                    }
                )

        print(
            f"[Rank {rank}] {len(self.samples)} / {len(self.entries)} "
            f"samples have cached images in {self.cache_dir}"
        )

    # Dataset 인터페이스
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ex = self.samples[idx]
        img_path = ex["img_path"]

        # 문장은 매번 랜덤 샘플링
        import random

        t_v = random.choice(ex["visual_sentences"])
        t_c = random.choice(ex["contextual_sentences"])

        return {
            "image_path": img_path,
            "text_v": t_v,
            "text_c": t_c,
        }
    def _download_one(
        self,
        entry: Dict[str, Any],
        max_retries: int = 2,
        base_sleep: float = 1.0,
    ) -> Optional[Path]:
        """
        개별 URL 다운로드 + 간단한 검증.
        - max_retries: 너무 크지 않게 (2~3 정도 추천)
        - 429(Too Many Requests) → Retry-After 헤더 우선, 없으면 지수 백오프 + jitter
        """
        out_path = self._local_path_for(entry)
        if out_path.is_file():
            return out_path

        url = entry["img_url"]

        # 한 세션을 재사용하면 TCP 연결 재사용 + 성능 개선
        session = requests.Session()

        for attempt in range(max_retries):
            try:
                resp = throttled_get(
                    session,
                    url,
                    min_interval=0.8,   # 도메인별 최소 0.8초 간격 등으로 여유 있게
                    timeout=15.0,
                )
                resp.raise_for_status()

                # 파일 저장
                with out_path.open("wb") as f:
                    f.write(resp.content)

                # 간단한 이미지 검증
                try:
                    with Image.open(out_path) as img:
                        img.verify()
                except UnidentifiedImageError:
                    print(f"[download_one] Invalid image file for id={entry['id']} url={url}")
                    out_path.unlink(missing_ok=True)
                    return None

                return out_path

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                print(
                    f"[download_one] HTTPError id={entry['id']} "
                    f"status={status} url={url} attempt={attempt+1}/{max_retries}"
                )

                # --- 429: rate limit 초과 → Retry-After + 백오프 --- #
                if status == 429:
                    retry_after = None
                    if e.response is not None:
                        retry_after = e.response.headers.get("Retry-After")

                    if retry_after is not None:
                        # 서버가 지정해준 대로 기다리기 
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = base_sleep * (2 ** attempt)
                    else:
                        # Retry-After 없으면 지수 백오프 + 약간의 랜덤 지연 
                        jitter = random.uniform(0, 1.0)
                        wait = base_sleep * (2 ** attempt) + jitter

                    wait = min(wait, 60.0)  # 너무 길게는 말고 상한 제한
                    print(f"  -> 429 Too Many Requests, sleep {wait:.1f}s then retry")
                    time.sleep(wait)
                    continue  # 다음 attempt로

                # 다른 HTTP 에러들은 짧게 쉬고 재시도 또는 바로 포기
                jitter = random.uniform(0, 0.5)
                wait = base_sleep + jitter
                print(f"  -> wait {wait:.2f}s then retry")
                time.sleep(wait)

            except Exception as e:
                print(
                    f"[download_one] ERROR id={entry['id']} url={url} "
                    f"attempt={attempt+1}/{max_retries}: {repr(e)}"
                )
                # 네트워크/타임아웃 등 일반 에러도 지수 백오프
                jitter = random.uniform(0, 0.5)
                wait = base_sleep * (2 ** attempt) + jitter
                wait = min(wait, 30.0)
                print(f"  -> sleep {wait:.2f}s then retry")
                time.sleep(wait)

        print(f"[download_one] Failed to download id={entry['id']} url={url}")
        return None
