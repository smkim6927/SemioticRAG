# eval.py
import os
import random
from typing import List, Tuple, Dict, Any
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm.auto import tqdm


# 공용 유틸: 로컬 이미지 로딩
def load_local_image_safe(path: str):
    """
    로컬 파일에서 이미지를 안전하게 로드.
    - 성공하면 PIL.Image 반환
    - 실패하면 None 반환
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[load_local_image_safe] Failed to open {path}: {repr(e)}", flush=True)
        return None


# Artpedia용 로컬 경로 유틸
def local_path_for_entry(entry: Dict[str, Any], cache_dir: str) -> str:
    """
    dataset.py의 _local_path_for(entry)와 동일한 규칙으로
    로컬 이미지 파일 경로를 생성해야 함.
    (dataset.py에서 사용한 규칙과 다르면 여기서 파일을 못 찾습니다.)
    """
    url = entry["img_url"]
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1]
    if not ext:
        ext = ".jpg"
    fname = f"{entry['id']}{ext}"
    return str(Path(cache_dir) / fname)


# AP 계산 유틸
def average_precision_from_scores(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    scores: (M,)  - similarity 점수 (클수록 visual일 확률이 높다고 가정)
    labels: (M,)  - 1: visual (positive), 0: contextual (negative)

    표준 AP 정의:
      - score 내림차순 정렬 후,
      - positive인 위치들에서의 precision 평균.
    """
    assert scores.ndim == 1 and labels.ndim == 1
    assert scores.shape[0] == labels.shape[0]

    pos_count = labels.sum().item()
    if pos_count == 0:
        return 0.0

    sorted_idx = torch.argsort(scores, descending=True)
    sorted_labels = labels[sorted_idx]

    cum_pos = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, scores.shape[0] + 1, device=scores.device, dtype=torch.float32)
    precision_at_k = cum_pos.to(torch.float32) / ranks

    ap = (precision_at_k * sorted_labels.to(torch.float32)).sum() / pos_count
    return ap.item()


# 평가용 페이지 빌더
def build_eval_pages(
    raw_data: Dict[str, Any],
    cache_dir: str,
    splits: Tuple[str, ...] = ("val", "valid", "test"),
) -> List[dict]:
    """
    Artpedia raw_data에서 평가용 'page' 리스트를 구성.
    raw_data 는 Dict[id, entry] 형태라고 가정 (dataset.py와 동일).
    각 page = {
        "id": 엔트리 id (str),
        "image_path": 로컬 이미지 경로,
        "visual_sentences": List[str],
        "contextual_sentences": List[str],
    }
    """
    pages: List[dict] = []

    for _id, entry in raw_data.items():
        split = entry.get("split")
        if split not in splits:
            continue

        vis = entry.get("visual_sentences") or []
        ctx = entry.get("contextual_sentences") or []
        if not vis or not ctx:
            continue

        img_url = entry.get("img_url")
        if not img_url:
            continue

        ex_for_path = {
            "id": _id,
            "img_url": img_url,
        }
        img_path = local_path_for_entry(ex_for_path, cache_dir)

        if not os.path.isfile(img_path):
            continue

        pages.append(
            {
                "id": _id,
                "image_path": img_path,
                "visual_sentences": vis,
                "contextual_sentences": ctx,
            }
        )

    return pages


# Intra-page AP (CLIP)
def evaluate_intra_page_AP_clip(
    fsdp_model: FSDP,
    raw_data: Dict[str, Any],
    cache_dir: str,
    splits: Tuple[str, ...],
    rank: int,
):
    if rank != 0:
        return None

    base_model = fsdp_model.module  # CLIPModule
    clip_model = base_model.clip
    processor = base_model.processor

    device = next(clip_model.parameters()).device
    clip_model.eval()

    pages = build_eval_pages(raw_data, cache_dir, splits=splits)
    if len(pages) == 0:
        print("[Intra-AP] No eval pages (check cache or splits).")
        return None

    ap_list = []

    with torch.no_grad():
        for page in tqdm(pages, desc="[Intra-AP] pages", disable=False):
            img = load_local_image_safe(page["image_path"])
            if img is None:
                continue

            visual = page["visual_sentences"]
            contextual = page["contextual_sentences"]
            all_texts = visual + contextual

            labels = torch.tensor(
                [1] * len(visual) + [0] * len(contextual),
                dtype=torch.float32,
                device=device,
            )

            img_inputs = processor(images=[img], return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            img_emb = clip_model.get_image_features(**img_inputs)  # (1, D)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            text_inputs = processor(
                text=all_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            txt_emb = clip_model.get_text_features(**text_inputs)  # (M, D)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            scores = (img_emb @ txt_emb.t()).squeeze(0)  # (M,)

            ap = average_precision_from_scores(scores, labels)
            ap_list.append(ap)

    if not ap_list:
        print("[Intra-AP] No valid pages after filtering.")
        return None

    mean_ap = sum(ap_list) / len(ap_list)
    print(f"[Intra-AP] mean AP over {len(ap_list)} pages = {mean_ap:.4f}")
    return mean_ap


# Retrieval을 위한 임베딩 계산
def compute_clip_embeddings_for_pages(
    fsdp_model: FSDP,
    pages: List[dict],
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    pages: build_eval_pages로 만든 리스트
    Returns:
      - img_embs:  (Ni, D)
      - txt_embs:  (Nt, D)  (visual 문장만)
      - image_ids: 길이 Ni, 각 인덱스의 이미지 id (str)
      - text_image_ids: 길이 Nt, 각 문장이 속한 이미지 id (str)
    """
    base_model = fsdp_model.module
    clip_model = base_model.clip
    processor = base_model.processor
    device = next(clip_model.parameters()).device

    clip_model.eval()

    # 이미지 쪽
    image_paths: List[str] = []
    image_ids: List[str] = []
    for page in pages:
        image_paths.append(page["image_path"])
        image_ids.append(page["id"])

    all_img_embs = []
    batch_size = 16

    with torch.no_grad():
        for i in tqdm(
            range(0, len(image_paths), batch_size),
            desc="[Retrieval] encode images",
        ):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                img = load_local_image_safe(p)
                if img is not None:
                    imgs.append(img)
                else:
                    imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            img_inputs = processor(images=imgs, return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
            img_emb = clip_model.get_image_features(**img_inputs)  # (B, D)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            all_img_embs.append(img_emb)

    img_embs = torch.cat(all_img_embs, dim=0)  # (Ni, D)

    # 텍스트(visual 문장) 쪽
    texts: List[str] = []
    text_image_ids: List[str] = []
    for page in pages:
        for s in page["visual_sentences"]:
            texts.append(s)
            text_image_ids.append(page["id"])

    all_txt_embs = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="[Retrieval] encode visual texts",
        ):
            batch_texts = texts[i : i + batch_size]
            text_inputs = processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            txt_emb = clip_model.get_text_features(**text_inputs)  # (B, D)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            all_txt_embs.append(txt_emb)

    txt_embs = torch.cat(all_txt_embs, dim=0)  # (Nt, D)

    return img_embs, txt_embs, image_ids, text_image_ids


# Retrieval Metrics (R@K)
def evaluate_retrieval_clip(
    img_embs: torch.Tensor,
    txt_embs: torch.Tensor,
    image_ids: List[str],
    text_image_ids: List[str],
    N_list=(10, 50, 100),
    K_list=(1, 5),
    seed: int = 42,
):
    """
    논문 Artpedia와 유사하게 N개 이미지 subset에서 R@K를 계산.
    img_embs: (Ni, D)
    txt_embs: (Nt, D)
    image_ids: len Ni
    text_image_ids: len Nt
    """
    Ni = img_embs.shape[0]
    Nt = txt_embs.shape[0]

    sim_i2t = img_embs @ txt_embs.t()  # (Ni, Nt)
    sim_t2i = sim_i2t.t()              # (Nt, Ni)

    rng = random.Random(seed)

    img_to_text_idx: Dict[str, List[int]] = {img_id: [] for img_id in image_ids}
    for t_idx, img_id in enumerate(text_image_ids):
        if img_id in img_to_text_idx:
            img_to_text_idx[img_id].append(t_idx)

    print("=== Cross-modal retrieval (Img↔Text) ===")
    for N in N_list:
        print(f"\n-- N = {N} --")

        # Img -> Text
        results_i2t = {k: 0 for k in K_list}
        valid_queries = 0

        for i in range(Ni):
            img_id = image_ids[i]
            pos_texts = img_to_text_idx.get(img_id, [])
            if len(pos_texts) == 0:
                continue
            valid_queries += 1

            other_img_indices = [j for j in range(Ni) if j != i]
            if len(other_img_indices) < N - 1:
                subset_other = other_img_indices
            else:
                subset_other = rng.sample(other_img_indices, N - 1)

            candidate_img_indices = [i] + subset_other
            candidate_img_ids = [image_ids[j] for j in candidate_img_indices]

            candidate_text_indices = set()
            for cid in candidate_img_ids:
                for t_idx in img_to_text_idx.get(cid, []):
                    candidate_text_indices.add(t_idx)
            candidate_text_indices = list(candidate_text_indices)

            sims = sim_i2t[i, candidate_text_indices]
            pos_in_candidate = [
                candidate_text_indices.index(t_idx)
                for t_idx in pos_texts
                if t_idx in candidate_text_indices
            ]
            if not pos_in_candidate:
                continue

            sorted_idx = torch.argsort(sims, descending=True)
            for K in K_list:
                topk = sorted_idx[:K].tolist()
                if any(pi in topk for pi in pos_in_candidate):
                    results_i2t[K] += 1

        if valid_queries > 0:
            for K in K_list:
                r_at_k = 100.0 * results_i2t[K] / valid_queries
                print(f"Img->Text R@{K} = {r_at_k:.2f}% (over {valid_queries} queries)")
        else:
            print("Img->Text: no valid queries")

        # Text -> Img
        results_t2i = {k: 0 for k in K_list}
        valid_queries_t = 0

        for t in range(Nt):
            gt_img_id = text_image_ids[t]
            if gt_img_id not in image_ids:
                continue
            valid_queries_t += 1

            gt_img_idx = image_ids.index(gt_img_id)

            other_img_indices = [j for j in range(Ni) if j != gt_img_idx]
            if len(other_img_indices) < N - 1:
                subset_other = other_img_indices
            else:
                subset_other = rng.sample(other_img_indices, N - 1)

            candidate_img_indices = [gt_img_idx] + subset_other

            sims = sim_t2i[t, candidate_img_indices]
            sorted_idx = torch.argsort(sims, descending=True)

            for K in K_list:
                topk = sorted_idx[:K].tolist()
                if 0 in topk:  # 정답 이미지는 candidate_img_indices에서 index 0
                    results_t2i[K] += 1

        if valid_queries_t > 0:
            for K in K_list:
                r_at_k = 100.0 * results_t2i[K] / valid_queries_t
                print(f"Text->Img R@{K} = {r_at_k:.2f}% (over {valid_queries_t} queries)")
        else:
            print("Text->Img: no valid queries")



# 최종 Artpedia-style 평가 엔트리
def evaluate_artpedia_metrics_clip(
    fsdp_model: FSDP,
    raw_data: Dict[str, Any],
    cache_dir: str,
    splits: Tuple[str, ...],
    rank: int,
):
    """
    CLIP 버전 Artpedia 평가:
      - intra-page mean AP
      - cross-modal Img↔Text R@K with N in {10,50,100}
    """
    if rank != 0:
        return

    print("========== Artpedia-style evaluation (CLIP) ==========")
    pages = build_eval_pages(raw_data, cache_dir, splits=splits)
    if len(pages) == 0:
        print("No eval pages found. Check cache_dir and splits.")
        return

    mean_ap = evaluate_intra_page_AP_clip(
        fsdp_model=fsdp_model,
        raw_data=raw_data,
        cache_dir=cache_dir,
        splits=splits,
        rank=rank,
    )

    img_embs, txt_embs, image_ids, text_image_ids = compute_clip_embeddings_for_pages(
        fsdp_model,
        pages,
    )

    evaluate_retrieval_clip(
        img_embs=img_embs,
        txt_embs=txt_embs,
        image_ids=image_ids,
        text_image_ids=text_image_ids,
        N_list=(10, 50, 100),
        K_list=(1, 5),
    )

    print("======================================================")
