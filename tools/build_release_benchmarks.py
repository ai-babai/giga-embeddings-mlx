from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

V1 = Path("runs/2026-08-25-giga-embeddings-0826-v1")
V3 = Path("runs/2026-08-26-giga-embeddings-0826-v3")

VARIANTS = {
    "480m-bf16": {
        "profile": "480M",
        "role": "upstream baseline",
        "speed": "speed-480m-bf16.json",
        "load": "load-480m-bf16.json",
        "manifest": "480m-bf16.json",
    },
    "480m-q8": {
        "profile": "480M",
        "role": "compact",
        "speed": "speed-480m-q8-g64.json",
        "load": "load-480m-q8-g64.json",
        "manifest": "480m-q8-g64.json",
    },
    "3b-bf16": {
        "profile": "3B",
        "role": "upstream baseline",
        "speed": "speed-3b-bf16.json",
        "load": "load-3b-bf16.json",
        "manifest": "3b-bf16.json",
    },
    "3b-q8": {
        "profile": "3B",
        "role": "balanced default",
        "speed": "speed-3b-q8-edges-bf16-g64.json",
        "load": "load-3b-q8-edges-bf16-g64.json",
        "manifest": "3b-q8-edges-bf16-g64.json",
    },
    "10b-a1.8b-bf16": {
        "profile": "10B-A1.8B",
        "role": "upstream baseline",
        "speed": "speed-10b-a1.8b-bf16.json",
        "load": "load-10b-a1.8b-bf16.json",
        "manifest": "10b-a1.8b-bf16.json",
    },
    "10b-a1.8b-q8": {
        "profile": "10B-A1.8B",
        "role": "compact / research",
        "speed": "speed-10b-a1.8b-q8-g64.json",
        "load": "load-10b-a1.8b-q8-g64.json",
        "manifest": "10b-a1.8b-q8-g64.json",
    },
}

QUANT_REPORTS = {
    "480m-q8": ("holdout-480m-final.json", "downstream-480m.json", "480m-q8-g64"),
    "3b-q8": (
        "holdout-3b-q8-edges.json",
        "downstream-3b.json",
        "3b-q8-edges-bf16-g64",
    ),
    "10b-a1.8b-q8": (
        "holdout-10b-a1.8b.json",
        "downstream-10b-a1.8b.json",
        "10b-a1.8b-q8-g64",
    ),
}


def read_json(path: Path, evidence: dict[str, str], root: Path) -> Any:
    raw = path.read_bytes()
    evidence[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def select_case(report: dict[str, Any], batch: int, length: int) -> dict[str, Any]:
    return next(
        case
        for case in report["cases"]
        if case["batch"] == batch and case["sequence_length"] == length
    )


def build_payload(data_root: Path, machine: str) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    scratch = data_root / V1 / "scratch"
    manifests = data_root / V3 / "manifests"
    resources = []

    for alias, spec in VARIANTS.items():
        speed = read_json(scratch / spec["speed"], evidence, data_root)
        load = read_json(scratch / spec["load"], evidence, data_root)
        manifest = read_json(manifests / spec["manifest"], evidence, data_root)
        b1_512 = select_case(speed, 1, 512)
        b16_1024 = select_case(speed, 16, 1024)
        resources.append(
            {
                "alias": alias,
                "profile": spec["profile"],
                "role": spec["role"],
                "artifact_bytes": manifest["artifact_bytes"],
                "process_max_rss_bytes": speed["process_max_rss_bytes"],
                "metal_peak_b16_1024_bytes": b16_1024["metal_peak_bytes"],
                "first_materialized_load_seconds": load["first_process_materialized_load_seconds"],
                "warm_reload_seconds": load["warm_reload_seconds"],
                "b1_512": {
                    "median_seconds": b1_512["median_seconds"],
                    "p95_seconds": b1_512["p95_seconds"],
                    "documents_per_second": b1_512["documents_per_second"],
                    "tokens_per_second": b1_512["tokens_per_second"],
                },
                "b16_1024": {
                    "median_seconds": b16_1024["median_seconds"],
                    "p95_seconds": b16_1024["p95_seconds"],
                    "documents_per_second": b16_1024["documents_per_second"],
                    "tokens_per_second": b16_1024["tokens_per_second"],
                },
            }
        )

    acceptance = read_json(
        data_root / V3 / "results/revision3-acceptance.json", evidence, data_root
    )
    bf16_quality = []
    for profile in acceptance["profiles"]:
        checks = {check["name"]: check["value"] for check in profile["checks"]}
        families = ("ru ", "en ", "code ", "multilingual ")
        family_mrr = [
            value
            for name, value in checks.items()
            if name.startswith(families) and name.endswith(" mrr_delta")
        ]
        family_ndcg = [
            value
            for name, value in checks.items()
            if name.startswith(families) and name.endswith(" ndcg_at_10_delta")
        ]
        bf16_quality.append(
            {
                "profile": profile["model"],
                "minimum_pooled_vector_cosine": checks["minimum pooled-vector cosine"],
                "mlx_single_padded_cosine": checks["MLX single/padded cosine"],
                "aggregate_mrr_delta": checks["aggregate mrr_delta"],
                "aggregate_ndcg_at_10_delta": checks["aggregate ndcg_at_10_delta"],
                "worst_family_mrr_delta": min(family_mrr),
                "worst_family_ndcg_at_10_delta": min(family_ndcg),
                "pass": profile["pass"],
            }
        )

    quant_quality = []
    for alias, (holdout_name, downstream_name, variant_name) in QUANT_REPORTS.items():
        holdout = read_json(scratch / holdout_name, evidence, data_root)
        downstream = read_json(scratch / downstream_name, evidence, data_root)
        quant = next(item for item in holdout["results"] if item["variant"] == variant_name)
        task = next(item for item in downstream["results"] if item["variant"] == variant_name)
        family_ndcg = [family["ndcg_at_10_delta"] for family in quant["ranking_by_family"].values()]
        quant_quality.append(
            {
                "alias": alias,
                "min_aligned_cosine": quant["min_aligned_cosine"],
                "mean_aligned_cosine": quant["mean_aligned_cosine"],
                "similarity_spearman": quant["similarity_spearman"],
                "top1_agreement": quant["ranking"]["top1_agreement"],
                "mean_top10_overlap": quant["ranking"]["mean_top10_overlap"],
                "ndcg_at_10_delta": quant["ranking"]["ndcg_at_10_delta"],
                "worst_family_ndcg_at_10_delta": min(family_ndcg),
                "rusts_spearman_delta": task["ru_sts"]["spearman_delta"],
                "classification_accuracy_delta": task["ru_classification"]["accuracy_delta"],
                "classification_macro_f1_delta": task["ru_classification"]["macro_f1_delta"],
            }
        )

    runtime = read_json(manifests / "480m-q8-g64.json", evidence, data_root)["runtime"]
    return {
        "schema_version": 1,
        "release_line": "0826",
        "acceptance": {
            "criteria": "effectiveness-based revision 3",
            "overall_pass": acceptance["overall_pass"],
            "report_sha256": evidence[str(V3 / "results/revision3-acceptance.json")],
        },
        "benchmark_environment": {
            "machine": machine,
            "python": runtime["python"],
            "mlx": runtime["mlx"],
            "mlx_lm": runtime["mlx_lm"],
            "warmups": 2,
            "measured_repetitions": 5,
            "os_page_cache_flushed_for_load_test": False,
        },
        "resources_and_speed": resources,
        "bf16_backend_quality": bf16_quality,
        "quant_quality": quant_quality,
        "evidence_sha256": dict(sorted(evidence.items())),
    }


def decimal_gb(value: int) -> str:
    return f"{value / 1_000_000_000:.3f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Giga Embeddings 0826 MLX — measured results",
        "",
        (
            "This file is generated from the Goal 001 JSON evidence by "
            "`tools/build_release_benchmarks.py`. Do not edit measured values by hand."
        ),
        "",
        "## Measurement context",
        "",
        f"- Machine: {payload['benchmark_environment']['machine']}.",
        (
            f"- Runtime: Python {payload['benchmark_environment']['python']}, MLX "
            f"{payload['benchmark_environment']['mlx']}, MLX-LM "
            f"{payload['benchmark_environment']['mlx_lm']}."
        ),
        "- Speed: 2 warmups, 5 measured repetitions; median and p95 are reported.",
        "- Load: first process-materialized and warm reload; OS page cache was not flushed.",
        "- Lower is better for seconds/bytes; higher is better for docs/s, tok/s and quality.",
        "",
        "## Disk, memory, load and speed",
        "",
        "| Alias | Role | Artifact (GB) ↓ | Process max RSS (GB) ↓ | Metal peak B16×1024 (GB) ↓ | Load first / warm (s) ↓ | B1×512 median / p95 (s) ↓ | B16×1024 docs/s ↑ | B16×1024 tok/s ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["resources_and_speed"]:
        lines.append(
            f"| `{row['alias']}` | {row['role']} | {decimal_gb(row['artifact_bytes'])} | "
            f"{decimal_gb(row['process_max_rss_bytes'])} | "
            f"{decimal_gb(row['metal_peak_b16_1024_bytes'])} | "
            f"{row['first_materialized_load_seconds']:.3f} / {row['warm_reload_seconds']:.3f} | "
            f"{row['b1_512']['median_seconds']:.3f} / {row['b1_512']['p95_seconds']:.3f} | "
            f"{row['b16_1024']['documents_per_second']:.2f} | "
            f"{row['b16_1024']['tokens_per_second']:.0f} |"
        )

    lines.extend(
        [
            "",
            "Artifact size, process RSS and Metal peak are distinct measurements and must not be added.",
            "On this M4 Pro, Q8 is a capacity/disk optimization, not a guaranteed speed-up.",
            "",
            "## BF16 backend preservation",
            "",
            "PyTorch BF16 reference versus native MLX BF16. Deltas are MLX minus reference; the acceptance margin for aggregate and each family was −0.005.",
            "",
            "| Profile | Min pooled cosine ↑ | MLX single/padded cosine ↑ | MRR Δ ↑ | NDCG@10 Δ ↑ | Worst family MRR Δ ↑ | Worst family NDCG@10 Δ ↑ | Gate |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["bf16_backend_quality"]:
        lines.append(
            f"| `{row['profile']}` | {row['minimum_pooled_vector_cosine']:.6f} | "
            f"{row['mlx_single_padded_cosine']:.6f} | {row['aggregate_mrr_delta']:+.6f} | "
            f"{row['aggregate_ndcg_at_10_delta']:+.6f} | "
            f"{row['worst_family_mrr_delta']:+.6f} | "
            f"{row['worst_family_ndcg_at_10_delta']:+.6f} | "
            f"{'pass' if row['pass'] else 'fail'} |"
        )

    lines.extend(
        [
            "",
            "Revision 1/2 stricter numerical diagnostics failed and remain part of the historical record. Revision 3 accepts observable retrieval effectiveness while retaining rank and hidden-state drift as diagnostics.",
            "",
            "## Quantized weight preservation",
            "",
            "All deltas are against the corresponding native MLX BF16 artifact.",
            "",
            "| Alias | Min / mean cosine ↑ | Spearman ↑ | Top-1 / top-10 agreement ↑ | NDCG@10 Δ ↑ | Worst family NDCG@10 Δ ↑ | RuSTS Δ ↑ | Classification acc. / macro-F1 Δ ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["quant_quality"]:
        lines.append(
            f"| `{row['alias']}` | {row['min_aligned_cosine']:.6f} / "
            f"{row['mean_aligned_cosine']:.6f} | {row['similarity_spearman']:.6f} | "
            f"{row['top1_agreement'] * 100:.2f}% / {row['mean_top10_overlap'] * 100:.2f}% | "
            f"{row['ndcg_at_10_delta']:+.5f} | "
            f"{row['worst_family_ndcg_at_10_delta']:+.5f} | "
            f"{row['rusts_spearman_delta']:+.6f} | "
            f"{row['classification_accuracy_delta']:+.6f} / "
            f"{row['classification_macro_f1_delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "The 10B Q8 code-family NDCG@10 delta is −0.01297. It is therefore a compact/research artifact, not the default and not a near-lossless claim.",
            "",
            "## Reproducibility",
            "",
            f"Acceptance JSON SHA-256: `{payload['acceptance']['report_sha256']}`.",
            "The companion `0826-results.json` includes SHA-256 for every source evidence file used here.",
            "Raw datasets, model weights and benchmark outputs are intentionally not stored in this Git repository.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    args = parser.parse_args()

    payload = build_payload(args.data_root, args.machine)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
