"""Versioned, deterministic text-only MasriSwitch benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from masriswitch.text.entities import parse_entities
from masriswitch.text.spans import analyze_switches

DOMAINS = {
    "telecom": ("الباقة", "Premium Plan", "تجديد الخط", "data bundle", "Vodafone"),
    "banking": ("الحساب", "Online Banking", "تحويل الفلوس", "credit card", "Visa"),
    "ecommerce": ("الطلب", "Express Delivery", "توصيل الأوردر", "tracking link", "Amazon"),
    "hospitality": ("الحجز", "Room Service", "تأكيد الحجز", "check-in", "Hilton"),
    "healthcare": ("المعاد", "Follow-up Visit", "تأكيد الكشف", "lab result", "Vezeeta"),
    "technical": ("الجهاز", "Wi-Fi Router", "حل المشكلة", "support ticket", "Microsoft"),
}
BUCKET_COUNTS = {"simple": 50, "ar_dominant": 100, "balanced": 30, "entity_heavy": 20}
CRITICAL = {"money", "date", "time", "phone", "id", "brand", "acronym"}


def _prompt(domain: str, bucket: str, index: int) -> str:
    arabic, product, action, english, brand = DOMAINS[domain]
    number = 1000 + index
    if bucket == "simple":
        return (
            f"يا فندم، {action} بخصوص {arabic} رقم {number} اتأكد النهارده.",
            f"حضرتك تقدر تراجع {arabic} رقم {number} من فضلك.",
            f"تمام، هنتابع {arabic} رقم {number} وهنرد عليك حالا.",
            f"لو سمحت اتأكد من بيانات {arabic} رقم {number} قبل المعاد.",
        )[index % 4]
    if bucket == "ar_dominant":
        return (
            f"يا فندم، {action} لمنتج {product} رقم {number} اتأكد النهارده.",
            f"ممكن تراجع {product} بتاع {arabic} رقم {number}؟",
            f"حضرتك هتلاقي تفاصيل {product} في {arabic} رقم {number}.",
            f"اتأكدنا إن {product} مربوط بـ {arabic} رقم {number} دلوقتي.",
        )[index % 4]
    if bucket == "balanced":
        return (
            f"يا فندم، {product} و {english} للطلب {number} جاهزين دلوقتي.",
            f"حضرتك، {english} مرتبط بـ {product} رقم {number} من النهارده.",
            f"راجع {product} مع {english} للملف {number} قبل ما نكمل.",
        )[index % 3]
    stress = (
        f"يا فندم، {brand} {product} رقم ORD{number} بقيمة {index + 25} EGP "
        "يوم 25/09/2026، ابعت OTP.",
        f"حضرتك، {brand} هيتصل على 0101234{index:04d} الساعة 4:30 PM بخصوص {product}.",
        f"راجع {brand} {product} رقم ACC{number} وحجم 12.5 GB على https://example.org/{number}.",
        f"طلب {brand} {product} رقم ORD{number} بقيمة {index + 25} USD وخصم 12.5%.",
        f"تأكيد {brand} {product} يوم 25/09/2026 الساعة 9:15 AM، ابعت SMS رقم {number}.",
    )
    return stress[index % len(stress)]


def generate_benchmark(seed: int = 42) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    locked: list[str] = []
    for domain_index, domain in enumerate(DOMAINS):
        for bucket, count in BUCKET_COUNTS.items():
            group: list[dict[str, Any]] = []
            for index in range(count):
                text = _prompt(domain, bucket, index)
                sample_id = f"msb1_{domain}_{bucket}_{index:03d}"
                entities = [entity.__dict__ for entity in parse_entities(text)]
                group.append(
                    {
                        "id": sample_id,
                        "version": "1",
                        "domain": domain,
                        "bucket": bucket,
                        "text": text,
                        "entities": entities,
                        "critical_entity_count": sum(item["kind"] in CRITICAL for item in entities),
                        "detected_switch_bucket": analyze_switches(text).bucket,
                    }
                )
            ranked = sorted(
                group,
                key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
            )
            target = count // 4 + (1 if count % 4 and domain_index % 2 == 0 else 0)
            locked.extend(row["id"] for row in ranked[:target])
            rows.extend(group)
    if len(rows) != 1200 or len(locked) != 300 or len({row["text"] for row in rows}) != 1200:
        raise AssertionError("Benchmark size or uniqueness changed")
    return rows, sorted(locked)


def write_benchmark(artifact_dir: Path, seed: int = 42) -> dict[str, Any]:
    rows, locked = generate_benchmark(seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompts = artifact_dir / "masriswitch_bench_v1.jsonl"
    prompts.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (artifact_dir / "benchmark_locked_ids.json").write_text(
        json.dumps({"version": "1", "seed": seed, "ids": locked}, indent=2), encoding="utf-8"
    )
    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    manifest = {
        "version": "1",
        "seed": seed,
        "prompts": len(rows),
        "locked_test_ids": len(locked),
        "sha256": digest,
        "domains": dict(Counter(row["domain"] for row in rows)),
        "buckets": dict(Counter(row["bucket"] for row in rows)),
    }
    (artifact_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
