"""Stage auditable F5/SILMA patches for exact update caps and pilot safety."""

from __future__ import annotations

import hashlib
import json
import shutil
from typing import Any

from masriswitch.config import Paths
from masriswitch.data.audit import sha256_file


def _replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"Upstream patch anchor changed: {before[:60]}")
    return text.replace(before, after)


def stage_training_code(paths: Paths) -> dict[str, Any]:
    upstream = paths.artifacts / "upstream" / "f5-tts"
    staged = paths.artifacts / "staged" / "f5-tts"
    if not upstream.is_dir():
        raise FileNotFoundError(upstream)
    if staged.exists():
        raise ValueError("Staged F5 code exists; remove it explicitly before restaging")
    shutil.copytree(upstream, staged, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    trainer = staged / "src" / "f5_tts" / "model" / "trainer.py"
    original_trainer = upstream / "src" / "f5_tts" / "model" / "trainer.py"
    original_silma = paths.artifacts / "upstream" / "silma" / "finetune_cli.py"
    finetune = staged / "src" / "f5_tts" / "train" / "finetune_cli.py"
    dataset = staged / "src" / "f5_tts" / "model" / "dataset.py"
    if not original_silma.is_file():
        raise FileNotFoundError(original_silma)
    upstream_trainer_text = original_trainer.read_text(encoding="utf-8")
    text = upstream_trainer_text
    text = _replace_once(
        text,
        '        if "update" in checkpoint or "step" in checkpoint:\n',
        '        if not latest_checkpoint.startswith("pretrained_") and '
        '("update" in checkpoint or "step" in checkpoint):\n',
    )
    text = _replace_once(
        text,
        "        global_update = start_update\n",
        "        global_update = start_update\n"
        "        max_updates = int(os.environ.get('MASRISWITCH_MAX_UPDATES', '0'))\n"
        "        if max_updates and global_update >= max_updates:\n"
        "            return\n",
    )
    text = _replace_once(
        text,
        "                    self.accelerator.backward(loss)\n",
        "                    if not torch.isfinite(loss):\n"
        "                        raise FloatingPointError('NaN or Inf training loss')\n"
        "                    self.accelerator.backward(loss)\n",
    )
    text = _replace_once(
        text,
        "        self.save_checkpoint(global_update, last=True)\n\n"
        "        self.accelerator.end_training()",
        "                probe_log = os.environ.get('MASRISWITCH_PROBE_LOG')\n"
        "                if probe_log and torch.cuda.is_available():\n"
        "                    free_bytes, _ = torch.cuda.mem_get_info()\n"
        "                    with open("
        "f'{probe_log}_{self.accelerator.process_index}.csv', 'a') as log:\n"
        "                        log.write("
        "f'{global_update},{free_bytes},{float(loss.item())}\\n')\n"
        "                if max_updates and global_update >= max_updates:\n"
        "                    break\n"
        "            if max_updates and global_update >= max_updates:\n"
        "                break\n"
        "        if os.environ.get('MASRISWITCH_NO_FINAL_SAVE') != '1':\n"
        "            self.save_checkpoint(global_update, last=True)\n\n"
        "        self.accelerator.end_training()",
    )
    compile(text, str(trainer), "exec")
    trainer.write_text(text, encoding="utf-8", newline="\n")
    silma_text = original_silma.read_text(encoding="utf-8")
    silma_text = _replace_once(
        silma_text,
        "resumable_with_seed=666,  # seed for shuffling dataset",
        "resumable_with_seed=int(os.environ.get('MASRISWITCH_SEED', '42')),",
    )
    silma_text = _replace_once(
        silma_text,
        "    args = parse_args()\n",
        "    args = parse_args()\n"
        "    from accelerate.utils import set_seed\n"
        "    set_seed(int(os.environ.get('MASRISWITCH_SEED', '42')))\n",
    )
    silma_text = _replace_once(
        silma_text,
        '    checkpoint_path = str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))',
        '    checkpoint_path = os.environ.get("MASRISWITCH_CHECKPOINT_DIR") or '
        'str(files("f5_tts").joinpath(f"../../ckpts/{args.dataset_name}"))',
    )
    silma_text = _replace_once(
        silma_text,
        "            shutil.copy2(ckpt_path, file_checkpoint)",
        "            try:\n"
        "                os.link(os.path.realpath(ckpt_path), file_checkpoint)\n"
        "            except FileExistsError:\n"
        "                pass  # another distributed rank linked the same immutable checkpoint\n"
        "            except OSError:\n"
        "                shutil.copy2(ckpt_path, file_checkpoint)",
    )
    silma_text = _replace_once(
        silma_text,
        "    model = CFM(\n",
        "    model_cfg['checkpoint_activations'] = "
        "os.environ.get('MASRISWITCH_CHECKPOINT_ACTIVATIONS') == '1'\n"
        "    model = CFM(\n",
    )
    silma_text = _replace_once(
        silma_text,
        "    train_dataset = load_dataset(args.dataset_name, tokenizer, "
        "mel_spec_kwargs=mel_spec_kwargs)",
        "    data_dir = os.environ.get('MASRISWITCH_DATA_DIR')\n"
        "    train_dataset = load_dataset(\n"
        "        data_dir or args.dataset_name, tokenizer,\n"
        "        dataset_type='CustomDatasetPath' if data_dir else 'CustomDataset',\n"
        "        mel_spec_kwargs=mel_spec_kwargs,\n"
        "    )",
    )
    silma_text = _replace_once(
        silma_text,
        "    trainer.train(\n        train_dataset,\n",
        "    trainer.train(\n        train_dataset,\n"
        "        num_workers=int(os.environ.get('MASRISWITCH_NUM_WORKERS', '2')),\n",
    )
    if "import os" not in silma_text:
        raise ValueError("SILMA patch no longer imports os")
    compile(silma_text, str(finetune), "exec")
    finetune.write_text(silma_text, encoding="utf-8", newline="\n")
    dataset_text = (upstream / "src" / "f5_tts" / "model" / "dataset.py").read_text(
        encoding="utf-8"
    )
    dataset_text = _replace_once(
        dataset_text,
        '            audio_path = row["audio_path"]\n',
        '            audio_path = row["audio_path"]\n'
        "            audio_root = os.environ.get('MASRISWITCH_AUDIO_ROOT')\n"
        "            if audio_root:\n"
        "                audio_path = os.path.join(audio_root, os.path.basename(audio_path))\n",
    )
    dataset_text = _replace_once(dataset_text, "import json\n", "import json\nimport os\n")
    dataset_text = _replace_once(
        dataset_text,
        '    elif dataset_type == "CustomDatasetPath":\n',
        '    elif dataset_type == "CustomDatasetPath":\n        preprocessed_mel = False\n',
    )
    compile(dataset_text, str(dataset), "exec")
    dataset.write_text(dataset_text, encoding="utf-8", newline="\n")
    result = {
        "upstream_f5_trainer_sha256": hashlib.sha256(
            upstream_trainer_text.encode("utf-8")
        ).hexdigest(),
        "upstream_silma_patch_sha256": sha256_file(original_silma),
        "staged_trainer_sha256": sha256_file(trainer),
        "staged_finetune_sha256": sha256_file(finetune),
        "staged_dataset_sha256": sha256_file(dataset),
    }
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    (paths.artifacts / "train_patch.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
