"""Aquila-VL model implementation."""

import copy
import json
import re
from pathlib import Path
from typing import Optional

import torch

from levante_bench.models.base import SYSTEM_PROMPT, VLMModel
from levante_bench.models.registry import register
from levante_bench.models._common import (
    DTYPE_MAP,
    load_pil_images,
)


@register("aquila_vl_checkpoint")
@register("aquila_vl")
class AquilaVLModel(VLMModel):
    """Aquila-VL via LLaVA-NeXT loader (official usage path)."""

    def __init__(
        self,
        model_name: str = "BAAI/Aquila-VL-2B-Intermediate",
        device: str = "cpu",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        conv_template: str = "qwen_1_5",
        checkpoint_subdir: str | None = None,
        revision: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)
        self.dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.attn_implementation = attn_implementation
        self.conv_template = conv_template
        self.checkpoint_subdir = checkpoint_subdir
        self.revision = revision
        self.tokenizer = None
        self.image_processor = None

    def load(self) -> None:
        """Load Aquila-VL using llava.model.builder.load_pretrained_model."""
        try:
            from llava.model.builder import load_pretrained_model
        except Exception as exc:
            raise RuntimeError(
                "Aquila-VL requires LLaVA-NeXT. Install with: "
                "pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
            ) from exc

        model_path = self.model_name
        if self.checkpoint_subdir:
            from huggingface_hub import snapshot_download

            local_repo = snapshot_download(
                repo_id=self.model_name,
                revision=self.revision,
                allow_patterns=[f"{self.checkpoint_subdir}/*"],
            )
            model_path = str(Path(local_repo) / self.checkpoint_subdir)
            self._normalize_local_config(Path(model_path))

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            "llava_qwen",
            device_map=self.device,
            attn_implementation=self.attn_implementation,
            revision=self.revision,
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.model.eval()

    def generate(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Generate text using Aquila-VL."""
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        pil_images = load_pil_images(image_paths)
        prompt_with_tokens = prompt_text
        if pil_images:
            prompt_with_tokens = re.sub(r"<image\d+>", DEFAULT_IMAGE_TOKEN, prompt_text)
            if DEFAULT_IMAGE_TOKEN not in prompt_with_tokens:
                prompt_with_tokens = f"{DEFAULT_IMAGE_TOKEN}\n{prompt_text}"

        conv = copy.deepcopy(conv_templates[self.conv_template])
        # LLaVA-NeXT conversation templates expose a mutable `system` field.
        conv.system = SYSTEM_PROMPT
        conv.append_message(conv.roles[0], prompt_with_tokens)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt_question,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)

        generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
        }
        if pil_images:
            image_tensor = process_images(pil_images, self.image_processor, self.model.config)
            image_tensor = [
                img.to(dtype=self.dtype, device=self.device) for img in image_tensor
            ]
            generate_kwargs["images"] = image_tensor
            generate_kwargs["image_sizes"] = [img.size for img in pil_images]

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generate_kwargs)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    def parse_response(self, raw_output: str) -> str:
        """Return generated text as-is (already decoded from generated tokens only)."""
        return raw_output.strip()

    def _normalize_local_config(self, model_dir: Path) -> None:
        """Patch known non-portable checkpoints fields after download.

        Some Aquila intermediate checkpoints reference a private absolute path
        for `mm_vision_tower`. Replace that with a public HF repo id.
        """
        cfg_path = model_dir / "config.json"
        if not cfg_path.exists():
            return
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return

        mm_vision_tower = str(cfg.get("mm_vision_tower") or "")
        if mm_vision_tower.startswith("/share/project/"):
            cfg["mm_vision_tower"] = "google/siglip-so400m-patch14-384"
            cfg_path.write_text(
                json.dumps(cfg, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
