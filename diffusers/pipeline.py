import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Union

os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, T5EncoderModel
from transformers import logging as transformers_logging

from diffusers import DiffusionPipeline, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from mmdit import DiffusionModel, MMJiTConfig

transformers_logging.set_verbosity_error()


class MiniT2IFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        if train_t_schedule not in {"uniform", "lognorm"}:
            raise ValueError(f"Unsupported train_t_schedule: {train_t_schedule}")

    def sample_train_timesteps(self, batch_size, device, dtype=torch.float32, generator=None):
        if self.config.train_t_schedule == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        normal = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
        normal = normal * self.config.t_lognorm_sigma + self.config.t_lognorm_mu
        return torch.sigmoid(normal).to(dtype=dtype)

    def get_inference_timesteps(self, num_inference_steps=None, device=None, dtype=torch.float32):
        steps = int(num_inference_steps or self.config.num_inference_steps)
        return torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)


class MiniT2IMMJiTModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        txt_input_size: int = 1024,
        hidden_size: int = 768,
        txt_hidden_size: int = 768,
        cond_vec_size: int = 768,
        depth_double: int = 17,
        txt_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6666666666666665,
        pca_channels: int = 128,
        prompt_length: int = 256,
        n_T: int = 100,
        prediction: str = "x",
        sampler: str = "euler",
        cfg_channels: int = 3,
        cfg_interval: tuple = (0.0, 1.0),
        llm: str = "google/flan-t5-large",
    ):
        super().__init__()
        cfg = MMJiTConfig(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            txt_input_size=txt_input_size,
            hidden_size=hidden_size,
            txt_hidden_size=txt_hidden_size,
            cond_vec_size=cond_vec_size,
            depth_double=depth_double,
            txt_preamble_depth=txt_preamble_depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            pca_channels=pca_channels,
            prompt_length=prompt_length,
            n_T=n_T,
            prediction=prediction,
            sampler=sampler,
            cfg_channels=cfg_channels,
            cfg_interval=tuple(cfg_interval),
            llm=llm,
        )
        self.model = DiffusionModel(cfg)

    @property
    def mmjit_config(self) -> MMJiTConfig:
        return self.model.cfg

    def forward(self, img, t, context, attn_mask):
        return self.model.net(img, t, context, attn_mask)

    def pred_velocity(self, x, t, text, mask):
        return self.model.pred_velocity(x, t, text, mask)

    def sample(self, text, mask, cfg_scale=6.0, generator=None, progress=False):
        return self.model.sample(text, mask, cfg_scale=cfg_scale, generator=generator, progress=progress)


class MiniT2ITextToImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer"

    def __init__(
        self,
        transformer: MiniT2IMMJiTModel,
        scheduler: Optional[MiniT2IFlowMatchScheduler] = None,
        tokenizer=None,
        text_encoder=None,
        text_encoder_name: str = "google/flan-t5-large",
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        if not isinstance(scheduler, MiniT2IFlowMatchScheduler):
            scheduler = MiniT2IFlowMatchScheduler(
                train_t_schedule=train_t_schedule,
                t_lognorm_mu=t_lognorm_mu,
                t_lognorm_sigma=t_lognorm_sigma,
                num_inference_steps=num_inference_steps,
            )
        self.register_modules(transformer=transformer, scheduler=scheduler, tokenizer=tokenizer, text_encoder=text_encoder)
        self.register_to_config(
            text_encoder_name=text_encoder_name,
            train_t_schedule=scheduler.config.train_t_schedule,
            t_lognorm_mu=scheduler.config.t_lognorm_mu,
            t_lognorm_sigma=scheduler.config.t_lognorm_sigma,
            num_inference_steps=scheduler.config.num_inference_steps,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        torch_dtype: Optional[torch.dtype] = None,
        text_encoder_dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
        revision: Optional[str] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        root = Path(pretrained_model_name_or_path)
        if not root.exists():
            root = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            )
        transformer = MiniT2IMMJiTModel.from_pretrained(root / "transformer", torch_dtype=torch_dtype, **kwargs)
        scheduler_dir = root / "scheduler"
        if scheduler_dir.exists():
            scheduler = MiniT2IFlowMatchScheduler.from_pretrained(scheduler_dir)
        else:
            scheduler = MiniT2IFlowMatchScheduler()
        text_encoder_name = transformer.mmjit_config.llm
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, local_files_only=local_files_only)
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_name,
            torch_dtype=text_encoder_dtype,
            local_files_only=local_files_only,
        )
        return cls(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_encoder_name=text_encoder_name,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(save_directory / "transformer", **kwargs)
        self.scheduler.save_pretrained(save_directory / "scheduler")
        super().save_pretrained(save_directory, **kwargs)

    def _encode_prompt(self, prompt: Union[str, List[str]], device):
        if isinstance(prompt, str):
            prompt = [prompt]
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_name)
        if self.text_encoder is None:
            self.text_encoder = T5EncoderModel.from_pretrained(self.config.text_encoder_name)
        if next(self.text_encoder.parameters()).device != device:
            self.text_encoder.to(device)
        cfg = self.transformer.mmjit_config
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.prompt_length,
        )
        input_ids = tokens.input_ids.to(device)
        attn = tokens.attention_mask.to(device)
        text = self.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        return text, attn

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        guidance_scale: float = 6.0,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        progress: bool = True,
    ):
        device = next(self.transformer.parameters()).device
        if isinstance(prompt, str):
            prompt_batch = [prompt] * num_images_per_prompt
        else:
            prompt_batch = []
            for p in prompt:
                prompt_batch.extend([p] * num_images_per_prompt)

        old_steps = self.transformer.mmjit_config.n_T
        self.transformer.model.cfg.n_T = int(num_inference_steps or self.scheduler.config.num_inference_steps)
        try:
            text, attn = self._encode_prompt(prompt_batch, device)
            model_dtype = next(self.transformer.parameters()).dtype
            images = self.transformer.sample(
                text.to(dtype=model_dtype),
                attn.to(dtype=model_dtype),
                cfg_scale=guidance_scale,
                generator=generator,
                progress=progress,
            )
        finally:
            self.transformer.model.cfg.n_T = old_steps

        images = (images.clamp(-1, 1) * 127.5 + 128.0).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "pil":
            images = [Image.fromarray(image) for image in images]
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


def build_transformer_from_checkpoint(ckpt_path: Union[str, os.PathLike]) -> MiniT2IMMJiTModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = MMJiTConfig(**payload["config"])
    transformer = MiniT2IMMJiTModel(**asdict(cfg))
    prefixed = payload["state_dict"]
    state_dict = {}
    for key, value in prefixed.items():
        if key.startswith("net."):
            state_dict[f"model.{key}"] = value
        else:
            state_dict[f"model.{key}"] = value
    transformer.load_state_dict(state_dict, strict=True)
    return transformer
