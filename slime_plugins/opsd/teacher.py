"""Teacher model helpers for OPSD."""

from __future__ import annotations

import torch


class OPSDTeacherManager:
    """Lazily manage the training teacher for OPSD.

    The training teacher is used for mixture-teacher distillation and may be EMA-updated.
    """

    def __init__(self) -> None:
        self._training_teacher = None
        self._bridge = None

    @staticmethod
    def _resolve_teacher_device_and_dtype(reference_model=None) -> tuple[torch.device, torch.dtype]:
        if reference_model is not None:
            ref_param = next(reference_model.parameters())
            return ref_param.device, ref_param.dtype

        if torch.cuda.is_available():
            return torch.device("cuda"), torch.bfloat16
        return torch.device("cpu"), torch.float32

    @staticmethod
    def _load_hf_teacher(args, reference_model=None):
        from transformers import AutoModelForCausalLM

        device, dtype = OPSDTeacherManager._resolve_teacher_device_and_dtype(reference_model)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.hf_checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        teacher = teacher.to(device)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        return teacher

    @property
    def training_teacher(self):
        return self._training_teacher

    def ensure_training_teacher(self, args, reference_model=None):
        if self._training_teacher is None:
            self._training_teacher = self._load_hf_teacher(args, reference_model=reference_model)
        return self._training_teacher

    def update_ema(self, args, model_chunks) -> None:
        if self._training_teacher is None:
            return

        from megatron.bridge import AutoBridge
        from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config, patch_megatron_model

        if self._bridge is None:
            self._bridge = patch_auto_bridge_hf_config(
                AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
            )

        decay = args.opsd_ema_decay
        teacher_params = dict(self._training_teacher.named_parameters())
        with patch_megatron_model(model_chunks):
            for hf_tuple in self._bridge.export_hf_weights(model_chunks, cpu=False, show_progress=False):
                t_param = teacher_params.get(hf_tuple.param_name)
                if t_param is None:
                    continue
                with torch.no_grad():
                    current = hf_tuple.weight.to(t_param.device, dtype=t_param.dtype)
                    t_param.mul_(decay).add_(current, alpha=1.0 - decay)
