"""Teacher model helpers for OPSD."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _get_tp_info() -> tuple[int, int, object | None]:
    """Return (tp_rank, tp_world_size, tp_group). Falls back to (0, 1, None) outside Megatron."""
    try:
        from megatron.core import mpu

        return (
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_tensor_model_parallel_world_size(),
            mpu.get_tensor_model_parallel_group(),
        )
    except Exception:
        return 0, 1, None


def _get_tp_src_rank() -> int:
    """Return the global rank of TP local rank 0 (the broadcast source)."""
    try:
        import torch.distributed as dist
        from megatron.core import mpu

        tp_group = mpu.get_tensor_model_parallel_group()
        ranks = dist.get_process_group_ranks(tp_group)
        return ranks[0]
    except Exception:
        return 0


class OPSDTeacherManager:
    """Lazily manage the training teacher for OPSD.

    The training teacher is used for mixture-teacher distillation and may be EMA-updated.

    Within a tensor-parallel (TP) group all ranks process the same batch, so
    we only load and run the HuggingFace teacher on **TP rank 0** and broadcast
    the resulting logits to the other ranks in the group.  Each data-parallel
    (DP) group does its own independent broadcast, so DP parallelism is fully
    preserved.
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
        """Load teacher on TP rank 0 only; other TP ranks skip the load."""
        tp_rank, tp_world_size, _tp_group = _get_tp_info()
        if self._training_teacher is None and (tp_rank == 0 or tp_world_size == 1):
            logger.info("OPSD: loading teacher model on TP rank %d", tp_rank)
            self._training_teacher = self._load_hf_teacher(args, reference_model=reference_model)
        return self._training_teacher

    def forward_and_broadcast(
        self,
        teacher_inputs: list[torch.Tensor],
        num_logits_to_keep: int = 0,
        device: torch.device | None = None,
        pad_id: int = 0,
    ) -> torch.Tensor | None:
        """Run teacher forward on TP rank 0 and broadcast result to the TP group.

        Returns the logit tensor on every rank (shape: [B, seq, vocab] or
        [B, num_logits_to_keep, vocab] when num_logits_to_keep > 0).
        Returns None when there are no inputs.
        """
        if not teacher_inputs:
            return None

        tp_rank, tp_world_size, tp_group = _get_tp_info()
        src_rank = _get_tp_src_rank()

        # --- TP rank 0 computes the logits ---
        if tp_rank == 0:
            assert self._training_teacher is not None, "Teacher not loaded on TP rank 0"
            max_len = max(t.size(0) for t in teacher_inputs)
            import torch.nn.functional as F

            padded = torch.stack([F.pad(t, (0, max_len - t.size(0)), value=pad_id) for t in teacher_inputs])
            kwargs = {"logits_to_keep": num_logits_to_keep} if num_logits_to_keep > 0 else {}
            torch.cuda.empty_cache()
            with torch.no_grad():
                result = self._training_teacher(padded, **kwargs).logits
            result = result.contiguous()
            # Encode shape + dtype index in a single int64 vector: [B, seq, vocab, dtype_id]
            # dtype_id: 0=float32, 1=bfloat16, 2=float16
            _DTYPE_TO_ID = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
            dtype_id = _DTYPE_TO_ID.get(result.dtype, 0)
            shape = torch.tensor([*result.shape, dtype_id], dtype=torch.long, device=result.device)
        else:
            result = None
            shape = torch.zeros(4, dtype=torch.long, device=device)

        # --- broadcast shape+dtype, then data ---
        if tp_world_size > 1:
            torch.distributed.broadcast(shape, src=src_rank, group=tp_group)
            if tp_rank != 0:
                _ID_TO_DTYPE = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16}
                recv_dtype = _ID_TO_DTYPE.get(int(shape[3].item()), torch.bfloat16)
                result = torch.empty(shape[:3].tolist(), dtype=recv_dtype, device=device)
            torch.distributed.broadcast(result, src=src_rank, group=tp_group)

        return result

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
