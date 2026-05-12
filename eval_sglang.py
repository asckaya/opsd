#!/usr/bin/env python3
"""Standalone eval script: starts a sglang server and runs evaluation.

Usage:
    python eval_sglang.py \\
        --model /path/to/model \\
        --tp 4 \\
        [--eval-config eval_only.yaml] \\
        [--n-samples 16] \\
        [--max-len 16384] \\
        [--temperature 0.7] \\
        [--top-p 1.0] \\
        [--port 30000] \\
        [--mem-fraction 0.8] \\
        [--output results.json]
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

import aiohttp
import numpy as np
from omegaconf import OmegaConf
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer

from slime.rollout.rm_hub import async_rm
from slime.utils.eval_config import build_eval_dataset_configs, ensure_dataset_list
from slime.utils.metric_utils import compute_pass_rate, compute_statistics
from slime.utils.types import Sample

# ── Server management ──────────────────────────────────────────────────────────


def start_sglang_server(model: str, tp: int, port: int, mem_fraction: float) -> subprocess.Popen:
    import torch

    num_gpus = torch.cuda.device_count()
    dp = max(1, num_gpus // tp)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--tp",
        str(tp),
        "--dp",
        str(dp),
        "--port",
        str(port),
        "--mem-fraction-static",
        str(mem_fraction),
        "--trust-remote-code",
        "--disable-radix-cache",
    ]
    print(f"[sglang] {num_gpus} GPUs total  →  tp={tp}  dp={dp}")
    print(f"[sglang] Starting server: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


async def wait_for_server(port: int, timeout: int = 300) -> None:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        print(f"[sglang] Server ready on port {port}")
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"sglang server did not become ready within {timeout}s")


# ── Inference ─────────────────────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(
        (
            aiohttp.ServerDisconnectedError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientOSError,
            aiohttp.ServerTimeoutError,
        )
    ),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    port: int,
    prompt: str,
    sampling_params: dict,
    progress: Progress,
    task_id,
) -> dict[str, Any]:
    url = f"http://localhost:{port}/generate"
    payload = {"text": prompt, "sampling_params": sampling_params, "return_logprob": True}
    async with semaphore:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
    progress.advance(task_id)
    return data


def _extract_response_length(response_data: Mapping[str, Any], tokenizer) -> int:
    meta_info = response_data.get("meta_info") or {}
    if output_token_logprobs := meta_info.get("output_token_logprobs"):
        return len(output_token_logprobs)

    response_text = response_data.get("text", "")
    return len(tokenizer.encode(response_text, add_special_tokens=False))


def _extract_sample_status(response_data: Mapping[str, Any]) -> Sample.Status:
    finish_reason = ((response_data.get("meta_info") or {}).get("finish_reason") or {}).get("type")
    if finish_reason == "length":
        return Sample.Status.TRUNCATED
    if finish_reason == "abort":
        return Sample.Status.ABORTED
    return Sample.Status.COMPLETED


def _parse_json_mapping(value: str | None, arg_name: str) -> dict[str, Any] | None:
    if value is None:
        return None

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object.")
    return parsed


def summarize_dataset_results(
    name: str, samples: list[Sample], rewards: list[float], n_samples: int
) -> dict[str, Any]:
    n_prompts = len(rewards) // n_samples if n_samples else 0
    per_prompt = [rewards[i * n_samples : (i + 1) * n_samples] for i in range(n_prompts)]
    avg_per_prompt = [sum(group) / len(group) for group in per_prompt]
    overall = sum(avg_per_prompt) / len(avg_per_prompt) if avg_per_prompt else 0.0

    response_lengths = [sample.response_length for sample in samples]
    pass_at_k = (
        compute_pass_rate(flat_rewards=rewards, group_size=n_samples, num_groups=n_prompts)
        if rewards and n_samples
        else {}
    )
    response_len_stats = (
        compute_statistics(response_lengths)
        if response_lengths
        else {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}
    )
    truncated_ratio = (
        np.mean([sample.status == Sample.Status.TRUNCATED for sample in samples]).item() if samples else 0.0
    )

    return {
        "name": name,
        "n_prompts": n_prompts,
        "n_samples": n_samples,
        "accuracy": overall,
        "pass_at_k": pass_at_k,
        "response_len": response_len_stats,
        "truncated_ratio": truncated_ratio,
        "per_prompt_avg": avg_per_prompt,
    }


# ── Eval loop ─────────────────────────────────────────────────────────────────


async def eval_dataset(
    dataset_cfg,
    tokenizer,
    port: int,
    n_samples: int,
    max_len: int,
    temperature: float,
    top_p: float,
    concurrency: int,
    apply_chat_template_kwargs: dict[str, Any] | None,
) -> dict:
    sampling_params = dict(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_len,
        skip_special_tokens=False,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    # Load dataset and expand to n_samples copies per prompt
    samples: list[Sample] = []
    with open(dataset_cfg.path) as f:
        for line in f:
            row = json.loads(line)
            prompt_msgs = row[dataset_cfg.input_key or "prompt"]
            label = row.get(dataset_cfg.label_key or "label", "")
            metadata = row.get(dataset_cfg.metadata_key or "metadata", {})

            if isinstance(prompt_msgs, list):
                chat_template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                chat_template_kwargs.update(apply_chat_template_kwargs or {})
                text = tokenizer.apply_chat_template(prompt_msgs, **chat_template_kwargs)
            else:
                text = str(prompt_msgs)

            for _ in range(n_samples):
                samples.append(Sample(prompt=text, label=label, metadata=metadata))

    total = len(samples)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)
    # Disable read timeout — long reasoning chains can take several minutes
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with progress:
        task_id = progress.add_task(dataset_cfg.name, total=total)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            responses = await asyncio.gather(
                *[
                    generate_one(session, semaphore, port, s.prompt, sampling_params, progress, task_id)
                    for s in samples
                ]
            )

    # Reward computation
    args_ns = Namespace(custom_rm_path=None, rm_type=dataset_cfg.rm_type or "", rm_url=None)
    rewards = []
    for sample, response_data in zip(samples, responses, strict=True):
        sample.response = response_data["text"]
        sample.response_length = _extract_response_length(response_data, tokenizer)
        sample.status = _extract_sample_status(response_data)
        rewards.append(float(await async_rm(args_ns, sample)))

    return summarize_dataset_results(dataset_cfg.name, samples, rewards, n_samples)


async def run_eval(args: argparse.Namespace) -> None:
    await wait_for_server(args.port)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load eval config — mirrors slime/utils/arguments.py _resolve_eval_datasets()
    cfg_dict = OmegaConf.to_container(OmegaConf.load(args.eval_config), resolve=True)
    eval_cfg = cfg_dict.get("eval", cfg_dict)
    defaults = dict(eval_cfg.get("defaults") or {})
    datasets_config = ensure_dataset_list(eval_cfg.get("datasets"))
    dataset_cfgs = build_eval_dataset_configs(Namespace(), datasets_config, defaults)

    results = []
    for cfg in dataset_cfgs:
        n_samples = cfg.n_samples_per_eval_prompt or args.n_samples
        max_len = cfg.max_response_len or args.max_len
        temp = cfg.temperature if cfg.temperature is not None else args.temperature
        top_p = cfg.top_p if cfg.top_p is not None else args.top_p

        print(f"\n── {cfg.name}  ({cfg.path})")
        result = await eval_dataset(
            dataset_cfg=cfg,
            tokenizer=tokenizer,
            port=args.port,
            n_samples=n_samples,
            max_len=max_len,
            temperature=temp,
            top_p=top_p,
            concurrency=args.concurrency,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
        results.append(result)
        print(f"  accuracy = {result['accuracy']:.4f}  ({result['n_prompts']} prompts × {n_samples} samples)")
        if result["pass_at_k"]:
            pass_at_k_str = "  ".join(f"{metric}={value:.4f}" for metric, value in result["pass_at_k"].items())
            print(f"  pass@k = {pass_at_k_str}")
        response_len = result["response_len"]
        print(
            f"  response_len = mean={response_len['mean']:.2f}  median={response_len['median']:.2f}  max={response_len['max']:.0f}  min={response_len['min']:.0f}"
        )
        print(f"  truncated_ratio = {result['truncated_ratio']:.4f}")

    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    for r in results:
        summary_parts = [f"acc={r['accuracy']:.4f}", f"resp_len_mean={r['response_len']['mean']:.2f}"]
        if r["pass_at_k"]:
            summary_parts.extend(f"{metric}={value:.4f}" for metric, value in r["pass_at_k"].items())
        print(f"  {r['name']:20s}  {'  '.join(summary_parts)}")

    if args.output:
        output_data = {
            "summary": {
                r["name"]: {
                    "accuracy": r["accuracy"],
                    "pass_at_k": r["pass_at_k"],
                    "response_len_mean": r["response_len"]["mean"],
                    "truncated_ratio": r["truncated_ratio"],
                }
                for r in results
            },
            "details": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone sglang eval")
    parser.add_argument("--model", required=True, help="HF model path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--eval-config", default="eval_only.yaml", help="Eval config yaml")
    parser.add_argument("--n-samples", type=int, default=16, help="Samples per prompt")
    parser.add_argument("--max-len", type=int, default=16384, help="Max response length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--port", type=int, default=30000, help="sglang server port")
    parser.add_argument("--mem-fraction", type=float, default=0.8, help="sglang static mem fraction")
    parser.add_argument("--concurrency", type=int, default=128, help="Max concurrent requests")
    parser.add_argument("--output", default="eval_results.json", help="Save results to JSON file")
    parser.add_argument(
        "--apply-chat-template-kwargs",
        default=None,
        help="JSON object passed to tokenizer.apply_chat_template, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Skip launching sglang (connect to existing server on --port)",
    )
    args = parser.parse_args()
    args.apply_chat_template_kwargs = _parse_json_mapping(
        args.apply_chat_template_kwargs, "--apply-chat-template-kwargs"
    )

    server_proc = None
    try:
        if not args.no_server:
            server_proc = start_sglang_server(args.model, args.tp, args.port, args.mem_fraction)
        asyncio.run(run_eval(args))
    finally:
        if server_proc is not None:
            print("\n[sglang] Shutting down server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()


if __name__ == "__main__":
    main()
