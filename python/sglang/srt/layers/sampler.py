import logging
from typing import List

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")
SYNC_BLOCKED_IDS_ACROSS_TP = get_bool_env_var("SYNC_BLOCKED_IDS_ACROSS_TP")
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tp_group().device_group

        if is_dp_attention_enabled():
            self.tp_sync_group = get_attention_tp_group().device_group

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits

        # DIAGNOSTIC: Log top10 before any additional processing
        try:
            if logger.isEnabledFor(logging.INFO):
                rows = min(len(logits), 2)
                for i in range(rows):
                    topk = torch.topk(logits[i], k=min(10, logits.shape[1]))
                    logger.info(
                        f"SAMPLER_ENTER_TOP10: batch_idx={i} top10_ids={topk.indices.tolist()} top10_logits={[float(x) for x in topk.values]}"
                    )
        except Exception:
            pass

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(logits, sampling_info)
            try:
                if logger.isEnabledFor(logging.INFO):
                    rows = min(len(logits), 2)
                    for i in range(rows):
                        topk = torch.topk(logits[i], k=min(10, logits.shape[1]))
                        logger.info(
                            f"SAMPLER_AFTER_CUSTOM_TOP10: batch_idx={i} top10_ids={topk.indices.tolist()} top10_logits={[float(x) for x in topk.values]}"
                        )
            except Exception:
                pass

        # Optionally synchronize blocked ids across TP to avoid rank divergence
        try:
            if SYNC_BLOCKED_IDS_ACROSS_TP and dist.is_initialized() and self.tp_sync_group is not None:
                hard = (
                    sampling_info.penalizer_orchestrator.get_hard_block_ids()
                    if sampling_info.penalizer_orchestrator is not None
                    else None
                )
                if hard:
                    # Compute local max length
                    local_max = 0
                    for ids in hard:
                        if ids is not None:
                            local_max = max(local_max, int(ids.numel()))
                    # Get global max across TP ranks
                    local_max_tensor = torch.tensor([local_max], device=logits.device, dtype=torch.int32)
                    dist.all_reduce(local_max_tensor, op=dist.ReduceOp.MAX, group=self.tp_sync_group)
                    max_len = int(local_max_tensor.item())
                    if max_len > 0:
                        B = len(hard)
                        pad = torch.full((B, max_len), -1, device=logits.device, dtype=torch.int64)
                        for i, ids in enumerate(hard):
                            if ids is None or ids.numel() == 0:
                                continue
                            l = min(int(ids.numel()), max_len)
                            pad[i, :l] = ids[:l]
                        # All-gather across TP
                        world_size = dist.get_world_size(self.tp_sync_group)
                        gathered = [torch.empty_like(pad) for _ in range(world_size)]
                        dist.all_gather(gathered, pad, group=self.tp_sync_group)
                        # Union per row
                        for i in range(B):
                            union = []
                            for t in gathered:
                                row = t[i]
                                valid = row[row >= 0]
                                if valid.numel() > 0:
                                    union.append(valid)
                            if union:
                                merged = torch.unique(torch.cat(union))
                                hard[i] = merged
                        if logger.isEnabledFor(logging.INFO):
                            # Log a small summary
                            for bi in range(min(B, 2)):
                                ids = hard[bi]
                                if ids is not None and ids.numel() > 0:
                                    logger.info(
                                        f"SAMPLER_SYNC_BLOCKS: batch_idx={bi} union_blocked_count={int(ids.numel())} sample_ids={ids[:min(8, ids.numel())].tolist()}"
                                    )
        except Exception:
            pass

        # Reapply hard blocks from penalizers just before sampling to ensure persistence
        try:
            before = None
            if logger.isEnabledFor(logging.INFO):
                # count how many rows have any hard-blocks
                pass
            sampling_info.enforce_hard_blocks(logits)
            if logger.isEnabledFor(logging.INFO):
                # Log a brief summary of blocked ids (first 2 rows)
                hard = (
                    sampling_info.penalizer_orchestrator.get_hard_block_ids()
                    if sampling_info.penalizer_orchestrator is not None
                    else None
                )
                if hard:
                    for bi in range(min(len(hard), 2)):
                        ids = hard[bi]
                        if ids is None or ids.numel() == 0:
                            continue
                        # report a few blocked ids and their logits
                        sample_ids = ids[: min(8, ids.numel())]
                        vals = logits[bi, sample_ids].tolist()
                        logger.info(
                            f"SAMPLER_REAPPLY: batch_idx={bi} blocked_count={int(ids.numel())} sample_blocked_ids={sample_ids.tolist()} sample_logits={vals}"
                        )
            # Also show top10 after enforcing
            try:
                if logger.isEnabledFor(logging.INFO):
                    rows = min(len(logits), 2)
                    for i in range(rows):
                        topk = torch.topk(logits[i], k=min(10, logits.shape[1]))
                        logger.info(
                            f"SAMPLER_AFTER_ENFORCE_TOP10: batch_idx={i} top10_ids={topk.indices.tolist()} top10_logits={[float(x) for x in topk.values]}"
                        )
            except Exception:
                pass
        except Exception:
            # Never break sampling due to diagnostics
            pass

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        else:
            # Post process original logits. if temperatures are all 1.0, no need to rescale
            if return_logprob and RETURN_ORIGINAL_LOGPROB:
                logprobs = torch.softmax(logits, dim=-1)

            # Post process logits
            logits.div_(sampling_info.temperatures)
            try:
                if logger.isEnabledFor(logging.INFO):
                    rows = min(len(logits), 2)
                    for i in range(rows):
                        logger.info(
                            f"SAMPLER_TEMPERATURE_APPLIED: batch_idx={i} first3_temps={sampling_info.temperatures[:3].view(-1).tolist()}"
                        )
            except Exception:
                pass
            logits[:] = torch.softmax(logits, dim=-1)
            try:
                if logger.isEnabledFor(logging.INFO):
                    rows = min(len(logits), 2)
                    for i in range(rows):
                        topk = torch.topk(logits[i], k=min(10, logits.shape[1]))
                        logger.info(
                            f"SAMPLER_AFTER_SOFTMAX_TOP10: batch_idx={i} top10_ids={topk.indices.tolist()} top10_probs={[float(x) for x in topk.values]}"
                        )
            except Exception:
                pass
            probs = logits

            # Belt-and-suspenders: zero out blocked ids on probs and renormalize
            try:
                hard = (
                    sampling_info.penalizer_orchestrator.get_hard_block_ids()
                    if sampling_info.penalizer_orchestrator is not None
                    else None
                )
                if hard:
                    # Zero masked indices
                    for bi, ids in enumerate(hard):
                        if ids is None or (hasattr(ids, "numel") and ids.numel() == 0):
                            continue
                        probs[bi, ids] = 0.0
                    # Renormalize each row to sum 1.0 (avoid div by zero)
                    row_sums = probs.sum(dim=-1, keepdim=True)
                    zero_mask = row_sums.squeeze(-1) == 0
                    if torch.any(~zero_mask):
                        probs[~zero_mask] = probs[~zero_mask] / row_sums[~zero_mask]
                    if torch.any(zero_mask):
                        # If everything got zeroed (pathological), fall back to original softmax already in `probs`
                        pass
                    if logger.isEnabledFor(logging.INFO):
                        # Log a brief summary for first 2 rows
                        for bi in range(min(len(hard), 2)):
                            ids = hard[bi]
                            if ids is None or (hasattr(ids, "numel") and ids.numel() == 0):
                                continue
                            sample_ids = ids[: min(8, ids.numel())]
                            vals = probs[bi, sample_ids].tolist()
                            logger.info(
                                f"SAMPLER_ZERO_MASK: batch_idx={bi} sample_blocked_ids={sample_ids.tolist()} sample_probs={vals}"
                            )
            except Exception:
                pass
            del logits

            if True:  # Keep this redundant check to simplify some internal code sync
                if global_server_args_dict["sampling_backend"] == "flashinfer":
                    if sampling_info.need_min_p_sampling:
                        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                        batch_next_token_ids = min_p_sampling_from_probs(
                            probs, sampling_info.min_ps
                        )
                    else:
                        batch_next_token_ids = top_k_top_p_sampling_from_probs(
                            probs.contiguous(),
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            filter_apply_order="joint",
                            check_nan=self.use_nan_detection,
                        )
                elif global_server_args_dict["sampling_backend"] == "pytorch":
                    # A slower fallback implementation with torch native operations.
                    batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                        probs,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        sampling_info.min_ps,
                        sampling_info.need_min_p_sampling,
                    )
                else:
                    raise ValueError(
                        f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                    )

            if return_logprob:
                # clamp to avoid -inf
                if RETURN_ORIGINAL_LOGPROB:
                    logprobs = torch.log(logprobs).clamp(
                        min=torch.finfo(logprobs.dtype).min
                    )
                else:
                    logprobs = torch.log(probs).clamp(min=torch.finfo(probs.dtype).min)

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        try:
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    f"SAMPLER_PICKED: next_token_ids={batch_next_token_ids.tolist()}"
                )
        except Exception:
            pass
        return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def sampling_from_probs_torch(probs: torch.Tensor):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering."""
    sampled_index = torch.multinomial(probs, num_samples=1)
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
):
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])

    return (
        output_top_logprobs_val,
        output_top_logprobs_idx,
    )


def get_token_ids_logprobs(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[List[int]],
):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return (
        output_token_ids_logprobs_val,
        output_token_ids_logprobs_idx,
    )


def apply_custom_logit_processor(
    logits: torch.Tensor,
    sampling_batch_info: SamplingBatchInfo,
    num_tokens_in_batch: int = 1,
):
    """Apply custom logit processors to the logits.
    This function will modify the logits in-place.
    num_tokens_in_batch is needed to support spec decoding, where each batch can contain multiple
    tokens. By default, we assume each batch contains only 1 token.
    """

    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "
        f"({num_tokens_in_batch})"
    )

    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)

        # Apply the processor to the logits
        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )
