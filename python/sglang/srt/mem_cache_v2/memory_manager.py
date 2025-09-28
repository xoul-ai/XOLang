from dataclasses import dataclass
from typing import Callable

import torch

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import TreeNode


@dataclass(frozen=True)
class AllocationResult:
    """Represent the result of allocation_request.

    Fields:
        cache_hit_length : Number of cached tokens
        out_cache_loc : Output cache locations for forward batch
        req_pool_idx : Request pool indices
        alloation_key: Key for cache index to manage the allocation of the request.
    """

    cached_length: int | None
    out_cache_loc: torch.Tensor
    req_pool_idx: int | None
    # Adapter   radix cache |  swa radix cache
    # For V2, the allocation key will be a per-request key and its mapping is managed by the cache index.
    alloation_key: TreeNode | tuple[TreeNode, int | None]


@dataclass(frozen=True)
class MatchResult:
    """Represent the result of match_prefix.

    Fields:
        prefix_indices: Indices of the KV cache of the matched prefix.
        matched_length: Length of matched prefix. Note that this may be different from len(prefix_indices).
    """

    prefix_indices: torch.Tensor
    matched_length: int


class MemoryManager:
    def __init__(self):
        pass

    # Allocation
    def allocate_request(
        self, req: Req, include_last: bool = False
    ) -> AllocationResult:
        """
        This function will allocate the memory for all the tokens in the request.
        If include_last is True, req.seqlen tokens will be allocated. (e.g. prefill, retract)
        If include_last is False, req.seqlen - 1 tokens will be allocated. (e.g. PD)
        """
        token_ids = req.origin_input_ids + req.output_ids
        if not include_last:
            token_ids = token_ids[:-1]
        pass

    def allocate_tokens(self, num_tokens: int) -> AllocationResult:
        """
        This function is called to allocate a number of tokens.
        The typical usage is to allocate for decode.
        """
        pass

    # Free
    def cache_finished_request(self, req: Req):
        pass

    def acache_unfinished_request(self, req: Req):
        pass
