from dataclasses import dataclass
from typing import Any, Callable, Iterable

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
from sglang.srt.mem_cache_v2.base_cache_index import BaseReqToResourcePool
from sglang.srt.mem_cache_v2.memory_manager import (
    AllocationResult,
    MatchResult,
    MemoryManager,
)


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


@dataclass(frozen=True)
class BufferInfo:
    """Represent the information of a buffer."""

    data_ptr: int
    length: int
    stride: int


KeyFunc_t = Callable[[slice[int]], slice[Any]]


class MemoryManagerAdapter(MemoryManager):
    def __init__(
        self,
        allocator: BaseTokenToKVPoolAllocator | SWATokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        req_to_token_pool: ReqToTokenPool | DecodeReqToTokenPool,
    ):
        super().__init__()
        self.allocator = allocator
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool

    # Allocation
    def allocate_request(
        self, reqs: list[Req], include_last: bool = False
    ) -> AllocationResult:
        """
        This should replace the alloc_extend functions in the allocator.
        """
        pass

    def allocate_tokens(self, reqs: list[Req], token_per_req: int) -> AllocationResult:
        """

        This should replace the alloc_decode functions in the allocator.
        """
        pass

    # Free memory and update index
    def update_cache(self, req: Req):
        """
        Update the cache index for the request.

        Similar to the cache_finished_req and cache_unfinished_req functions but without free.
        """
        pass

    def release_req(self, req: Req, chunked: bool = False):
        """
        Release the request from the cache.
        Free from req_to_resource_pool, and cache index.
        """
        pass

    # Query info & attribute
    def match_prefix(self, req: Req, key_func: KeyFunc_t) -> MatchResult:
        """
        Match the prefix of the request from the cache.
        Args:
            req: The request to match the prefix.
            key_func: The function to generate the key from the token ids.
        Returns:
            The match result.
        """
        pass

    def get_kv_buffer(self, layer_key) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV buffer from the cache.
        Args:
            layer_key: The key of the layer.
        Returns:
            The KV buffer.
        """
        pass

    def set_kv_buffer(self, layer_key, loc, values: Iterable[torch.Tensor]):
        """
        Set the KV buffer to the cache.
        Args:
            layer_key: The key of the layer.
            loc: The location of the buffer.
            tensors: The tensors to set.
        """
        pass

    def get_contiguous_buf_infos(self) -> Iterable[BufferInfo]:
        """
        Get the contiguous buffer infos from the cache.
        """
        pass

    def get_req_to_resource_pool(self):
        """
        Get the req to resource pool from the cache.
        """
        return self.req_to_token_pool
