from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.managers.schedule_batch import Req
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class AllocationResult:
    """Represent the result of allocation_request.

    Fields:
        cache_hit_length : Number of cached tokens
        out_cache_loc : Output cache locations for forward batch
        req_pool_idx : Request pool indices
    """
    cached_length: int | None
    out_cache_loc: torch.Tensor
    req_pool_idx: int | None

class MemoryManager:
    def __init__(self):
        pass

    # Allocation
    def allocate_request(self, req: Req, include_last: bool = False) -> AllocationResult:
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

    def cache_unfinished_request(self, req: Req):
        pass




class MemoryManagerAdapter(MemoryManager):
    def __init__(self, allocator: BaseTokenToKVPoolAllocator | SWATokenToKVPoolAllocator, tree_cache: BasePrefixCache):
        super().__init__()
        self.allocator = allocator
        self.tree_cache = tree_cache

    def allocate_request(self, req: Req, include_last: bool = False) -> AllocationResult:
        """
        This should replace the alloc_extend functions in the allocator.
        """
        pass
    
    def allocate_tokens(self, num_tokens: int) -> AllocationResult:
        """

        This should replace the alloc_decode functions in the allocator.
        """
        pass

    # Free
    def cache_finished_request(self, req: Req):
        self.tree_cache.cache_finished_req(req)

    def cache_unfinished_request(self, req: Req, chunked: bool = False):
        self.tree_cache.cache_unfinished_req(req, chunked)