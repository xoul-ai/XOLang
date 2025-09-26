from abc import ABC

class BaseReqToResourcePool(ABC):
    """A memory pool that maps a request to its resource locations."""
    
    def __init__(self, size: int):
        self.size = size
        self.free_slots = list(range(size))
    
    def alloc(self, need_size: int) -> list[int] | None:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index
        
    def free(self, indices: list[int] | int):
        if isinstance(indices, int):
            self.free_slots.append(indices)
        else:
            self.free_slots.extend(indices)

    def clear(self):
        self.free_slots = list(range(self.size))


class CacheIndex(ABC):
    def __init__(self, req_to_token_pool: BaseReqToResourcePool):
        self.req_to_token_pool = req_to_token_pool