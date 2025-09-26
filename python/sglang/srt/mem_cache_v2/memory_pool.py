
class MemoryLocation:
    base_ptr: int
    stride: int
    indices: torch.Tensor
    tensor: torch.Tensor

class PhysicalMemoryPool:
    def __init__(self, size: int):
        self.size = torch.empty(size, dtype=torch.int32, device="cuda")

    def get_location(self, tensor_name: str, indices: torch.Tensor) -> MemoryLocation:
        pass
