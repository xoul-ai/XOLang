import logging

import torch
from torch.serialization import add_safe_globals

from sglang.srt.mem_cache.radix_cache import RadixCache

add_safe_globals([torch.nn.modules.container.ParameterList])
logger = logging.getLogger(__name__)


class CartridgeManager:
    def __init__(self, radix_cache: RadixCache):
        self.radix_cache = radix_cache
        self.cartridges = {}

    def reset(self):
        for key in self.cartridges.keys():
            self.load_cartridge(key)

    def clear(self):
        self.cartridges = {}

    def token_id_generator(self, length):
        start_id = -len(self.cartridges)
        return list(range(start_id, length - start_id))

    def load_cartridge(self, file_path: str):
        cartridge_data = torch.load(file_path, map_location="cpu")

        # todo, more robust configuration check
        layers = len(cartridge_data["trainable_keys"])
        num_tokens = (
            cartridge_data["trainable_keys"][0].shape[2]
            + cartridge_data["fixed_keys"][0].shape[2]
        )
        indices = self.radix_cache.token_to_kv_pool_allocator.alloc(num_tokens)
        device = self.radix_cache.token_to_kv_pool_allocator.device

        for layer_idx in range(layers):
            trainable_keys = cartridge_data["trainable_keys"][layer_idx]
            trainable_values = cartridge_data["trainable_values"][layer_idx]
            fixed_keys = cartridge_data["fixed_keys"][layer_idx]
            fixed_values = cartridge_data["fixed_values"][layer_idx]

            k_cache = (
                torch.cat([fixed_keys, trainable_keys], dim=2)
                .squeeze(0)
                .transpose(0, 1)
                .to(device)
            )
            v_cache = (
                torch.cat([fixed_values, trainable_values], dim=2)
                .squeeze(0)
                .transpose(0, 1)
                .to(device)
            )

            self.radix_cache.token_to_kv_pool_allocator.get_kvcache().set_kv_buffer(
                None, indices, k_cache, v_cache, layer_id_override=layer_idx
            )

        # using the file path as the key
        token_ids = self.token_id_generator(num_tokens)
        self.radix_cache.insert(token_ids, indices)
        self.cartridges[file_path] = token_ids
        logger.info(f"Loaded cartridge with {layers=}. {num_tokens=} from {file_path=}")

    def get_cartridge_token_ids(self, key: str):
        token_ids = self.cartridges.get(key, None)
        if token_ids is None:
            return None
        indices, _, _, _ = self.radix_cache.match_prefix(token_ids)
        logger.info(
            f"Cartridge {key} match_prefix got {len(indices)=}, {len(token_ids)=}"
        )
        if len(indices) == len(token_ids):
            return token_ids
        return None

    def get_cartridge_keys(self):
        return list(self.cartridges.keys())
