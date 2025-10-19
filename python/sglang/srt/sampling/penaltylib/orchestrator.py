from __future__ import annotations

import abc
import logging
import weakref
from typing import TYPE_CHECKING, Optional, Set, Type

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type["_BatchedPenalizer"]],
    ):
        import logging
        logger = logging.getLogger(__name__)

        self.vocab_size = vocab_size
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}
        self._backup_reqs = None  # Fallback reqs list for worker thread usage

        logger.info(f"Orchestrator __init__: created new orchestrator, self_id={id(self)}, batch_id={id(batch)}, num_reqs={len(batch.reqs) if batch and batch.reqs else 0}")

        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

        logger.info(f"Orchestrator __init__: initialized, self_id={id(self)}, is_required={is_required}")

    def __getstate__(self):
        # For pickling: convert weakref to None since batch won't survive pickling
        # The batch reference is only needed during initialization
        state = self.__dict__.copy()
        state['_batch_ref'] = None
        return state

    def __setstate__(self, state):
        # For unpickling: restore with a dead weakref
        self.__dict__.update(state)
        if self._batch_ref is None:
            self._batch_ref = lambda: None

    @property
    def batch(self) -> ScheduleBatch | None:
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        # Prefer backup_reqs if it's been set (more up-to-date after merge/filter)
        if self._backup_reqs is not None:
            return self._backup_reqs
        # Fallback to batch.reqs if backup not set
        batch = self.batch
        if batch is not None:
            return batch.reqs
        return None

    def set_backup_reqs(self, reqs):
        """Set fallback reqs list for worker thread usage (when weakref is dead)."""
        self._backup_reqs = reqs

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (torch.Tensor): The output tokens.
        """
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the penalizers to the logits.
        Note that it may apply the penalizers in-place.

        Args:
            logits (torch.Tensor): The logits to apply the penalizers to.

        Returns:
            torch.Tensor: The logits after applying the penalizers.
        """
        for penalizer in self.penalizers.values():
            logits = penalizer.apply(logits)
        return logits

    def filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizers based on the indices to keep in the batch.

        Args:
            keep_indices (torch.Tensor): Tensor of indices to keep in the batch.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Orchestrator filter: self_id={id(self)}, keep={len(keep_indices)}, is_required={self.is_required}")

        if not self.is_required:
            return

        # Clear backup_reqs during filter so penalizers use the actual batch.reqs
        self._backup_reqs = None

        if len(keep_indices) == 0:
            self.is_required = False
            for penalizer in self.penalizers.values():
                penalizer.teardown()
            logger.info(f"Orchestrator filter: teardown (empty batch), self_id={id(self)}")
            return

        is_required = False
        for penalizer in self.penalizers.values():
            tmp_is_required = penalizer.is_required()
            is_required |= tmp_is_required
            if tmp_is_required:
                penalizer.filter(keep_indices=keep_indices)
            else:
                penalizer.teardown()
        self.is_required = is_required

    def merge(self, their: "BatchedPenalizerOrchestrator"):
        """
        Merge the penalizers of another orchestrator into this one.

        Note that this function **must** be called _before_ self.batch.reqs is updated (filtered).
        Each unprepared penalizers would have to be prepared (creating tensors, etc.) first before merging.
        This step requires the original batch.reqs, before it gets merged with other batch.reqs.

        Args:
            their (BatchedPenalizerOrchestrator): The orchestrator to merge into this one.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Orchestrator merge: self_id={id(self)}, their_id={id(their)}, self.is_required={self.is_required}, their.is_required={their.is_required}")

        if not self.is_required and not their.is_required:
            return

        # Clear backup_reqs during merge so penalizers use the actual batch.reqs
        self._backup_reqs = None
        their._backup_reqs = None

        self.is_required = True
        for penalizer, their_penalizer in their.penalizers.items():
            self.penalizers[penalizer].merge(their_penalizer)

        logger.info(f"Orchestrator merge: completed, self_id={id(self)}")

    def get_hard_block_ids(self):
        """Collect per-request hard-block token IDs from penalizers.

        Returns a list of Optional[torch.Tensor] of length batch_size. Each
        entry is a 1-D tensor of token ids that should be hard-blocked for the
        corresponding request; None or empty tensor means no hard block.
        """
        if not self.is_required:
            return None
        reqs = self.reqs()
        if not reqs:
            return None
        # Initialize accumulator per request
        merged: list = [None] * len(reqs)
        for pen in self.penalizers.values():
            lst = pen.get_last_hard_block_ids()
            if not lst:
                continue
            for i, ids in enumerate(lst):
                if ids is None:
                    continue
                if merged[i] is None:
                    merged[i] = ids
                else:
                    # union: concatenate then unique
                    merged[i] = torch.unique(torch.cat([merged[i], ids]))
        return merged

    def get_hard_block_ids_now(self):
        """Compute per-request hard-block token IDs for the current step.

        This asks each penalizer to compute the to-be-blocked ids directly from
        the current request state (BOS/start detection etc.), avoiding reliance
        on whether `_apply` has already run on this rank.
        If a penalizer does not support compute-now, falls back to last ids.
        """
        if not self.is_required:
            return None
        reqs = self.reqs()
        if not reqs:
            return None

        merged: list = [None] * len(reqs)
        for pen in self.penalizers.values():
            pen_name = pen.__class__.__name__
            # Prefer compute-now
            try:
                lst_computed = pen.get_computed_hard_block_ids()
            except Exception as e:
                lst_computed = None

            # Check if lst has any non-None values
            has_values = False
            if lst_computed is not None:
                has_values = any(x is not None for x in lst_computed)
                if has_values and len(lst_computed) > 0:
                    non_none_count = sum(1 for x in lst_computed if x is not None)

            lst = lst_computed
            if not has_values:
                # fall back to last ids if compute-now not available or all None
                lst_last = pen.get_last_hard_block_ids()
                lst = lst_last

            if not lst:
                continue

            for i, ids in enumerate(lst):
                if ids is None:
                    continue
                if merged[i] is None:
                    merged[i] = ids
                else:
                    merged[i] = torch.unique(torch.cat([merged[i], ids]))

        # Final result
        non_none_merged = sum(1 for x in merged if x is not None)
        return merged


class _BatchedPenalizer(abc.ABC):
    """
    An abstract class for a batched penalizer.
    """

    def is_prepared(self) -> bool:
        return self._is_prepared

    def is_required(self) -> bool:
        return self._is_required()

    def prepare(self):
        if not self._is_prepared:
            self._prepare()
            self._is_prepared = True

    def prepare_if_required(self):
        if self._is_required():
            self.prepare()
            return True
        else:
            return False

    def teardown(self):
        self._is_prepared = False

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        if not self._is_prepared:
            return

        self._cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._is_prepared:
            return logits

        self._apply(logits=logits)
        return logits

    def filter(self, keep_indices: torch.Tensor):
        if not self._is_prepared:
            return

        self._filter(keep_indices=keep_indices)

    def merge(self, their: "_BatchedPenalizer"):
        import logging
        logger = logging.getLogger(__name__)

        if not self._is_prepared and not their._is_prepared:
            logger.info(f"{self.__class__.__name__} merge: both unprepared, skipping merge")
            return

        self.prepare()
        their.prepare()
        self._merge(their)

    # Optional: penalizers can expose the ids they hard-blocked at last _apply
    def get_last_hard_block_ids(self):
        return None

    # Optional: compute hard-block ids based on current request state
    def get_computed_hard_block_ids(self):
        return None

    @abc.abstractmethod
    def _is_required(self) -> bool:
        """
        Check if the penalizer is required to be prepared.
        """
        pass

    @abc.abstractmethod
    def _prepare(self):
        """
        Prepare the penalizer.
        Usually, this is where the penalizer initializes its tensors.
        """
        pass

    @abc.abstractmethod
    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        """
        Cumulate the output tokens.
        Orchestrator will call this function to feed the output tokens to the penalizer.
        """
        pass

    @abc.abstractmethod
    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the penalizer to the logits.
        Penalizers can modify the logits in-place if needed.
        """
        pass

    @abc.abstractmethod
    def _filter(self, keep_indices: torch.Tensor):
        """
        Filter the penalizer (tensors or underlying data) based on the indices to keep in the batch.
        """
        pass

    @abc.abstractmethod
    def _merge(self, their: "_BatchedPenalizer"):
        """
        Merge the penalizer with another penalizer.
        """
        pass
