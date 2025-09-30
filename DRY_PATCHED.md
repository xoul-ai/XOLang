This repository is based on SGLang v0.5.2 (checked out from upstream tag) with DRY sampling integrated.

Changes (unstaged):
- New penalizer file: python/sglang/srt/sampling/penaltylib/dry.py
- Registered DRY in sampling pipeline (__init__.py and sampling_batch_info.py)
- OpenAI endpoints: /v1/completions and /v1/chat/completions accept top-level custom_params and extra_body.custom_params (OpenAI SDK) and forward them to the sampler.
