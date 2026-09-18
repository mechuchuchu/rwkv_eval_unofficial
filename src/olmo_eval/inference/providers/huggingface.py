"""Hugging Face Transformers provider."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import (
    LMOutput,
    LMRequest,
    LogProbEntry,
    RequestType,
    SamplingParams,
)
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.request_utils import chat_messages_for_request
from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

logger = get_logger(__name__)

if TYPE_CHECKING:
    import torch


def _get_device() -> torch.device:
    """Detect the best available device."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_torch_dtype(name: str) -> Any:
    """Map a dtype string to a ``torch.dtype`` (for autocast)."""
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def _ensure_default_rope_registered() -> None:
    """Re-register ``ROPE_INIT_FUNCTIONS["default"]`` if transformers removed it.

    transformers >= 5 dropped the ``"default"`` key from
    ``modeling_rope_utils.ROPE_INIT_FUNCTIONS``, but the Molmo2 modeling code
    bundled inside the released HF checkpoints still references it. Call this
    before ``from_pretrained`` so the model can be instantiated; it is a no-op
    when ``"default"`` is already present. Vendored transformers-compat shim
    (no olmo-core dependency).
    """
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _default_rope(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _default_rope


def _reinit_rope_buffers(model: Any) -> None:
    """Re-initialise non-persistent RoPE ``inv_freq`` buffers after loading.

    transformers >= 5 uses meta-device fast-init in ``from_pretrained`` which
    skips ``__init__`` for non-persistent buffers, leaving ``inv_freq`` as
    uninitialised memory and breaking positional encoding beyond position 0.
    Call this after ``from_pretrained`` on any Molmo2 model loaded with
    ``trust_remote_code=True``. Vendored transformers-compat shim (no olmo-core
    dependency).
    """
    for _, mod in model.named_modules():
        if hasattr(mod, "rope_init_fn") and hasattr(mod, "inv_freq") and hasattr(mod, "config"):
            inv_freq, attn_scaling = mod.rope_init_fn(mod.config, None)
            mod.register_buffer("inv_freq", inv_freq, persistent=False)
            mod.original_inv_freq = mod.inv_freq
            mod.attention_scaling = attn_scaling


def _patch_processor_optional_attribute_kwargs() -> None:
    """Let transformers >= 5 load legacy remote-code processors that forward
    extra (non-sub-processor) keyword arguments to ``ProcessorMixin.__init__``.

    transformers 4.x accepted a processor's ``optional_attributes`` as
    ``__init__`` keyword arguments; transformers 5.x removed that mechanism, so
    ``ProcessorMixin.__init__`` now raises ``TypeError: Unexpected keyword
    argument <x>`` for any kwarg that is not a declared sub-processor attribute.
    The released Molmo2 processor (``trust_remote_code``) still forwards
    optional attributes such as ``image_use_col_tokens`` /
    ``use_single_crop_col_tokens`` to ``super().__init__``. This wrapper strips
    those extra kwargs before the base validation and re-attaches them as plain
    attributes afterward, reproducing the 4.x behaviour the processor relies on
    (e.g. ``self.image_use_col_tokens`` in ``get_image_tokens``). It is a no-op
    for processors that pass no such kwargs and is idempotent. Vendored
    transformers-compat shim (no olmo-core dependency).
    """
    try:
        from transformers.processing_utils import ProcessorMixin
    except ImportError:
        return  # transformers internals unavailable (e.g. mocked in tests)

    original_init = ProcessorMixin.__init__
    if getattr(original_init, "_olmo_eval_optional_attr_patch", False):
        return

    # Handled specially by the base __init__ (popped before kwarg validation).
    _reserved = frozenset({"chat_template", "audio_tokenizer"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        valid = set(self.get_attributes())
        extra = {k: kwargs.pop(k) for k in list(kwargs) if k not in valid and k not in _reserved}
        original_init(self, *args, **kwargs)
        for key, value in extra.items():
            setattr(self, key, value)

    __init__._olmo_eval_optional_attr_patch = True  # ty: ignore[unresolved-attribute]
    ProcessorMixin.__init__ = __init__  # ty: ignore[invalid-assignment]


def _patch_molmo2_generation_cache_position(model: Any) -> None:
    """Make the Molmo2 remote-code generation glue work under transformers >= 5.

    The released Molmo2 ``prepare_inputs_for_generation`` (``trust_remote_code``)
    branches on ``cache_position[0] == 0`` to inject the image/video features on
    the prefill step. transformers 4.x always passed ``cache_position`` as a
    keyword argument; transformers 5.x no longer maintains it in the generation
    kwargs (it computes a fresh ``cache_position`` inside the base
    ``prepare_inputs_for_generation`` and returns it in ``model_inputs``
    instead), so the Molmo2 method receives ``cache_position=None`` and raises
    ``TypeError: 'NoneType' object is not subscriptable``. This wrapper keeps
    the original feature-injection logic but derives the prefill condition from
    the ``cache_position`` transformers computes, falling back to
    ``past_key_values is None`` (the definition of the prefill step). No-op for
    non-Molmo2 models and idempotent. Vendored transformers-compat shim (no
    olmo-core dependency).
    """
    cls = type(model)
    if cls.__name__ != "Molmo2ForConditionalGeneration":
        return
    original = cls.prepare_inputs_for_generation
    if getattr(original, "_olmo_eval_cache_position_patch", False):
        return

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_token_pooling=None,
        image_grids=None,
        image_num_crops=None,
        pixel_values_videos=None,
        video_token_pooling=None,
        video_grids=None,
        attention_mask=None,
        token_type_ids=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super(cls, self).prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        cp = cache_position if cache_position is not None else model_inputs.get("cache_position")
        is_prefill = int(cp[0]) == 0 if cp is not None else past_key_values is None
        if is_prefill:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids
        return model_inputs

    prepare_inputs_for_generation._olmo_eval_cache_position_patch = True  # ty: ignore[unresolved-attribute]
    cls.prepare_inputs_for_generation = prepare_inputs_for_generation


class HuggingFaceProvider(InferenceProvider):
    """Provider using Hugging Face Transformers for local inference.

    Supports text-only causal LMs (the default path) and, when
    ``multimodal=True``, image-text-to-text models such as the released
    ``allenai/Molmo2-4B`` checkpoint via ``AutoProcessor`` +
    ``AutoModelForImageTextToText``. Images travel on ``LMRequest.images``.
    """

    model: Any
    tokenizer: Any
    processor: Any
    device: torch.device
    is_multimodal: bool
    max_crops: int
    autocast_dtype: str | None

    # kwargs that may be passed by the runner but are not valid for HF from_pretrained
    _IGNORED_KWARGS = frozenset(
        {
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "attention_backend",
            "use_tqdm_on_load",
            "add_bos_token",
            "max_model_len",
            "load_format",
            "model_loader_extra_config",
            "enable_auto_tool_choice",
        }
    )

    _TOKENIZER_KWARGS = frozenset(
        {
            "cache_dir",
            "force_download",
            "local_files_only",
            "revision",
            "token",
            "trust_remote_code",
        }
    )

    def __init__(
        self,
        model_name: str,
        tokenizer: str | None = None,
        *,
        multimodal: bool = False,
        max_crops: int = 24,
        autocast_dtype: str | None = None,
        batch_size: int = 8,
        **model_kwargs,
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: HuggingFace model identifier or local path.
            tokenizer: Tokenizer path/identifier. If not specified, uses the model path.
            multimodal: Load an image-text-to-text model (AutoProcessor +
                AutoModelForImageTextToText) instead of a text-only causal LM.
            max_crops: Maximum image crops passed to the multimodal processor.
            autocast_dtype: If set (e.g. ``"bfloat16"``), run multimodal generation under
                ``torch.autocast`` with this dtype. Pair with fp32 weights (``dtype="float32"``)
                to match mm_olmo's ``amp_bf16`` eval numerics (fp32 master weights + bf16
                autocast); the model keeps attention in fp32 via its own ``float32_attention``.
            batch_size: Maximum number of text requests processed in one model call.
            **model_kwargs: Additional arguments passed to from_pretrained.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Strip kwargs meant for other providers (e.g., vLLM)
        for key in self._IGNORED_KWARGS:
            model_kwargs.pop(key, None)

        super().__init__(model_name)
        self.is_multimodal = bool(multimodal)
        self.supports_images = self.is_multimodal
        self.max_crops = int(max_crops)
        self.autocast_dtype = autocast_dtype
        self.batch_size = int(batch_size)
        self.processor = None
        self.device = _get_device()
        if self.is_multimodal:
            self._init_multimodal(model_name, model_kwargs)
        else:
            self._init_text(model_name, tokenizer, model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _init_text(
        self, model_name: str, tokenizer: str | None, model_kwargs: dict[str, Any]
    ) -> None:
        """Load a text-only causal LM (tokenizer + AutoModelForCausalLM)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for HuggingFaceProvider. "
                "Install with: pip install transformers"
            ) from e

        tokenizer_path = tokenizer or model_name
        tokenizer_kwargs = {
            key: value for key, value in model_kwargs.items() if key in self._TOKENIZER_KWARGS
        }
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    def _init_multimodal(self, model_name: str, model_kwargs: dict[str, Any]) -> None:
        """Load an image-text-to-text model (AutoProcessor + AutoModelForImageTextToText).

        Applies the transformers-compat RoPE shims required by the released
        Molmo2 ``trust_remote_code`` checkpoints, and reuses the processor's
        tokenizer as ``self.tokenizer`` so the shared decode / stop-sequence
        helpers work unchanged.
        """
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers>=5.4.0 is required for multimodal HuggingFaceProvider. "
                "Install with: pip install 'transformers>=5.4.0'"
            ) from e

        _ensure_default_rope_registered()
        _patch_processor_optional_attribute_kwargs()

        processor_kwargs = {
            key: value for key, value in model_kwargs.items() if key in self._TOKENIZER_KWARGS
        }
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        _reinit_rope_buffers(self.model)
        _patch_molmo2_generation_cache_position(self.model)
        self.tokenizer = self.processor.tokenizer

    def get_tokenizer(self) -> Any:
        """Get the tokenizer for this provider."""
        return self.tokenizer

    def _context_length(self) -> int:
        """Model context length, used to cap uncapped (max_tokens=None) generation."""
        for attr in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            value = getattr(self.model.config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return 2048

    def _build_generate_kwargs(self, params: SamplingParams, prompt_len: int = 0) -> dict[str, Any]:
        """Convert SamplingParams to HuggingFace generate kwargs."""
        # Use explicit do_sample flag, overriding temperature-based inference
        do_sample = params.do_sample and params.temperature > 0

        # max_tokens=None means "uncapped"; transformers would otherwise fall back
        # to its tiny default max_length. max_new_tokens counts generated tokens on
        # top of the prompt, so reserve the room left in the context after it.
        if params.max_tokens is not None:
            max_new_tokens = params.max_tokens
        else:
            max_new_tokens = max(self._context_length() - prompt_len, 1)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            if params.temperature > 0:
                kwargs["temperature"] = params.temperature
            if params.top_p is not None:
                kwargs["top_p"] = params.top_p
            if params.top_k is not None:
                kwargs["top_k"] = params.top_k

        return kwargs

    def _ensure_padding_token(self) -> int:
        """Return the tokenizer padding ID, deriving it from EOS when necessary."""
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token is None or eos_token_id is None:
                raise ValueError("Batching Hugging Face requests requires a padding or EOS token")
            self.tokenizer.pad_token = eos_token
            pad_token_id = eos_token_id
        return int(pad_token_id)

    def _tokenize_text_batch(self, prompts: list[str]) -> Any:
        """Tokenize and move a padded text batch to the model device."""
        import torch

        self._ensure_padding_token()
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        if hasattr(encoded, "to"):
            return encoded.to(self.device)
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }

    def _eos_token_ids(self) -> set[int]:
        """Return EOS IDs used to trim padded generation rows."""
        generation_config = getattr(self.model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, int):
            return {eos_token_id}
        return {int(token_id) for token_id in eos_token_id}

    def _trim_generated_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Trim tokens emitted after the first EOS in a padded generation row."""
        eos_ids = self._eos_token_ids()
        if not eos_ids:
            return generated_ids
        for index, token_id in enumerate(generated_ids.tolist()):
            if token_id in eos_ids:
                return generated_ids[: index + 1]
        return generated_ids

    def _generation_log_probs(
        self,
        output_ids: torch.Tensor,
        input_attention_mask: torch.Tensor,
        input_width: int,
    ) -> torch.Tensor:
        """Compute generated-token log probabilities for a padded batch."""
        import torch

        generated_width = output_ids.shape[1] - input_width
        if generated_width <= 0:
            return output_ids.new_empty((output_ids.shape[0], 0, 0), dtype=torch.float32)

        generated_mask = torch.ones(
            (output_ids.shape[0], generated_width),
            dtype=input_attention_mask.dtype,
            device=output_ids.device,
        )
        attention_mask = torch.cat([input_attention_mask, generated_mask], dim=1)
        positions = torch.arange(
            input_width - 1,
            output_ids.shape[1] - 1,
            device=output_ids.device,
            dtype=torch.long,
        )
        logits = self._forward_selected_logits(output_ids, attention_mask, positions)
        return torch.log_softmax(logits, dim=-1)

    def _forward_selected_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits at selected positions without materializing the full LM head."""
        import torch

        if positions.numel() == 0:
            return input_ids.new_empty((input_ids.shape[0], 0, 0), dtype=torch.float32)

        try:
            with torch.no_grad():
                logits = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=positions,
                ).logits
        except TypeError as error:
            if "logits_to_keep" not in str(error):
                raise
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits
            return logits[:, positions, :]

        if logits.shape[1] == input_ids.shape[1]:
            return logits[:, positions, :]
        return logits

    def _build_generation_output(
        self,
        generated_ids: torch.Tensor,
        log_probs: torch.Tensor | None,
        params: SamplingParams,
    ) -> LMOutput:
        """Build one output row from generated tokens and optional log probabilities."""
        gen_ids = self._trim_generated_ids(generated_ids)
        gen_ids, text = self._truncate_at_stop(gen_ids, params.stop_sequences)

        logprob_entries: list[LogProbEntry] | None = None
        metadata: dict[str, Any] = {}
        if len(gen_ids) > 0 and log_probs is not None:
            logprob_entries = []
            for i, tok in enumerate(gen_ids):
                logprob = log_probs[i, tok].item()
                token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                logprob_entries.append(
                    {
                        "token": token_str,
                        "logprob": logprob,
                        "token_id": int(tok),
                        "bytes": list(token_str.encode("utf-8")),
                    }
                )

            sum_logits = sum(entry["logprob"] for entry in logprob_entries)
            num_tokens = len(logprob_entries)
            metadata = {
                "sum_logits": sum_logits,
                "num_tokens": num_tokens,
                "num_tokens_all": num_tokens,
            }

        return LMOutput(text=text.strip(), logprobs=logprob_entries, metadata=metadata)

    def _truncate_at_stop(
        self, tokens: torch.Tensor, stop_sequences: tuple[str, ...] | None
    ) -> tuple[torch.Tensor, str]:
        """Truncate generated tokens at first stop sequence."""
        if not stop_sequences:
            return tokens, self.tokenizer.decode(tokens, skip_special_tokens=True)

        decoded_parts: list[str] = []
        for idx, token in enumerate(tokens):
            decoded_parts.append(self.tokenizer.decode(token, skip_special_tokens=True))
            decoded = "".join(decoded_parts)
            for stop in stop_sequences:
                if stop in decoded:
                    return tokens[: idx + 1], decoded.split(stop)[0]

        return tokens, "".join(decoded_parts)

    def _build_chat_messages(self, request: LMRequest) -> list[dict[str, Any]]:
        """Build processor chat messages, injecting images into the user turn.

        Mirrors the released Molmo2 HF eval: a single user turn whose content is
        the image(s) followed by the question text. A system message (if present)
        is preserved as a text-only turn; images attach to the first user turn.
        """
        image_parts = [{"type": "image", "image": img} for img in (request.images or ())]
        messages = chat_messages_for_request(request)

        chat: list[dict[str, Any]] = []
        attached = False
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "") or ""
            if role == "user" and not attached and image_parts:
                chat.append(
                    {"role": role, "content": [*image_parts, {"type": "text", "text": text}]}
                )
                attached = True
            else:
                chat.append({"role": role, "content": [{"type": "text", "text": text}]})
        if not attached and image_parts:
            chat.insert(0, {"role": "user", "content": image_parts})
        return chat

    def _generate_multimodal(
        self, requests: list[LMRequest], params: SamplingParams
    ) -> list[list[LMOutput]]:
        """Greedy/sampled generation for image-text-to-text models.

        Ports the verified released-Molmo2 HF path: build the processor chat
        template with image+text content, run ``model.generate``, and decode the
        newly generated tokens.
        """
        import torch

        # Optionally run generation under autocast (e.g. bf16 over fp32 weights) to match
        # mm_olmo's amp_bf16 numerics. `getattr` guard keeps this a no-op under mocked torch.
        autocast = getattr(torch, "autocast", None)
        if self.autocast_dtype and autocast is not None:
            ac_dtype = _to_torch_dtype(self.autocast_dtype)
            ac_device = self.device.type

            def _autocast_ctx() -> Any:
                return autocast(device_type=ac_device, dtype=ac_dtype)
        else:
            _autocast_ctx = contextlib.nullcontext

        results = []
        for request in requests:
            chat = self._build_chat_messages(request)
            pil_images = []
            for img in request.images or ():
                if hasattr(img, "mode") and img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)

            text = self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            if pil_images:
                inputs = self.processor(
                    images=pil_images, text=text, max_crops=self.max_crops, return_tensors="pt"
                )
            else:
                inputs = self.processor(text=text, return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[1]
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            gen_kwargs = self._build_generate_kwargs(params, prompt_len)

            request_outputs = []
            for _ in range(params.num_samples):
                with _autocast_ctx(), torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_kwargs)[0]
                gen_ids = output_ids[prompt_len:]
                gen_ids, generated = self._truncate_at_stop(gen_ids, params.stop_sequences)
                num_tokens = int(len(gen_ids))
                request_outputs.append(
                    LMOutput(
                        text=generated.strip(),
                        logprobs=None,
                        metadata={"num_tokens": num_tokens, "num_tokens_all": num_tokens},
                    )
                )
            results.append(request_outputs)

        return results

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        import torch

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the HuggingFaceProvider and will not be used."
            )
        if self.is_multimodal:
            return self._generate_multimodal(requests, params)

        results: list[list[LMOutput]] = [[] for _ in requests]
        batch_size = getattr(self, "batch_size", 1)
        for batch_start in range(0, len(requests), batch_size):
            request_batch = requests[batch_start : batch_start + batch_size]
            encoded = self._tokenize_text_batch([request.prompt for request in request_batch])
            input_width = encoded["input_ids"].shape[1]
            prompt_len = int(encoded["attention_mask"].sum(dim=1).max().item())
            gen_kwargs = self._build_generate_kwargs(params, prompt_len)

            for _ in range(params.num_samples):
                with torch.no_grad():
                    output_ids = self.model.generate(**encoded, **gen_kwargs)

                generated_ids = output_ids[:, input_width:]
                log_probs = self._generation_log_probs(
                    output_ids, encoded["attention_mask"], input_width
                )
                for row, request_outputs in enumerate(
                    results[batch_start : batch_start + len(request_batch)]
                ):
                    request_outputs.append(
                        self._build_generation_output(generated_ids[row], log_probs[row], params)
                    )

        return results

    def describe_request(
        self,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> dict[str, Any] | None:
        params = self._default_sampling_params(sampling_params)
        trace = super().describe_request(request, sampling_params)
        if trace is None:
            return None

        if request.request_type != RequestType.LOGLIKELIHOOD:
            # Pass the prompt length so an uncapped (max_tokens=None) trace shows the
            # same per-prompt max_new_tokens budget that generate() will use.
            prompt_len = len(self.tokenizer.encode(request.prompt)) if request.prompt else 0
            trace["provider"] = "HuggingFaceProvider"
            trace["endpoint"] = "transformers.generate"
            trace["generation_kwargs"] = {
                "max_gen_toks": params.max_tokens,
                **self._build_generate_kwargs(params, prompt_len),
            }
            trace["stop_sequences"] = list(params.stop_sequences or ())
        return trace

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        if self.is_multimodal:
            raise NotImplementedError(
                "Multimodal HuggingFaceProvider supports generation only, "
                "not loglikelihood scoring."
            )
        import torch

        params = self._default_sampling_params(sampling_params)
        if params.truncate_prompt_tokens is not None or params.truncation_side is not None:
            logger.warning(
                "truncate_prompt_tokens or truncation_side has been set in the params, "
                "but is not supported for the HuggingFaceProvider and will not be used."
            )
        results: list[list[LMOutput]] = [[] for _ in requests]
        work_items: list[tuple[int, str, list[int], list[int], int]] = []
        for request_idx, request in enumerate(requests):
            cont_prompts = request.continuation_prompts
            for continuation_idx, continuation in enumerate(request.continuations or ()):
                prompt = cont_prompts[continuation_idx] if cont_prompts else request.prompt
                context_enc, continuation_enc = encode_context_and_continuation(
                    self.tokenizer, prompt, continuation
                )
                work_items.append(
                    (
                        request_idx,
                        continuation,
                        context_enc + continuation_enc,
                        continuation_enc,
                        len(context_enc),
                    )
                )

        pad_token_id = self._ensure_padding_token()
        for batch_start in range(0, len(work_items), getattr(self, "batch_size", 1)):
            batch = work_items[batch_start : batch_start + getattr(self, "batch_size", 1)]
            max_length = max(len(item[2]) for item in batch)
            input_ids = torch.full(
                (len(batch), max_length),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            offsets: list[int] = []
            for row, (_, _, full_ids, _, _) in enumerate(batch):
                offset = max_length - len(full_ids)
                offsets.append(offset)
                token_ids = torch.tensor(full_ids, dtype=torch.long, device=self.device)
                input_ids[row, offset:] = token_ids
                attention_mask[row, offset:] = 1

            positions = sorted(
                {
                    offset + context_len + continuation_idx - 1
                    for offset, (_, _, _, continuation_enc, context_len) in zip(
                        offsets, batch, strict=True
                    )
                    for continuation_idx in range(len(continuation_enc))
                }
            )
            position_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
            logits = self._forward_selected_logits(input_ids, attention_mask, position_tensor)
            log_probs = torch.log_softmax(logits, dim=-1)
            position_indices = {position: index for index, position in enumerate(positions)}

            for row, (request_idx, continuation, _, continuation_enc, context_len) in enumerate(
                batch
            ):
                logprob_entries: list[LogProbEntry] = []
                total = 0.0
                offset = offsets[row]
                for j, tok in enumerate(continuation_enc):
                    position = offset + context_len + j - 1
                    lp = log_probs[row, position_indices[position], tok].item()
                    token_str = self.tokenizer.decode(tok, skip_special_tokens=False)
                    logprob_entries.append(
                        {
                            "token": token_str,
                            "logprob": lp,
                            "token_id": int(tok),
                            "bytes": list(token_str.encode("utf-8")),
                        }
                    )
                    total += lp

                results[request_idx].append(
                    LMOutput(
                        text=continuation,
                        logprobs=logprob_entries,
                        metadata={"total_logprob": total},
                    )
                )

        return results

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate completions.

        Runs the synchronous HuggingFace generate in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests to process.
            sampling_params: Sampling configuration.

        Returns:
            List of output lists, one per request.
        """
        return await asyncio.to_thread(self.generate, requests, sampling_params)

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async compute logprobs for continuations.

        Runs the synchronous HuggingFace logprobs in a thread pool to avoid blocking.

        Args:
            requests: Batch of requests with continuations to score.

        Returns:
            List of output lists with logprobs populated.
        """
        return await asyncio.to_thread(self.logprobs, requests, sampling_params)
