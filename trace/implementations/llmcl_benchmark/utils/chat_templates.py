"""Chat-template fallbacks needed by base (non-Instruct) backbones."""

import hashlib


LLAMA31_STANDARD_TEMPLATE_VERSION = "llama31_instruct_standard_messages_v1"

# meta-llama/Llama-3.1-8B has the Llama-3 special tokens but intentionally has
# no chat_template. This is the standard-message subset of the official
# Llama-3.1-Instruct template. TRACE supplies only system/user/assistant
# messages, so tool-call branches are neither needed nor silently emulated.
LLAMA31_STANDARD_CHAT_TEMPLATE = r"""{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: 26 Jul 2024\n\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


def ensure_llama31_chat_template(tokenizer, model_path):
    """Install a deterministic fallback only when a Llama tokenizer lacks one."""
    if getattr(tokenizer, "chat_template", None):
        return "tokenizer_config"
    if "llama-3.1" not in str(model_path).lower():
        raise ValueError(
            f"tokenizer at {model_path} has no chat_template and is not "
            "a supported Llama-3.1 tokenizer")
    required = (
        "<|begin_of_text|>", "<|start_header_id|>",
        "<|end_header_id|>", "<|eot_id|>")
    missing = [token for token in required
               if tokenizer.convert_tokens_to_ids(token) is None]
    if missing:
        raise ValueError(
            f"Llama-3.1 tokenizer is missing chat special tokens: {missing}")
    tokenizer.chat_template = LLAMA31_STANDARD_CHAT_TEMPLATE
    return LLAMA31_STANDARD_TEMPLATE_VERSION


def update_fingerprint_for_chat_template(digest, tokenizer_config):
    """Make a base-model token cache depend on its injected template version."""
    if not tokenizer_config.get("chat_template"):
        digest.update(LLAMA31_STANDARD_TEMPLATE_VERSION.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(
            LLAMA31_STANDARD_CHAT_TEMPLATE.encode("utf-8")).digest())
