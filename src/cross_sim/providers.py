"""
LLM provider adapters, ported verbatim from PenPal-EMNLP src/nes/simulation.py.

Nothing here is new. It is a copy so the COLING repo can run a simulation
without importing from the replication repo, and so the EMNLP pipeline stays
untouched. Any change to how a turn is generated belongs upstream first.

Context management per provider (matching the experiment's server/adapters):
- OpenAI: native conversation threading via the Conversations API. Only the
          current turn text is sent as 'input'; the provider keeps history.
          System prompt goes in 'instructions'.
- Anthropic: routed through an OpenAI-compatible chat/completions endpoint,
             the same transport the experiment used for Claude. The system
             prompt sits in the messages array, not in a native 'system' field.
- HuggingFace: stateless. Builds [system, ...history, user] and trims to 12k
               characters.

Post-processing: if a response begins with the partner's input
(case-insensitively), the echo is stripped, matching the experiment frontend.
"""

from typing import Optional, List, Dict, Literal
from abc import ABC, abstractmethod


# Provider type definitions
ProviderType = Literal["openai", "anthropic", "huggingface"]


def trim_messages_to_char_limit(messages: List[Dict[str, str]], limit: int = 12000) -> List[Dict[str, str]]:
    """Trim messages to fit within character limit, keeping most recent messages.
    
    Matches the experiment's trimMessagesToCharLimit utility (server/utils/messages.ts).
    Keeps system message (if first) plus as many recent messages as fit within limit.
    Trims from the oldest first, always preserving the most recent messages.
    """
    if not messages:
        return messages
    
    system_message = messages[0] if messages[0].get('role') == 'system' else None
    rest = messages[1:] if system_message else messages[:]
    
    remaining = limit
    kept: List[Dict[str, str]] = []
    
    for msg in reversed(rest):
        content_length = len(msg.get('content', ''))
        if content_length > remaining and kept:
            continue
        remaining -= content_length
        kept.insert(0, msg)
        if remaining <= 0:
            break
    
    if system_message:
        kept.insert(0, system_message)

    return kept


def build_messages_for_compat(
    system_prompt: Optional[str],
    history: Optional[List[Dict[str, str]]],
    user_input: str,
    char_limit: int = 12000,
) -> List[Dict[str, str]]:
    """Mirror of server/utils/messages.ts buildMessagesForCompat.

    Prepends a system message (unless history already starts with one), copies
    history skipping empty content, appends the current user input if non-empty,
    then trims to the char limit.
    """
    history_copy: List[Dict[str, str]] = [dict(m) for m in (history or [])]
    result: List[Dict[str, str]] = []

    if system_prompt:
        first = history_copy[0] if history_copy else None
        if not first or first.get("role") != "system":
            result.append({"role": "system", "content": system_prompt})

    for msg in history_copy:
        content = msg.get("content", "")
        if not content or not content.strip():
            continue
        result.append(msg)

    if user_input and user_input.strip():
        result.append({"role": "user", "content": user_input})

    return trim_messages_to_char_limit(result, limit=char_limit)


def truncate_user_input(text: str, words_to_truncate: int = 2) -> str:
    """Mirror of truncateUserInput from src/utils/submissionHelpers.js.

    Removes the last `words_to_truncate` whitespace-separated words from
    `text`. If the text has fewer words than that, returns an empty string
    (matching the JS behavior of `slice(0, len - n)` when n >= len).
    """
    if not isinstance(text, str):
        return ""
    trimmed = text.strip()
    if not trimmed:
        return ""
    words = trimmed.split()  # \\s+ splits, matches JS split(/\\s+/)
    if words_to_truncate <= 0:
        return trimmed
    clipped = words[: max(0, len(words) - words_to_truncate)]
    return " ".join(clipped)


def strip_input_echo(response_text: str, user_input: str) -> str:
    """Mirror of the echo-strip in src/services/aiService.jsx generateAIResponse.

    If the response begins (case-insensitively) with the partner's input,
    strip that prefix. Applied to all providers in the experiment frontend.
    """
    ai = (response_text or "").strip()
    ui = (user_input or "").strip()
    if ai and ui and ai.lower().startswith(ui.lower()):
        ai = ai[len(ui):].strip()
    return ai


class BaseProvider(ABC):
    """Abstract base class for LLM providers.
    
    Interface matches the experiment's ModelAdapter pattern:
    - system_prompt: Updated each turn with pacing metadata
    - user_input: Just the current turn's text from the partner
    - history: Prior turns as alternating user/assistant messages
    """
    
    @abstractmethod
    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate a response from the model.
        
        Args:
            system_prompt: System instructions with pacing metadata
            user_input: Current turn's text from the partner (not cumulative story)
            history: Prior conversation as alternating user/assistant messages.
                     Ignored by threaded providers (OpenAI). Required by
                     stateless providers (Anthropic, HuggingFace).
            temperature: Sampling temperature (default 1.0 to match experiment)
            max_tokens: Maximum output tokens (default 35 to match experiment)
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def reset_conversation(self):
        """Reset conversation state for a new story."""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI API provider with native conversation threading.
    
    Matches experiment's OpenAIAdapter (server/adapters/OpenAIAdapter.ts):
    - Uses responses.create() with conversation threading
    - Sends system_prompt as 'instructions' (not seeded in conversation items)
    - Sends user_input as 'input' (just current turn text, not cumulative story)
    - Provider manages full history internally via conversation ID
    - history parameter is ignored (provider handles it)
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.conversation_id = None
    
    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate using OpenAI's responses API with conversation threading.
        
        Matches experiment's OpenAIAdapter.respond():
        - Creates conversation with empty items on first call
        - Sends only current user_input as 'input' (not cumulative story)
        - Uses 'instructions' for system prompt (updated each turn)
        - Conversation thread manages all history internally
        - history parameter is ignored
        """
        if self.conversation_id is None:
            # Match experiment: create conversation with empty items
            # (experiment: this.client.conversations.create({ metadata: ..., items: [] }))
            conversation = self.client.conversations.create(
                metadata={"topic": "ai-ai-simulation"},
                items=[]
            )
            self.conversation_id = conversation.id
        
        # Match experiment: send instructions + input only, let thread handle history
        # (experiment: this.client.responses.create({ instructions: systemPrompt, input: userInput, conversation: id }))
        response = self.client.responses.create(
            model=self.model_name,
            conversation=self.conversation_id,
            instructions=system_prompt,
            input=user_input,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Extract text from response (try output_text first, matching experiment)
        out = ""
        if hasattr(response, 'output_text') and response.output_text:
            out = response.output_text
        elif hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content'):
                    if isinstance(item.content, str):
                        out += item.content
                    elif isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                out += content_item.text
        
        return out.strip()
    
    def reset_conversation(self):
        """Reset conversation ID for new story."""
        self.conversation_id = None


class AnthropicProvider(BaseProvider):
    """Claude provider via an OpenAI-compatible endpoint.

    Matches experiment's ClaudeAdapter (server/adapters/ClaudeAdapter.ts), which
    routes Claude through an OpenAI-compatible chat/completions endpoint
    (e.g. OpenRouter) — NOT Anthropic's native messages API.

    - STATELESS: no internal history
    - System prompt is the first message in the messages array
      ({role: 'system', ...}), placed by buildMessagesForCompat
    - Messages trimmed to 12k char limit
    - Requires base_url pointing at the compat endpoint root (e.g.
      "https://openrouter.ai/api/v1"). The OpenAI SDK appends /chat/completions.
    - model_name must be the proxy's identifier for Claude
      (e.g. "anthropic/claude-sonnet-4.5" on OpenRouter).
    """

    def __init__(self, api_key: str, model_name: str, base_url: str):
        from openai import OpenAI
        if not base_url:
            raise ValueError(
                "AnthropicProvider requires base_url (OpenAI-compatible endpoint). "
                "Set 'base_url' in model_configs to match the experiment's MODEL2_BASE_URL."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate via OpenAI-compatible chat completions.

        Mirrors ClaudeAdapter.respond(): builds messages with
        buildMessagesForCompat (system in-array), then POSTs to the compat
        endpoint with the same payload shape (model, messages, temperature,
        max_tokens).
        """
        messages = build_messages_for_compat(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
            char_limit=12000,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        out = response.choices[0].message.content if response.choices else ""
        return out.strip() if out else ""

    def reset_conversation(self):
        """No internal state to reset (stateless provider)."""
        pass


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Inference API provider (OpenAI-compatible).
    
    Matches experiment's VLLM adapters (server/adapters/VLLMAdapterA.ts):
    - STATELESS: no internal conversation state
    - Builds [system, ...history, user: current_input] message array
    - Uses OpenAI-compatible chat completions endpoint
    - Messages trimmed to 12k char limit (matching server/utils/messages.ts)
    """
    
    def __init__(self, api_key: str, model_name: str, base_url: str = "https://router.huggingface.co/v1"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
    
    def generate(self, system_prompt: str, user_input: str,
                 history: Optional[List[Dict[str, str]]] = None,
                 temperature: float = 1.0, max_tokens: int = 35) -> str:
        """Generate using HuggingFace's OpenAI-compatible API (stateless).
        
        Matches experiment's VLLM adapters via buildMessagesForCompat():
        - Builds [system, ...history, user: current_input]
        - No internal conversation state
        - Trims to 12k char limit
        """
        messages = build_messages_for_compat(
            system_prompt=system_prompt,
            history=history,
            user_input=user_input,
            char_limit=12000,
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        out = response.choices[0].message.content if response.choices else ""
        return out.strip() if out else ""
    
    def reset_conversation(self):
        """No internal state to reset (stateless provider)."""
        pass


def get_provider(provider_type: ProviderType, api_key: str, model_name: str, 
                 base_url: str = None) -> BaseProvider:
    """Factory function to create provider instances.
    
    Args:
        provider_type: One of 'openai', 'anthropic', 'huggingface'
        api_key: API key for the provider
        model_name: Model name/identifier
        base_url: Base URL for HuggingFace (OpenAI-compatible endpoint)
        
    Returns:
        Provider instance
    """
    if provider_type == "openai":
        return OpenAIProvider(api_key=api_key, model_name=model_name)
    elif provider_type == "anthropic":
        return AnthropicProvider(api_key=api_key, model_name=model_name, base_url=base_url)
    elif provider_type == "huggingface":
        return HuggingFaceProvider(api_key=api_key, model_name=model_name, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")
