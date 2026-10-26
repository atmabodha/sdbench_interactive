import asyncio
import functools
import os
import random
import time
from typing import Any, Callable, Iterable, Protocol, TypeVar, cast

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

MODEL_MAP = {
    "o3": "openai/o3",
    "o3-mini": "openai/o3-mini",
    "o4-mini": "openai/o4-mini",
    "o4-mini-high": "openai/o4-mini-high",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "gpt-5": "openai/gpt-5",
    "gpt-5-high": "openai/gpt-5",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5-mini-high": "openai/gpt-5-mini",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-5.2": "openai/gpt-5.2",
    "gpt-5.2-high": "openai/gpt-5.2",
    "gpt-5-chat": "openai/gpt-5-chat",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "claude-4-sonnet": "anthropic/claude-sonnet-4",
    "claude-4-opus": "anthropic/claude-opus-4",
    "claude-4.1-opus": "anthropic/claude-opus-4.1",
    "claude-4.5-sonnet": "anthropic/claude-sonnet-4.5",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemma3": "google/gemma-3-27b-it",
    "llama-4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "deepseek-r1": "deepseek/deepseek-r1-0528",
    "grok-3": "x-ai/grok-3-beta",
    "grok-3-mini": "x-ai/grok-3-mini-beta",
    "grok-4": "x-ai/grok-4",
    "glm-4.7": "z-ai/glm-4.7",
    "kimi-k2": "moonshotai/kimi-k2",
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
}  # The mapping of sdbench model names to OpenRouter model names

MODEL_ARG_DEFAULTS = {
    "gpt-5-high": {"reasoning_effort": "high"},
    "gpt-5-mini-high": {"reasoning_effort": "high"},
    "gpt-5.2-high": {"reasoning_effort": "high"},
    "glm-4.7": {"extra_body": {"reasoning": {"effort": "high"}}},
    "kimi-k2": {"extra_body": {"reasoning": {"effort": "high"}}},
    "qwen3-235b": {"extra_body": {"reasoning": {"effort": "high"}}},
    "deepseek-v3.2": {"extra_body": {"reasoning": {"effort": "high"}}},
}

AZURE_MODEL_MAP = {
    "gpt-5-high": "gpt-5",
    "gpt-5-mini-high": "gpt-5-mini",
    "gpt-5.2-high": "gpt-5.2",
    "gpt-3.5-turbo": "gpt-35-turbo",
}  # The mapping of sdbench model names to Azure model names

# Type variable for the retry decorator
_F = TypeVar("_F", bound=Callable[..., Any])


def _calculate_retry_delay(
    delay: float, backoff_factor: float, max_delay: float, jitter: bool
) -> tuple[float, float]:
    """Calculate the actual delay and next delay for retry logic.

    Args:
        delay: Current delay value
        backoff_factor: Multiplier for the delay
        max_delay: Maximum delay cap
        jitter: Whether to add randomness

    Returns:
        Tuple of (actual_delay, next_delay)
    """
    # Calculate the delay with optional jitter
    if jitter:
        actual_delay = delay * (0.5 + random.random())
    else:
        actual_delay = delay

    # Calculate next delay with cap
    next_delay = min(delay * backoff_factor, max_delay)

    return actual_delay, next_delay


def _log_retry(error: Exception, actual_delay: float, retry_count: int, max_retries: int) -> None:
    """Log a retry attempt."""
    error_type = type(error).__name__
    print(f"API error: {error_type}: {error!s}")
    print(f"Retrying in {actual_delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")


def _retry_with_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_errors: tuple[type[Exception], ...] = (
        Exception,  # Catch all exceptions - we can refine this later
    ),
) -> Callable[[_F], _F]:
    """Retry decorator with exponential backoff for API calls.

    Args:
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for the delay after each retry
        jitter: Whether to add randomness to the delay
        retryable_errors: Tuple of exceptions that should trigger a retry

    Returns:
        Decorator function
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore
            delay = initial_delay
            retry_count = 0

            while True:
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise

                    actual_delay, delay = _calculate_retry_delay(
                        delay, backoff_factor, max_delay, jitter
                    )
                    _log_retry(e, actual_delay, retry_count, max_retries)
                    time.sleep(actual_delay)

        return cast(_F, wrapper)

    return decorator


def _retry_with_backoff_async(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_errors: tuple[type[Exception], ...] = (
        Exception,  # Catch all exceptions - we can refine this later
    ),
) -> Callable[[_F], _F]:
    """Async retry decorator with exponential backoff for API calls.

    Args:
        max_retries: Maximum number of retries before giving up
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for the delay after each retry
        jitter: Whether to add randomness to the delay
        retryable_errors: Tuple of exceptions that should trigger a retry

    Returns:
        Decorator function
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore
            delay = initial_delay
            retry_count = 0

            while True:
                try:
                    return await func(*args, **kwargs)
                except retryable_errors as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise

                    actual_delay, delay = _calculate_retry_delay(
                        delay, backoff_factor, max_delay, jitter
                    )
                    _log_retry(e, actual_delay, retry_count, max_retries)
                    await asyncio.sleep(actual_delay)

        return cast(_F, wrapper)

    return decorator


@functools.cache
def _get_openrouter_client(api_key_var: str = "OPENROUTER_API_KEY") -> OpenAI:
    api_key = os.getenv(api_key_var)
    if not api_key:
        raise ValueError(f"{api_key_var} environment variable is not set.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


@functools.cache
def _get_async_openrouter_client(api_key_var: str = "OPENROUTER_API_KEY") -> AsyncOpenAI:
    api_key = os.getenv(api_key_var)
    if not api_key:
        raise ValueError(f"{api_key_var} environment variable is not set.")
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


LOCAL_MODEL_PREFIX = "local-"


def _parse_headers(raw: str | dict[str, str] | None) -> dict[str, str] | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SDBENCH_LOCAL_HEADERS must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("SDBENCH_LOCAL_HEADERS must decode to a JSON object.")
    return {str(k): str(v) for k, v in parsed.items()}


def _make_local_client(kwargs: dict[str, Any], debug: bool = False) -> OpenAI:
    """Create a local OpenAI client, extracting local_model_* args from kwargs in-place."""
    base_url = kwargs.pop("local_model_base_url", None) or os.getenv("SDBENCH_LOCAL_BASE_URL")
    api_key = kwargs.pop("local_model_api_key", None) or os.getenv(
        "SDBENCH_LOCAL_API_KEY", "sk-local"
    )
    headers = _parse_headers(
        kwargs.pop("local_model_headers", None) or os.getenv("SDBENCH_LOCAL_HEADERS")
    )
    if not base_url:
        raise ValueError(
            "local_model_base_url model arg or SDBENCH_LOCAL_BASE_URL env var "
            "must be set to use 'local-*' models (e.g. http://localhost:8000/v1)."
        )
    if debug:
        print(
            f"Local client: base_url={base_url}, api_key={'***' if api_key else None}, headers={headers}"
        )
    return OpenAI(base_url=base_url, api_key=api_key, default_headers=headers)


def _make_async_local_client(kwargs: dict[str, Any], debug: bool = False) -> AsyncOpenAI:
    """Create an async local OpenAI client, extracting local_model_* args from kwargs in-place."""
    base_url = kwargs.pop("local_model_base_url", None) or os.getenv("SDBENCH_LOCAL_BASE_URL")
    api_key = kwargs.pop("local_model_api_key", None) or os.getenv(
        "SDBENCH_LOCAL_API_KEY", "sk-local"
    )
    headers = _parse_headers(
        kwargs.pop("local_model_headers", None) or os.getenv("SDBENCH_LOCAL_HEADERS")
    )
    if not base_url:
        raise ValueError(
            "local_model_base_url model arg or SDBENCH_LOCAL_BASE_URL env var "
            "must be set to use 'local-*' models (e.g. http://localhost:8000/v1)."
        )
    if debug:
        print(
            f"Async local client: base_url={base_url}, api_key={'***' if api_key else None}, headers={headers}"
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=headers)


@functools.cache
def _get_azure_client() -> OpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not azure_endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT environment variable is not set - e.g. https://<your-resource>.openai.azure.com/."
        )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    print(f"Using Azure OpenAI endpoint: {azure_endpoint} with API version {api_version}")

    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )


@functools.cache
def _get_async_azure_client() -> AsyncOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not azure_endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT environment variable is not set - e.g. https://<your-resource>.openai.azure.com/."
        )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    print(f"Using Azure OpenAI endpoint: {azure_endpoint} with API version {api_version}")

    return AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )


@_retry_with_backoff()
def _chat_completion_with_retry(
    client: OpenAI, model: str, messages: Iterable[ChatCompletionMessageParam], **kwargs: Any
) -> ChatCompletion:
    response = client.chat.completions.create(
        model=model, messages=messages, stream=False, **kwargs
    )

    # Check if response contains an error field instead of raising an exception
    if hasattr(response, "error") and response.error is not None:  # pyright: ignore[reportAttributeAccessIssue]
        error_message = (
            response.error.get("message", "Unknown error")  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(response.error, dict)  # pyright: ignore[reportAttributeAccessIssue]
            else str(response.error)  # pyright: ignore[reportAttributeAccessIssue]
        )
        error_code = (
            response.error.get("code", "unknown") if isinstance(response.error, dict) else "unknown"  # pyright: ignore[reportAttributeAccessIssue]
        )
        raise Exception(f"API Error ({error_code}): {error_message}")

    return response


@_retry_with_backoff()
def call_llm_with_retry(
    llm: "ModelCreateChatCompletion",
    messages: Iterable[ChatCompletionMessageParam],
) -> ChatCompletion:
    """Call a pre-built ModelCreateChatCompletion callable with retry and backoff."""
    return llm(messages=messages)


@_retry_with_backoff_async()
async def _chat_completion_with_retry_async(
    client: AsyncOpenAI, model: str, messages: Iterable[ChatCompletionMessageParam], **kwargs: Any
) -> ChatCompletion:
    response = await client.chat.completions.create(
        model=model, messages=messages, stream=False, **kwargs
    )

    # Check if response contains an error field instead of raising an exception
    if hasattr(response, "error") and response.error is not None:  # pyright: ignore[reportAttributeAccessIssue]
        error_message = (
            response.error.get("message", "Unknown error")  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(response.error, dict)  # pyright: ignore[reportAttributeAccessIssue]
            else str(response.error)  # pyright: ignore[reportAttributeAccessIssue]
        )
        error_code = (
            response.error.get("code", "unknown") if isinstance(response.error, dict) else "unknown"  # pyright: ignore[reportAttributeAccessIssue]
        )
        raise Exception(f"API Error ({error_code}): {error_message}")

    return response


def create_chat_completion(
    model: str, messages: Iterable[ChatCompletionMessageParam], debug: bool = False, **kwargs: Any
) -> ChatCompletion:
    """Create a chat completion for the given messages.

    Chooses between using OpenRouter or an AI Foundry hosted Azure OpenAI model.

    OpenRouter reads OPENROUTER_API_KEY by default. Set the openrouter_api_key_var
    model argument to read a different environment variable.
    Azure OpenAI requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to be set.

    OpenRouter is used by default. If zero data retention (ZDR) is required (the default),
    OpenRouter is asked to only route to ZDR endpoints via its native `zdr` provider flag, so any
    model slug works without any per-model configuration. Azure OpenAI is only used when explicitly
    forced (see SDBENCH_FORCE_AZURE), e.g. to serve models from your own ZDR Foundry deployment.

    To use Azure OpenAI you must configure your own endpoint in AI Foundry with ZDR. You must
    also provision the models you want to serve using the same models names SDBench is using,
    which are the keys in MODEL_MAP (or AZURE_MODEL_MAP where they differ).

    Configurable with the following environment variables:
    - SDBENCH_FORCE_AZURE - if set to "1" or "true" (case insensitive), forces using Azure OpenAI
        (otherwise OpenRouter is used by default)
    - SDBENCH_OPENROUTER_ZDR_ONLY - if set to "0" or "false" (case insensitive), allows using
        non-zero-data-retention providers on OpenRouter

    Args:
        model: The model to use. May be a key in MODEL_MAP, or any OpenRouter model slug, which is
            passed through to OpenRouter directly.
        messages: The messages to send.
        debug: Whether to enable debug mode for additional sysout.
        **kwargs: Additional arguments to pass to the models.

    Returns:
        The API response
    """
    api_key_var = kwargs.pop("openrouter_api_key_var", "OPENROUTER_API_KEY")
    provider, provider_model, kwargs = _configure_completion(model, **kwargs)
    if provider == "azure":
        client = _get_azure_client()
        if debug:
            print(f"Using Azure OpenAI for model '{provider_model}' with args: {kwargs}")
    elif provider == "openrouter":
        client = _get_openrouter_client(api_key_var)
        if debug:
            print(f"Using OpenRouter for model '{provider_model}' with args: {kwargs}")
    elif provider == "local":
        client = _make_local_client(kwargs, debug=debug)
        if debug:
            print(f"Using local endpoint for model '{provider_model}' with args: {kwargs}")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return _chat_completion_with_retry(client, provider_model, messages, **kwargs)


async def create_chat_completion_async(
    model: str, messages: Iterable[ChatCompletionMessageParam], debug: bool = False, **kwargs: Any
) -> ChatCompletion:
    """Asynchronously create a chat completion for the given messages.

    Implementation matches create_chat_completion with async execution.
    """
    api_key_var = kwargs.pop("openrouter_api_key_var", "OPENROUTER_API_KEY")
    provider, provider_model, kwargs = _configure_completion(model, **kwargs)
    if provider == "azure":
        client = _get_async_azure_client()
        if debug:
            print(f"Using Azure OpenAI async for model '{provider_model}' with args: {kwargs}")
    elif provider == "openrouter":
        client = _get_async_openrouter_client(api_key_var)
        if debug:
            print(f"Using OpenRouter async for model '{provider_model}' with args: {kwargs}")
    elif provider == "local":
        client = _make_async_local_client(kwargs, debug=debug)
        if debug:
            print(f"Using local endpoint async for model '{provider_model}' with args: {kwargs}")
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return await _chat_completion_with_retry_async(client, provider_model, messages, **kwargs)


def _configure_completion(model: str, **kwargs: Any) -> tuple[str, str, Any]:

    if model.startswith(LOCAL_MODEL_PREFIX):
        # Route to local endpoint (e.g. SSH-tunneled inference server). Model name is passed
        # through verbatim; ZDR / provider routing does not apply.
        kwargs.pop("force_azure", None)
        kwargs.pop("openrouter_zdr_only", None)
        default_model_args = MODEL_ARG_DEFAULTS.get(model, {})
        kwargs = {**default_model_args, **kwargs}
        return "local", model, kwargs

    env_force_azure = os.getenv("SDBENCH_FORCE_AZURE", "0").lower() in ("1", "true")
    kwargs_force_azure = kwargs.pop("force_azure", False)
    force_azure = env_force_azure or kwargs_force_azure
    # Default to ZDR only, overrideable with env var or with a (non propogated) model argument
    env_zdr_only = os.getenv("SDBENCH_OPENROUTER_ZDR_ONLY", "1").lower() not in ("0", "false")
    kwargs_zdr_only = kwargs.pop("openrouter_zdr_only", True)
    zdr_only = env_zdr_only and kwargs_zdr_only

    # Unmapped names are passed through as-is: an OpenRouter slug, or an Azure deployment name.
    openrouter_model = MODEL_MAP.get(model, model)

    azure_model = AZURE_MODEL_MAP.get(model, model)  # Default to same model name if not mapped

    default_model_args = MODEL_ARG_DEFAULTS.get(model, {})
    kwargs = {
        **default_model_args,
        **kwargs,
    }  # Merge with passed args, with passed args taking precedence

    # Azure is only used when explicitly forced (e.g. to use your own ZDR Foundry deployment).
    # Otherwise OpenRouter handles ZDR routing natively via the `zdr` provider flag below, including
    # routing OpenAI models to their ZDR Azure endpoints, so no per-model provider map is needed.
    use_azure = force_azure

    if use_azure:
        # Our Foundry hosted Azure OpenAI instance is inherently ZDR
        return "azure", azure_model, kwargs
    else:
        # Use OpenRouter, letting it restrict to Zero Data Retention endpoints when required
        provider_config: dict[str, Any] = {
            # Always disallow any data collecting providers (this is distinct from ZDR)
            "data_collection": "deny"
        }
        if zdr_only:
            # OpenRouter only routes to endpoints with a Zero Data Retention policy
            provider_config["zdr"] = True

        extra_body = kwargs.setdefault("extra_body", {})
        existing_provider = extra_body.get("provider", {})
        extra_body["provider"] = {**provider_config, **existing_provider}

        return "openrouter", openrouter_model, kwargs


class ModelCreateChatCompletion(Protocol):
    """A create_chat_completion function that has the model pre-specified."""

    def __call__(
        self, messages: Iterable[ChatCompletionMessageParam], **kwargs: Any
    ) -> ChatCompletion: ...


def model_create_chat_completion_fn(model: str, **kwargs: Any) -> ModelCreateChatCompletion:
    """Return a function that creates chat completions with the specified model and kwargs.

    Args:
        model: The model to bind.
        **kwargs: Additional arguments to bind to the function.

    Returns:
        A create_chat_completion function with the model and kwargs prespecified.
    """
    return functools.partial(create_chat_completion, model=model, **kwargs)


def print_messages(
    *args: Iterable[ChatCompletionMessageParam]
    | ChatCompletionMessageParam
    | ChatCompletionMessage,
) -> None:
    """Print a list of chat messages in a readable format.

    Args:
        args: Messages to print. Can be individual ChatCompletionMessageParam or
            ChatCompletionMessage objects, an iterable of ChatCompletionMessageParam,
            or a mix of the two. Any iterables will be flattened.
    """
    for arg in args:  # Flatten if an iterable is passed without splat
        if isinstance(arg, Iterable) and not isinstance(arg, (dict, ChatCompletionMessage)):
            for m in arg:
                print_message(m)
        else:
            print_message(cast(ChatCompletionMessageParam | ChatCompletionMessage, arg))


def print_message(message: ChatCompletionMessageParam | ChatCompletionMessage) -> None:
    """Print a message in a readable format.

    Args:
        message: Messages to print.
    """
    header_width = 80
    if isinstance(message, ChatCompletionMessage):
        # Treat as ChatCompletionMessageParam for printing
        message = cast(ChatCompletionAssistantMessageParam, message.model_dump(exclude_unset=True))
    role = message["role"].upper()
    if "name" in message:
        role += f" ({message['name']})"
    outputs: list[str] = []
    outputs.append(f" {role} ".center(header_width, "="))
    content = message.get("content", "[Content Missing]")
    if isinstance(content, str):
        outputs.append(content)
    else:
        # Content may be a list of chunks for streamed responses, but just dump as strings for now.
        outputs.append(str(content))
    remaining_keys = set(message.keys()) - {"role", "content", "name"}
    if remaining_keys:
        remaining_keys = [k for k in remaining_keys if message[k] is not None and message[k] != []]
    if remaining_keys:
        outputs.append(" (other fields) ".center(header_width, "-"))
        for key in remaining_keys:
            outputs.append(f"{key}: {message[key]}")
    outputs.append(f" /{role} ".center(header_width, "="))
    print("\n".join(outputs))


def as_assistant_message(
    completion_message: ChatCompletionMessage,
) -> ChatCompletionAssistantMessageParam:
    """Convert a ChatCompletionMessage to a ChatCompletionAssistantMessageParam.

    This is useful for appending the assistant message to a list of messages, potentially to send
    back to the model in subsequent calls.

    Args:
        completion_message: The message to convert. By definition has the role "assistant".

    Returns:
        The message as a ChatCompletionAssistantMessageParam.
    """
    if completion_message.role != "assistant":
        raise ValueError(f"Expected assistant message, got role: {completion_message.role}")
    # NOTE: We just propagate the content, and not other fields including any reasoning_details
    #   We can convert to ResponsesAPI for reasoning propogation if needed, and deal with the
    #   OpenRouter divergence then
    # output_message_dict = completion_message.model_dump(exclude_unset=True)
    # return cast(ChatCompletionAssistantMessageParam, output_message_dict)
    return ChatCompletionAssistantMessageParam(
        {
            "role": "assistant",
            "content": completion_message.content,
        }
    )
