"""
LLM client implementation supporting multiple providers.

Supports:
- vLLM: Local inference with HuggingFace models
- OpenAI: OpenAI API (gpt-4, gpt-3.5-turbo, etc.)
- OpenRouter: OpenRouter API (access to multiple models)
"""

from typing import Optional, Dict, Any
import asyncio
import os
import random as _random


class vLLMClient:
    """
    vLLM client for inference with language models.
    
    Uses vLLM's offline inference API for efficient batched generation.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        Initialize vLLM client.
        
        Args:
            model_name: HuggingFace model ID
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            **kwargs: Additional vLLM arguments
        """
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize vLLM engine
        print(f"Loading vLLM model: {model_name}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            **kwargs
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None
        )
        print(f"vLLM model loaded successfully.")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Override sampling parameters
            
        Returns:
            Generated text
        """
        # Override sampling params if provided
        sampling_params = self.sampling_params
        if kwargs:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stop=kwargs.get('stop', None)
            )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """
        Generate text for multiple prompts (batched for efficiency).
        
        Args:
            prompts: List of input prompts
            **kwargs: Override sampling parameters
            
        Returns:
            List of generated texts
        """
        sampling_params = self.sampling_params
        if kwargs:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stop=kwargs.get('stop', None)
            )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    async def async_generate(self, prompt: str, **kwargs) -> str:
        """Async wrapper: runs vLLM generate in a thread (GPU-bound)."""
        return await asyncio.to_thread(self.generate, prompt, **kwargs)


class OpenAIClient:
    """
    OpenAI API client.
    
    Supports GPT-4, GPT-3.5-turbo, and compatible APIs.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 512,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional base URL for API (for compatible services)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            reasoning_effort: Reasoning effort for o-series / gpt-5 models ("low", "medium", "high").
                              Defaults to "medium" for supported models, None for others.
            **kwargs: Additional arguments
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key parameter.")

        self._api_key = api_key
        self._base_url = base_url

        # reasoning_effort: 명시 지정 없으면 신형 모델에만 medium 기본 적용
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        elif any(model_name.startswith(p) for p in self._NEW_STYLE_PREFIXES):
            self.reasoning_effort = "medium"
        else:
            self.reasoning_effort = None

        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = None  # lazy-initialized in async_generate
        print(f"Using OpenAI API with model: {model_name}"
              + (f", reasoning_effort={self.reasoning_effort}" if self.reasoning_effort else ""))

    # gpt-5, o1, o3, o4 계열은 max_completion_tokens 사용; 그 외는 max_tokens
    _NEW_STYLE_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def _token_param(self, override: Optional[int] = None) -> dict:
        val = override if override is not None else self.max_tokens
        if any(self.model_name.startswith(p) for p in self._NEW_STYLE_PREFIXES):
            return {"max_completion_tokens": val}
        return {"max_tokens": val}

    def _reasoning_param(self, **kwargs) -> dict:
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if effort:
            return {"reasoning_effort": effort}
        return {}

    def _is_new_style(self) -> bool:
        return any(self.model_name.startswith(p) for p in self._NEW_STYLE_PREFIXES)

    def _temperature_param(self, **kwargs) -> dict:
        """신형 모델(gpt-5, o-series)은 temperature를 지원하지 않으므로 생략."""
        if self._is_new_style():
            return {}
        return {"temperature": kwargs.get("temperature", self.temperature)}

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self._temperature_param(**kwargs),
            **self._token_param(kwargs.get('max_tokens')),
            **self._reasoning_param(**kwargs),
        )
        return response.choices[0].message.content.strip()
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Override generation parameters
            
        Returns:
            List of generated texts
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    async def async_generate(
        self,
        prompt: str,
        max_retries: int = 8,
        base_delay: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Async generate with automatic exponential-backoff retry on rate limits.

        Handles openai.RateLimitError (429) and any APIStatusError with status 429.
        Jitter is added to spread retries across concurrent callers.
        """
        import openai as _openai

        if self._async_client is None:
            self._async_client = _openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )

        reasoning_params = self._reasoning_param(**kwargs)
        temperature_params = self._temperature_param(**kwargs)

        for attempt in range(max_retries):
            try:
                resp = await self._async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **temperature_params,
                    **self._token_param(kwargs.get("max_tokens")),
                    **reasoning_params,
                )
                return resp.choices[0].message.content.strip()
            except _openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt) + _random.uniform(0, 1)
                print(f"  [rate_limit] 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
                await asyncio.sleep(delay)
            except _openai.APIStatusError as exc:
                if exc.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + _random.uniform(0, 1)
                    print(f"  [rate_limit] 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
                    await asyncio.sleep(delay)
                else:
                    raise


class OpenRouterClient:
    """
    OpenRouter API client.
    
    Provides access to multiple models through a single API.
    """
    
    def __init__(
        self,
        model_name: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            model_name: Model name (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from parameter or environment
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY or pass api_key parameter.")

        self._api_key = api_key
        self._base_url = "https://openrouter.ai/api/v1"

        # Initialize OpenAI-compatible client with OpenRouter endpoint
        self.client = openai.OpenAI(api_key=api_key, base_url=self._base_url)
        self._async_client = None  # lazy-initialized in async_generate
        print(f"Using OpenRouter with model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Override generation parameters
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )
        return response.choices[0].message.content.strip()
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Override generation parameters
            
        Returns:
            List of generated texts
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    async def async_generate(
        self,
        prompt: str,
        max_retries: int = 8,
        base_delay: float = 1.0,
        **kwargs,
    ) -> str:
        """Async generate with exponential-backoff retry on rate limits (same as OpenAIClient)."""
        import openai as _openai

        if self._async_client is None:
            self._async_client = _openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )

        for attempt in range(max_retries):
            try:
                resp = await self._async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                return resp.choices[0].message.content.strip()
            except _openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt) + _random.uniform(0, 1)
                print(f"  [rate_limit] 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
                await asyncio.sleep(delay)
            except _openai.APIStatusError as exc:
                if exc.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + _random.uniform(0, 1)
                    print(f"  [rate_limit] 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
                    await asyncio.sleep(delay)
                else:
                    raise


class GeminiClient:
    """
    Google Gemini API client via OpenAI-compatible endpoint.

    Uses Google's OpenAI-compatible API (no google-genai SDK required).
    Endpoint: https://generativelanguage.googleapis.com/v1beta/openai/
    API key: GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
    """

    _OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 8192,
        reasoning_effort: str = "medium",
        **kwargs,
    ):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError("openai not installed. Install with: pip install openai")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not provided. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "or pass --api_key."
            )

        self._api_key = api_key
        self.client = _openai.OpenAI(api_key=api_key, base_url=self._OPENAI_COMPAT_BASE_URL)
        self._async_client = None  # lazy-initialized in async_generate
        print(f"Using Google Gemini (OpenAI-compat) with model: {model_name}, reasoning_effort={reasoning_effort}")

    def _build_extra(self, **kwargs) -> dict:
        effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        extra = {}
        if effort:
            extra["extra_body"] = {"reasoning_effort": effort}
        return extra

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **self._build_extra(**kwargs),
        )
        return response.choices[0].message.content.strip()

    def batch_generate(self, prompts: list, **kwargs) -> list:
        return [self.generate(p, **kwargs) for p in prompts]

    async def async_generate(
        self,
        prompt: str,
        max_retries: int = 8,
        base_delay: float = 1.0,
        **kwargs,
    ) -> str:
        """Async generate with exponential-backoff retry on rate limits (429)."""
        import openai as _openai

        if self._async_client is None:
            self._async_client = _openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._OPENAI_COMPAT_BASE_URL,
            )

        extra = self._build_extra(**kwargs)

        for attempt in range(max_retries):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    **extra,
                )
                return response.choices[0].message.content.strip()
            except _openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt) + _random.uniform(0, 1)
                print(f"  [rate_limit] Gemini 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
                await asyncio.sleep(delay)
            except _openai.APIStatusError as exc:
                if exc.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + _random.uniform(0, 1)
                    print(f"  [rate_limit] Gemini 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
                    await asyncio.sleep(delay)
                else:
                    raise


class DummyLLMClient:
    """
    Dummy LLM client for testing without GPU/API.
    
    Returns simple pattern-based responses.
    """
    
    def __init__(self, model_name: str = "dummy", **kwargs):
        self.model_name = model_name
        print(f"Using DummyLLMClient (no actual LLM inference)")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return dummy response."""
        return "0"
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """Return dummy responses for batch."""
        return ["0"] * len(prompts)

    async def async_generate(self, prompt: str, **kwargs) -> str:
        """Async dummy generate."""
        return "0"


def create_llm_client(
    provider: str = "vllm",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    api_key: Optional[str] = None,
    use_dummy: bool = False,
    **kwargs
) -> Any:
    """
    Factory function to create LLM client.

    Args:
        provider: Provider type ('vllm', 'openai', 'openrouter', 'gemini', 'dummy')
        model_name: Model identifier
        api_key: API key for cloud providers
        use_dummy: If True, use DummyLLMClient (overrides provider)
        **kwargs: Additional arguments for LLM client

    Returns:
        LLM client instance
    """
    if use_dummy:
        return DummyLLMClient(model_name=model_name, **kwargs)

    if provider == "vllm":
        return vLLMClient(model_name=model_name, **kwargs)
    elif provider == "openai":
        return OpenAIClient(model_name=model_name, api_key=api_key, **kwargs)
    elif provider == "openrouter":
        return OpenRouterClient(model_name=model_name, api_key=api_key, **kwargs)
    elif provider == "gemini":
        return GeminiClient(model_name=model_name, api_key=api_key, **kwargs)
    elif provider == "dummy":
        return DummyLLMClient(model_name=model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: vllm, openai, openrouter, gemini, dummy"
        )
