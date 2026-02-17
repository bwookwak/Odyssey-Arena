"""
LLM client implementation supporting multiple providers.

Supports:
- vLLM: Local inference with HuggingFace models
- OpenAI: OpenAI API (gpt-4, gpt-3.5-turbo, etc.)
- OpenRouter: OpenRouter API (access to multiple models)
"""

from typing import Optional, Dict, Any
import os


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
        temperature: float = 0.7,
        max_tokens: int = 512,
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
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        print(f"Using OpenAI API with model: {model_name}")
    
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
        
        # Initialize OpenAI-compatible client with OpenRouter endpoint
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
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
        provider: Provider type ('vllm', 'openai', 'openrouter', 'dummy')
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
    elif provider == "dummy":
        return DummyLLMClient(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: vllm, openai, openrouter, dummy")
