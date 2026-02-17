"""
Quick test script for LLM providers.

Tests basic connectivity and response from different providers.
"""

import os
import sys
from experiments.llm import create_llm_client


def test_openai(api_key=None, model="gpt-4o-mini"):
    """Test OpenAI API connection."""
    print(f"\n{'='*60}")
    print(f"Testing OpenAI: {model}")
    print(f"{'='*60}")
    
    try:
        client = create_llm_client(
            provider="openai",
            model=model,
            api_key=api_key,
            temperature=0.7,
            max_tokens=100
        )
        
        # Test simple prompt
        test_prompt = "What is 2+2? Answer with just the number."
        print(f"Prompt: {test_prompt}")
        
        response = client.generate(test_prompt)
        print(f"Response: {response}")
        print("✓ OpenAI test successful!")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI test failed: {e}")
        return False


def test_vllm(model="Qwen/Qwen2.5-7B-Instruct"):
    """Test vLLM local model."""
    print(f"\n{'='*60}")
    print(f"Testing vLLM: {model}")
    print(f"{'='*60}")
    
    try:
        print("Loading model (this may take a while)...")
        client = create_llm_client(
            provider="vllm",
            model=model,
            temperature=0.7,
            max_tokens=100,
            gpu_memory_utilization=0.8
        )
        
        # Test simple prompt
        test_prompt = "What is 2+2? Answer with just the number."
        print(f"Prompt: {test_prompt}")
        
        response = client.generate(test_prompt)
        print(f"Response: {response}")
        print("✓ vLLM test successful!")
        return True
        
    except Exception as e:
        print(f"✗ vLLM test failed: {e}")
        return False


def test_reflector_memory(provider="openai", model="gpt-4o-mini", api_key=None):
    """Test Reflector memory system."""
    print(f"\n{'='*60}")
    print(f"Testing Reflector Memory with {provider}/{model}")
    print(f"{'='*60}")
    
    try:
        from experiments.memory import create_memory_system
        
        # Create LLM client
        llm_client = create_llm_client(
            provider=provider,
            model=model,
            api_key=api_key
        )
        
        # Create Reflector memory
        memory = create_memory_system(
            memory_type="reflector",
            llm_client=llm_client,
            reflection_frequency=3
        )
        
        # Simulate some experiences
        print("\nSimulating experiences...")
        for i in range(5):
            memory.update(
                obs_before="000000",
                action=i % 3,
                obs_after=f"{i%2}00000",
                feedback="toggled"
            )
            print(f"  Experience {i+1}/5 recorded")
        
        # Get context
        context = memory.get_context("000000")
        print(f"\nMemory context generated:")
        print(context if context else "(empty)")
        
        stats = memory.get_stats()
        print(f"\nMemory stats: {stats}")
        print("✓ Reflector memory test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Reflector memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM providers")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "vllm", "all"],
                        help="Provider to test")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="Model name")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key (or use OPENAI_API_KEY env var)")
    parser.add_argument("--test_reflector", action="store_true",
                        help="Test Reflector memory system")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    results = []
    
    if args.provider == "openai" or args.provider == "all":
        if not api_key:
            print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or use --api_key")
            sys.exit(1)
        results.append(("OpenAI", test_openai(api_key, args.model)))
    
    if args.provider == "vllm" or args.provider == "all":
        results.append(("vLLM", test_vllm(args.model if args.provider == "vllm" else "Qwen/Qwen2.5-7B-Instruct")))
    
    if args.test_reflector:
        if args.provider == "openai":
            results.append(("Reflector", test_reflector_memory("openai", args.model, api_key)))
        elif args.provider == "vllm":
            results.append(("Reflector", test_reflector_memory("vllm", args.model)))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name}: {status}")
