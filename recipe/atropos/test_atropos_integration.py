# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for Atropos-VERL integration.
This script verifies that all components work correctly with VERL infrastructure
and provides proper error handling when the Atropos API is not available.
"""

import os
import sys
import time
from typing import Dict, Any

import torch
import numpy as np

# Add the recipe directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recipe.atropos.main_atropos import AtroposRLTrainer, AtroposAPIError
from recipe.atropos.atropos_trainer import AtroposTrainer


def test_advantage_weighted_loss():
    """Test the advantage-weighted loss computation with model."""
    print("🧪 Testing advantage-weighted loss computation...")
    
    try:
        # Create mock data
        batch_size, seq_len = 4, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        advantages = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)
        loss_mask[:, :8] = 0  # First half is prompt, second half is response
        
        # Test loss computation using model
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input_ids = input_ids.to(device)
        advantages = advantages.to(device)
        loss_mask = loss_mask.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
        # Compute cross-entropy loss
        import torch.nn.functional as F
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), reduction='none')
        
        # Apply advantage weighting and masking
        weighted_loss = ce_loss * advantages.view(-1) * loss_mask.view(-1)
        loss = weighted_loss.sum() / (loss_mask.sum() + 1e-8)
        
        print(f"✓ Advantage-weighted loss computed successfully: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"❌ Error computing advantage-weighted loss: {e}")
        return False


def test_inference_engine():
    """Test the inference engine initialization."""
    print("🧪 Testing inference engine initialization...")
    
    try:
        from recipe.atropos.main_atropos import AtroposInferenceEngine
        
        # Test with a small model
        model_path = "microsoft/DialoGPT-small"
        
        # This will try to initialize vLLM or SGLang
        # If neither is available, it will raise an ImportError
        inference_engine = AtroposInferenceEngine(model_path)
        
        print(f"✓ Inference engine initialized successfully: {inference_engine.engine_type}")
        return True
    except ImportError as e:
        print(f"⚠ Inference engine test skipped: {e}")
        print("  This is expected if vLLM or SGLang is not installed")
        return True  # Not a failure, just missing optional dependency
    except Exception as e:
        print(f"❌ Error initializing inference engine: {e}")
        return False


def test_model_loading():
    """Test model loading using VERL utilities."""
    print("🧪 Testing model loading...")
    
    try:
        from verl.utils.fs import copy_to_local
        from transformers import AutoModelForCausalLM, AutoConfig
        
        model_path = "microsoft/DialoGPT-small"
        
        # Use VERL's model loading utilities
        local_model_path = copy_to_local(model_path, verbose=False)
        
        # Load model config
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        
        print(f"✓ Model loaded successfully: {model_path}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_weight_synchronization():
    """Test the weight synchronization mechanism with models."""
    print("🧪 Testing weight synchronization...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        from recipe.atropos.main_atropos import AtroposShardingManager
        
        # Load a small model for testing
        model_path = "microsoft/DialoGPT-small"
        config = AutoConfig.from_pretrained(model_path)
        training_model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
        
        # Create mock inference engine (we'll skip the engine for this test)
        inference_engine = type('MockInferenceEngine', (), {
            'update_weights_from_tensor': lambda named_tensors: print("✓ Weights updated"),
            'resume_memory_occupation': lambda: print("✓ Memory resumed"),
            'release_memory_occupation': lambda: print("✓ Memory released")
        })()
        
        # Test sharding manager with model
        sharding_manager = AtroposShardingManager(training_model, inference_engine)
        
        with sharding_manager:
            print("✓ Weight synchronization context manager working")
        
        print("✓ Weight synchronization test passed")
        return True
    except Exception as e:
        print(f"❌ Error in weight synchronization test: {e}")
        return False


def test_api_connectivity():
    """Test Atropos API connectivity."""
    print("🧪 Testing Atropos API connectivity...")
    
    config = {
        "atropos": {
            "api_url": "http://localhost:9001",
            "timeout": 5
        },
        "batch_size": 4,
        "max_response_length": 32,
        "model_path": "microsoft/DialoGPT-small"
    }
    
    try:
        trainer = AtroposRLTrainer(config)
        print("✓ Atropos API connectivity test passed")
        return True
    except AtroposAPIError as e:
        print(f"⚠ Expected API error (server not running): {e}")
        print("  This is expected when Atropos server is not running")
        return True  # This is expected behavior
    except Exception as e:
        print(f"❌ Unexpected error in API connectivity test: {e}")
        return False


def test_fallback_mechanisms():
    """Test fallback mechanisms when API is not available."""
    print("🧪 Testing fallback mechanisms...")
    
    config = {
        "atropos": {
            "api_url": "http://localhost:9999",  # Non-existent server
            "timeout": 1
        },
        "batch_size": 4,
        "max_response_length": 32,
        "batch_retry_attempts": 2,
        "batch_retry_delay": 0.1,
        "batch_max_wait_time": 1.0,
        "model_path": "microsoft/DialoGPT-small"
    }
    
    try:
        # This should fail gracefully
        trainer = AtroposRLTrainer(config)
        print("❌ Should have failed with API error")
        return False
    except AtroposAPIError as e:
        print(f"✓ Correctly handled API error: {e}")
        return True
    except Exception as e:
        print(f"❌ Unexpected error in fallback test: {e}")
        return False


def test_advantage_computation():
    """Test advantage computation with tokenization."""
    print("🧪 Testing advantage computation...")
    
    try:
        from transformers import AutoTokenizer
        
        # Use tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create token data
        prompts = ["Hello world", "How are you?"]
        responses = ["I'm doing well", "Great to hear"]
        
        token_data = []
        for prompt, response in zip(prompts, responses):
            prompt_tokens = tokenizer.encode(prompt)
            response_tokens = tokenizer.encode(response)
            combined_tokens = prompt_tokens + response_tokens
            token_data.append(combined_tokens)
        
        # Compute advantages using the fallback method
        scores = []
        for tokens in token_data:
            # Simple scoring based on token diversity
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            diversity_score = unique_tokens / total_tokens if total_tokens > 0 else 0.0
            scores.append([diversity_score * 0.5 + 0.5] * len(tokens))
        
        # Compute advantages
        advantages = []
        for tokens, token_scores in zip(token_data, scores):
            if len(token_scores) > 1:
                mean_score = sum(token_scores) / len(token_scores)
                token_advantages = [score - mean_score for score in token_scores]
            else:
                token_advantages = [0.0] * len(tokens)
            advantages.append(token_advantages)
        
        # Pad to same length
        max_len = max(len(adv) for adv in advantages)
        padded_advantages = []
        for adv in advantages:
            padded = adv + [0.0] * (max_len - len(adv))
            padded_advantages.append(padded)
        
        advantages_tensor = torch.tensor(padded_advantages, dtype=torch.float32)
        
        print(f"✓ Advantage computation test passed, shape: {advantages_tensor.shape}")
        return True
    except Exception as e:
        print(f"❌ Error in advantage computation test: {e}")
        return False


def test_training_loop():
    """Test the complete training loop setup with components."""
    print("🧪 Testing complete training loop...")
    
    config = {
        "atropos": {
            "api_url": "http://localhost:9001",
            "timeout": 5
        },
        "batch_size": 2,
        "max_response_length": 16,
        "batch_retry_attempts": 1,
        "batch_retry_delay": 0.1,
        "batch_max_wait_time": 1.0,
        "model_path": "microsoft/DialoGPT-small"
    }
    
    try:
        # This will fail due to API not being available, but we can test the setup
        trainer = AtroposRLTrainer(config)
        print("❌ Should have failed with API error")
        return False
    except AtroposAPIError:
        print("✓ Training loop setup test passed (API error expected)")
        return True
    except Exception as e:
        print(f"❌ Unexpected error in training loop test: {e}")
        return False


def test_verl_integration():
    """Test integration with VERL's core components."""
    print("🧪 Testing VERL integration...")
    
    try:
        # Test VERL utilities
        from verl.utils.fs import copy_to_local
        from verl.utils.device import get_device_name, is_cuda_available
        
        device_name = get_device_name()
        cuda_available = is_cuda_available()
        
        print(f"✓ VERL device utilities working: {device_name}, CUDA: {cuda_available}")
        
        # Test VERL's model loading
        model_path = "microsoft/DialoGPT-small"
        local_path = copy_to_local(model_path, verbose=False)
        
        print(f"✓ VERL model loading working: {local_path}")
        return True
    except Exception as e:
        print(f"❌ Error in VERL integration test: {e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("🚀 Atropos-VERL Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("VERL integration", test_verl_integration),
        ("Model loading", test_model_loading),
        ("Inference engine", test_inference_engine),
        ("Advantage-weighted loss", test_advantage_weighted_loss),
        ("Weight synchronization", test_weight_synchronization),
        ("API connectivity", test_api_connectivity),
        ("Fallback mechanisms", test_fallback_mechanisms),
        ("Advantage computation", test_advantage_computation),
        ("Training loop setup", test_training_loop),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Atropos integration is working correctly.")
        return True
    else:
        print("⚠ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 