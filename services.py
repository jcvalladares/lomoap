import torch
import os
import pathlib
import time

# Shared runtime state
models = {}
tokenizers = {}
loading_status = {}
model_device_info = {}
jobs = {}


def _log_job(job_id, msg):
    entry = jobs.get(job_id)
    if not entry:
        return
    entry['logs'].append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# Model configurations with distilled versions
# Keep this here so services handles loading logic
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "description": "Qwen2.5-Coder 1.5B - Distilled, faster, smaller version",
        "config": {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else "auto",
            "device_map": "auto",
            "trust_remote_code": False,
            "low_cpu_mem_usage": True
        }
    },
    "deepseek": {
        "name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        "description": "DeepSeek-Coder-V2 Lite Base - Compact base model",
        "config": {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else "auto",
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
    },
    "qwen_tiny": {
        "name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "description": "Qwen2.5-Coder 0.5B - Ultra-lightweight version",
        "config": {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else "auto",
            "device_map": "auto",
            "trust_remote_code": False,
            "low_cpu_mem_usage": True
        }
    },
    "codellama": {
        "name": "codellama/CodeLlama-7b-Python-hf",
        "description": "CodeLlama 7B Python - Meta's open-source Python-focused model (no chat template)",
        "config": {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else "auto",
            "device_map": "auto",
            "trust_remote_code": False,
            "low_cpu_mem_usage": True
        }
    },
    "deepseek_tiny": {
        "name": "deepseek-ai/deepseek-coder-1.3b-base",
        "description": "DeepSeek-Coder 1.3B Base - Ultra-compact base model",
        "config": {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else "auto",
            "device_map": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
    }
}


def get_gpu_info():
    if not torch.cuda.is_available():
        return {"cuda_available": False, "message": "CUDA not available - running on CPU"}

    gpu_info = {
        "cuda_available": True,
        "gpu_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
        memory_total = props.total_memory / 1024**3
        gpu_info["devices"].append({
            "id": i,
            "name": props.name,
            "memory_total_gb": f"{memory_total:.2f}",
            "memory_allocated_gb": f"{memory_allocated:.2f}",
            "memory_reserved_gb": f"{memory_reserved:.2f}",
            "memory_free_gb": f"{memory_total - memory_reserved:.2f}",
            "utilization_percent": f"{(memory_reserved / memory_total) * 100:.1f}"
        })
    return gpu_info


def detect_model_device(model):
    if model is None:
        return {"status": "not_loaded"}
    device_info = {"devices": [], "primary_device": None, "is_distributed": False}
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_info["is_distributed"] = True
        device_info["device_map"] = model.hf_device_map
        devices = set(str(device) for device in model.hf_device_map.values())
        device_info["devices"] = list(devices)
        device_counts = {}
        for device in model.hf_device_map.values():
            device_str = str(device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1
        device_info["primary_device"] = max(device_counts.keys(), key=lambda k: device_counts[k])
    else:
        try:
            first_param = next(model.parameters())
            device = str(first_param.device)
            device_info["devices"] = [device]
            device_info["primary_device"] = device
        except StopIteration:
            device_info["devices"] = ["unknown"]
            device_info["primary_device"] = "unknown"
    return device_info


def load_model_on_demand(model_key: str):
    global models, tokenizers, loading_status, model_device_info
    if model_key in models and model_key in tokenizers:
        return
    if loading_status.get(model_key) == "loading":
        raise RuntimeError(f"Model {model_key} is currently loading")
    if model_key not in MODEL_CONFIGS:
        raise RuntimeError(f"Model {model_key} not available")
    try:
        # Import transformers lazily to avoid heavy imports at module import time
        from transformers import AutoTokenizer, AutoModelForCausalLM
        loading_status[model_key] = "loading"
        model_info = MODEL_CONFIGS[model_key]
        model_name = model_info["name"]
        config = model_info["config"]
        # unload others
        for other_key in list(models.keys()):
            if other_key != model_key:
                del models[other_key]
                del tokenizers[other_key]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        tokenizers[model_key] = AutoTokenizer.from_pretrained(model_name, trust_remote_code=config.get("trust_remote_code", False))
        models[model_key] = AutoModelForCausalLM.from_pretrained(model_name, **config)
        device_info = detect_model_device(models[model_key])
        model_device_info[model_key] = device_info
        loading_status[model_key] = "ready"
    except Exception as e:
        loading_status[model_key] = "error"
        raise


def generate_with_model(request, model_key: str, model_name: str):
    try:
        # Ensure transformers classes are available lazily
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        load_model_on_demand(model_key)
        model = models[model_key]
        tokenizer = tokenizers[model_key]
        has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
        if has_chat_template:
            messages = [{"role": "user", "content": request.prompt}]
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = request.prompt
        else:
            text = request.prompt
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return {"generated_text": generated_text, "model_used": model_name}
    except Exception as e:
        raise


def _fine_tune_run(model_key: str, req, job_id: str):
    """Synchronous worker that performs fine-tuning. Returns a result dict or raises exceptions.

    This function is intended to be run in a ThreadPoolExecutor from the async handler above.
    It logs progress into the job entry via `_log_job`.
    """
    # Silence tokenizers parallelism warning
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    _log_job(job_id, 'Preparing training data')

    # Import transformers lazily inside the worker to avoid side-effects at module import
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load tokenizer to prepare data
    try:
        base_name = MODEL_CONFIGS[model_key]['name']
        trust_code = MODEL_CONFIGS[model_key]['config'].get('trust_remote_code', False)
        tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=trust_code)
    except Exception as e:
        _log_job(job_id, f'Failed to load tokenizer: {e}')
        return {'error': f'Failed to load tokenizer: {e}'}

    records = []
    try:
        import json
        if req.training_examples:
            for ex in req.training_examples:
                if not isinstance(ex, dict) or 'prompt' not in ex or 'completion' not in ex:
                    continue
                text = ex['prompt'] + (tokenizer.eos_token or '') + ex['completion']
                records.append({'text': text})
        elif req.training_file:
            if not os.path.isfile(req.training_file):
                return {'error': f'Training file not found: {req.training_file}'}

            with open(req.training_file, 'r', encoding='utf-8') as rf:
                content = rf.read().strip()

            parsed = None
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None

            if isinstance(parsed, list):
                for obj in parsed:
                    if not isinstance(obj, dict):
                        continue
                    if 'prompt' in obj and 'completion' in obj:
                        text = obj['prompt'] + (tokenizer.eos_token or '') + obj['completion']
                        records.append({'text': text})
                    elif 'messages' in obj and isinstance(obj['messages'], list):
                        user = None
                        assistant = None
                        for m in obj['messages']:
                            role = m.get('role') if isinstance(m, dict) else None
                            if role == 'user' and user is None:
                                user = m.get('content')
                            if role == 'assistant' and assistant is None:
                                assistant = m.get('content')
                        if user is not None and assistant is not None:
                            text = user + (tokenizer.eos_token or '') + assistant
                            records.append({'text': text})
            else:
                # Try JSONL
                with open(req.training_file, 'r', encoding='utf-8') as rf:
                    for line in rf:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if 'prompt' in obj and 'completion' in obj:
                            text = obj['prompt'] + (tokenizer.eos_token or '') + obj['completion']
                            records.append({'text': text})
                        elif 'messages' in obj and isinstance(obj['messages'], list):
                            user = None
                            assistant = None
                            for m in obj['messages']:
                                role = m.get('role') if isinstance(m, dict) else None
                                if role == 'user' and user is None:
                                    user = m.get('content')
                                if role == 'assistant' and assistant is None:
                                    assistant = m.get('content')
                            if user is not None and assistant is not None:
                                text = user + (tokenizer.eos_token or '') + assistant
                                records.append({'text': text})
        else:
            return {'error': 'No training_examples or training_file provided'}
    except Exception as e:
        _log_job(job_id, f'Failed to process training data: {e}')
        return {'error': f'Failed to process training data: {e}'}

    if not records:
        _log_job(job_id, 'No training records found')
        return {'error': 'No training records found after processing input'}

    _log_job(job_id, f'Prepared {len(records)} training records')

    # Lazy import heavy training libs
    try:
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    except Exception as e:
        _log_job(job_id, f'Missing training dependencies: {e}')
        return {'error': f'Missing training dependencies: {e}'}

    output_dir = req.output_dir or os.path.join('fine_tuned', model_key)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    ds = Dataset.from_list(records)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='longest', max_length=req.max_seq_length)

    # Use a single process for dataset mapping to avoid multiprocessing pickling issues
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=['text'], num_proc=1)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Determine whether to use 8-bit quantization. Note: LoRA/PEFT is frequently
    # incompatible with some bitsandbytes matmul internals (MatmulLtState) and
    # will raise "'MatmulLtState' object has no attribute 'memory_efficient_backward'".
    # As a safe fallback, disable 8-bit when LoRA is requested.
    use_lora = bool(getattr(req, 'use_lora', False))
    if use_lora and getattr(req, 'load_in_8bit', False):
        _log_job(job_id, 'LoRA requested together with 8-bit; disabling 8-bit to avoid bitsandbytes/PEFT incompatibility')
        load_in_8bit = False
    else:
        load_in_8bit = bool(getattr(req, 'load_in_8bit', False))

    # Load base model with optional 8-bit support
    try:
        if load_in_8bit:
            try:
                import bitsandbytes as bnb  # noqa: F401
            except Exception as e:
                _log_job(job_id, 'bitsandbytes not available for 8-bit training')
                return {'error': f'bitsandbytes is required for 8-bit training: {e}'}

            offload_dir = os.path.join(output_dir, 'offload')
            pathlib.Path(offload_dir).mkdir(parents=True, exist_ok=True)

            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception:
                bnb_config = None

            model_kwargs = {
                'trust_remote_code': trust_code,
                'device_map': 'auto',
                'low_cpu_mem_usage': True,
                'offload_folder': offload_dir,
                'offload_buffers': True
            }
            if bnb_config is not None:
                model_kwargs['quantization_config'] = bnb_config
            else:
                model_kwargs['load_in_8bit'] = True

            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                **model_kwargs
            )
            _log_job(job_id, f'Loaded model in 8-bit with device_map=auto and offload_folder={offload_dir}')
            
            # Prepare 8-bit model for training (required for gradient support)
            if not use_lora:
                # When not using LoRA, we need to manually enable gradients on some layers
                _log_job(job_id, 'Warning: 8-bit training without LoRA is not recommended. Forcing requires_grad on model parameters.')
                for param in model.parameters():
                    param.requires_grad = True
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                trust_remote_code=trust_code,
                torch_dtype=(torch.float16 if torch.cuda.is_available() else None),
                low_cpu_mem_usage=True,
                device_map='auto'
            )
            _log_job(job_id, 'Loaded model in fp16 with device_map=auto')
    except Exception as e:
        _log_job(job_id, f'Failed to load base model: {e}')
        return {'error': f'Failed to load base model: {e}'}

    # Optionally apply LoRA via PEFT
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Always prepare model for training when using LoRA
            # This enables gradient checkpointing and casts certain layers to fp32 for stability
            _log_job(job_id, 'Preparing model for LoRA training with PEFT')
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=req.gradient_checkpointing)

            lora_config = LoraConfig(
                r=req.lora_r,
                lora_alpha=req.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=req.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)

            # Ensure input embeddings require grad so LoRA adapters receive gradients
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()

            # Gradient checkpointing requires disabling cache
            if req.gradient_checkpointing and hasattr(model.config, 'use_cache'):
                model.config.use_cache = False

            model.print_trainable_parameters()  # Log trainable params for debugging
            _log_job(job_id, 'Applied LoRA/PEFT to model')
        except Exception as e:
            _log_job(job_id, f'Failed to initialize LoRA/PEFT: {e}')
            return {'error': f'Failed to initialize LoRA/PEFT: {e}'}

    # Enable gradient checkpointing if requested (skip if already done by prepare_model_for_kbit_training)
    if req.gradient_checkpointing and not use_lora:
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                _log_job(job_id, 'Enabled gradient checkpointing')
        except Exception:
            pass
    
    # Verify model has trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    _log_job(job_id, f'Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)')
    
    if trainable_params == 0:
        _log_job(job_id, 'ERROR: No trainable parameters found!')
        return {'error': 'No trainable parameters - model cannot be trained'}

    # Training arguments
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=req.per_device_train_batch_size,
            num_train_epochs=req.num_train_epochs,
            logging_strategy='epoch',
            save_strategy='epoch',
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=req.gradient_checkpointing
        )
    except Exception as e:
        _log_job(job_id, f'Failed to prepare TrainingArguments: {e}')
        return {'error': f'Failed to prepare TrainingArguments: {e}'}

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator
        )
    except Exception as e:
        _log_job(job_id, f'Failed to create Trainer: {e}')
        return {'error': f'Failed to create Trainer: {e}'}

    # Run training
    _log_job(job_id, 'Starting trainer.train()')
    try:
        trainer.train()
        # Save model/adapters
        try:
            model.save_pretrained(output_dir)
            _log_job(job_id, f'Model saved to {output_dir}')
        except Exception as e:
            _log_job(job_id, f'Warning: failed to save model: {e}')

    except Exception as e:
        _log_job(job_id, f'Training failed: {e}')
        return {'error': f'Training failed: {e}'}

    return {
        'message': 'Fine-tuning finished',
        'output_dir': output_dir,
        'num_examples': len(records),
        'use_lora': use_lora
    }
