from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/data/namtd12/llm_models/llama-7b'
quant_path = 'quant_models/llm_models/llama-7b'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, save_scaled_model=True)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is scaled and saved at "{quant_path}"')