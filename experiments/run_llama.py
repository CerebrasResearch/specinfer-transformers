import sys
sys.path
sys.path.append('..')
from src.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, GenerationConfig
from huggingface_hub import HfApi
from pathlib import Path
import numpy as np
from huggingface_hub import login

login("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")


llm_path = "meta-llama/Llama-2-7b-hf"

llm = AutoModelForCausalLM.from_pretrained(llm_path)
ssm1 = AutoModelForCausalLM.from_pretrained(Path("/cb/cold/andrewz/cerebras-research/models/llama2-7b-s50-no-train"))
ssm2 = AutoModelForCausalLM.from_pretrained(Path("/cb/cold/andrewz/cerebras-research/models/llama2-7b-s50-retrain"))
ssm3 = AutoModelForCausalLM.from_pretrained(Path("/cb/cold/andrewz/cerebras-research/models/llama2-7b-s75-retrain"))

# ssm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenizer = LlamaTokenizer.from_pretrained(llm_path)
ssm_generation_config = GenerationConfig(return_dict_in_generate = True, 
                                     eos_token_id = 2, 
                                     num_assistant_tokens_schedule="constant",
                                     num_assistant_tokens=9)

llm_generation_config = GenerationConfig(return_dict_in_generate = True, 
                                     eos_token_id = 2, 
                                     num_assistant_tokens_schedule="constant",
                                     num_assistant_tokens=9,
                                     use_target_kv_cache=True)

ssms = [ssm1, ssm2, ssm3]
prompts = [
           "Implement a Python function to compute the Fibonacci numbers. def",
           "Who does Harry turn into a balloon? <eos>",
           "What were the major contributing factors to the fall of the Roman Empire? <eos>",
           "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts. <eos>",
           "If a train travels 120 kilometers in 2 hours, what is its average speed? <eos>",
           "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig? <eos>"
           ]

for ssm in ssms:
    ssm.generation_config = ssm_generation_config

    for prompt in prompts:
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        aux_metrics = {}

        # generation_config.return_dict_in_generate = True
        outputs = llm.generate(**inputs, 
                            assistant_model=ssm, 
                            generation_config=llm_generation_config,
                            max_length = 128,
                            # temperature = 0.9,
                            aux_metrics = aux_metrics
                            )

        n_matches = aux_metrics["n_matches"]
        n_correct = aux_metrics["n_correct"]

        print(tokenizer.batch_decode(outputs["sequences"]))
        print(f"acceptance_rate: {np.sum(n_matches) / ssm.generation_config.num_assistant_tokens / len(n_matches)}")
        print(f"correct_rate: {np.sum(n_correct) / ssm.generation_config.num_assistant_tokens / len(n_correct)}")


