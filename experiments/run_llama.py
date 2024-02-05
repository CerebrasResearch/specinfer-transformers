import sys
sys.path
sys.path.append('..')
from src.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, GenerationConfig
from huggingface_hub import HfApi
from pathlib import Path
import numpy as np
from huggingface_hub import login

login("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")

# llm_path = "meta-llama/Llama-2-7b-hf"
# llm = AutoModelForCausalLM.from_pretrained(llm_path)
llm_path_1 = "/cb/home/abhay/extra-storage/sparse_inference/pruning/qc_stuff/converted_checkpoints/llama2_7B_ultrachat_ift_1epoch"
llm_path_2 = "/cb/home/abhay/extra-storage/sparse_inference/pruning/qc_stuff/converted_checkpoints/llama2_7B_ultrachat_ift_2epoch"
llm_path_3 = "/cb/home/abhay/extra-storage/sparse_inference/pruning/qc_stuff/converted_checkpoints/llama2_7B_ultrachat_ift_3epoch"

llm_paths = [llm_path_1, llm_path_2, llm_path_3]


ssm1 = AutoModelForCausalLM.from_pretrained(Path("/cb/home/abhay/extra-storage/sparse_inference/pruning/qc_stuff/converted_checkpoints/llama2_450M_ultrachat_ift_1epoch"))
ssm2 = AutoModelForCausalLM.from_pretrained(Path("/cb/home/abhay/extra-storage/sparse_inference/pruning/qc_stuff/converted_checkpoints/llama2_450M_ultrachat_ift_2epoch"))
ssm3 = AutoModelForCausalLM.from_pretrained(Path("/cb/home/abhay/extra-storage/sparse_inference/pruning/qc_stuff/converted_checkpoints/llama2_450M_ultrachat_ift_3epoch"))
ssms = [ssm1, ssm2, ssm3]

configs = []
for num_assistant_tokens in [6, 16]:
    # for use_target_kv_cache in [True, False]:
    ssm_generation_config = GenerationConfig(return_dict_in_generate = True, 
                                    eos_token_id = 2, 
                                    num_assistant_tokens_schedule="constant",
                                    num_assistant_tokens=num_assistant_tokens)

    llm_generation_config = GenerationConfig(return_dict_in_generate = True, 
                                        eos_token_id = 2, 
                                        num_assistant_tokens_schedule="constant",
                                        num_assistant_tokens=num_assistant_tokens,
                                        use_target_kv_cache=False)
    configs.append((llm_generation_config, ssm_generation_config))

prompts = [
           "Implement a Python function to compute the Fibonacci numbers. def",
           "Who does Harry turn into a balloon? <eos>",
           "What were the major contributing factors to the fall of the Roman Empire? <eos>",
           "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts. <eos>",
           "If a train travels 120 kilometers in 2 hours, what is its average speed? <eos>",
           "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig? <eos>"
           ]

def get_correct_tokens(ground_truth_seq, speculative_seqs, n_matches, starting_idx=0):
    ground_truth_seq = np.squeeze(ground_truth_seq)
    speculative_seqs = np.squeeze(speculative_seqs)
    
    idx = starting_idx
    n_correct = 0
    n_total = 0
    for i in range(len(n_matches)):
        
        # print(ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy())
        # print(speculative_seqs[i])
        # print(np.where(ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy() == speculative_seqs[i], 1, 0))
        ground_truth = ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy()
        speculative = speculative_seqs[i]
        if len(ground_truth) == len(speculative):
            correct = np.where(ground_truth == speculative, 1, 0).sum()
        else:
            n = np.min([len(ground_truth), len(speculative)])
            correct = np.where(ground_truth[:n] == speculative[:n], 1, 0).sum()
        
        # print(correct)
        n_correct += correct
        idx += n_matches[i] + 1
        n_total += len(speculative_seqs[i])

    return n_correct / n_total

def get_diverge_index(ground_truth_seq, speculative_seqs, n_matches, starting_idx=0):
    ground_truth_seq = np.squeeze(ground_truth_seq)
    speculative_seqs = np.squeeze(speculative_seqs)
    
    idx = starting_idx
    n_diverge = 0
    n_total = 0
    for i in range(len(n_matches)):
        
        # print(ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy())
        # print(speculative_seqs[i])
        # print(np.where(ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy() == speculative_seqs[i], 1, 0))
        ground_truth = ground_truth_seq[idx:idx+len(speculative_seqs[i])].numpy()
        speculative = speculative_seqs[i]
        if len(ground_truth) == len(speculative):
            mismatch = ground_truth != speculative
        else:
            n = np.min([len(ground_truth), len(speculative)])
            mismatch = ground_truth[:n] != speculative[:n]
        if np.max(mismatch) > 0:
            diverge = np.argmax(mismatch)
        else: 
            diverge = len(mismatch)
        
        n_diverge += diverge
        idx += n_matches[i] + 1
        n_total += 1
        # print(diverge)
        # print(n_total)

    return n_diverge / n_total

if __name__ == '__main__':
    for ssm, llm_path in zip(ssms, llm_paths):

        llm = AutoModelForCausalLM.from_pretrained(Path(llm_path))
        tokenizer = LlamaTokenizer.from_pretrained(llm_path)

        print(f"llm: {llm}")
        print(f"ssm: {ssm}")

        for llm_generation_config, ssm_generation_config in configs:
            print(f"*** num_assistant_tokens: {llm_generation_config.num_assistant_tokens}; share_kv_cache: {llm_generation_config.use_target_kv_cache} ***")
            ssm.generation_config = ssm_generation_config

            for prompt in prompts:
                print(prompt)
                inputs = tokenizer(prompt, return_tensors="pt")
                aux_metrics = {}
                print(inputs)

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

                # print(f"output sequence: {outputs['sequences']}")
                # print(f'candidate tokens: {aux_metrics["candidate_tokens"]}')
                print(tokenizer.batch_decode(outputs["sequences"]))
                print(f"acceptance_rate: {np.sum(n_matches) / ssm.generation_config.num_assistant_tokens / len(n_matches)}")
                print(f"correct_rate (BC): {np.sum(n_correct) / ssm.generation_config.num_assistant_tokens / len(n_correct)}")
                print(f'correct_rate (GT): {get_correct_tokens(outputs["sequences"], aux_metrics["candidate_tokens"], aux_metrics["n_matches"], len(np.squeeze(inputs["input_ids"])))}')
                print(f'GT BC avg divergence point: {get_diverge_index(outputs["sequences"], aux_metrics["selected_tokens"], aux_metrics["n_matches"], len(np.squeeze(inputs["input_ids"])))}')

