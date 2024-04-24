import sys
sys.path
sys.path.append('..')
from src.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, GenerationConfig
from huggingface_hub import HfApi
from pathlib import Path
import numpy as np
from huggingface_hub import login
import csv


login("your API key here")


llm_paths = ["/net/extradisks1/srv/nfs/abhay-extra-data/sparse_inference/sharing/qc/spec_infer_demo/llama2_7B_ift", 
             "/net/extradisks1/srv/nfs/abhay-extra-data/sparse_inference/sharing/qc/sparsity_demo/llama2_13B_dense_ift", 
             "/net/extradisks1/srv/nfs/abhay-extra-data/sparse_inference/sharing/qc/sparsity_demo/llama2_13B_sparse_ift_v2",
             "/net/extradisks1/srv/nfs/abhay-extra-data/sparse_inference/sharing/qc/sparsity_demo/llama2_13B_sparse85_ift"
             ]

ssm_paths = ["/cb/home/abhay/extra-storage/sparse_inference/sharing/qc/spec_infer_demo/llama2_115M_ift",
             "/cb/home/abhay/extra-storage/sparse_inference/sharing/qc/spec_infer_demo/llama2_134M_ift"]


ssms = [AutoModelForCausalLM.from_pretrained(Path(ssm_path)) for ssm_path in ssm_paths]

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
        #    "<|user|> Can you explain the role of the conductor in orchestral music?<s> <|assistant|>",
        #    "<|user|> What role did ancient Egyptian mythology play in the creation of hieroglyphics?<s> <|assistant|>",
        #    "<|user|> Can you recommend a good place to learn how to make fresh pasta in Tuscany?<s> <|assistant|>",
        #    "<|user|> Can you suggest some local restaurants to try out in Limassol?<s> <|assistant|>",
        #    "<|user|> How does the gravitational pull of the Moon impact the Earth's atmosphere?<s> <|assistant|>",
        #    "<|user|> Write a list of camping essentials for a first-time camper.<s> <|assistant|>",
        #    "<|user|> What are some effective meditation techniques for reducing stress?<s> <|assistant|>",
        #    "<|user|> Can you provide examples of iconic film scores that have stood the test of time?<s> <|assistant|>",
        #    "<|user|> How do ants communicate and organize themselves in their colonies?<s> <|assistant|>",
        #    "<|user|> Are there any good urban parks in the city of Northglenn ideal for cycling and picnicking?<s> <|assistant|>"
        
        # Chosen good prompts
           "<|user|> Can you provide examples of iconic film scores that have stood the test of time?<s> <|assistant|>",
           "<|user|> What industries have experienced the most growth in Twente in the past decade?<s> <|assistant|>",
           "<|user|> Write a poem about the tranquility of a peaceful river.<s> <|assistant|>",
           "<|user|> What is the average GPA required for the biology program at University of Maryland?<s> <|assistant|>",
        #    "<|user|> Create a Facebook post promoting a pop-up shop event.<s> <|assistant|>",
        #    "<|user|> Develop a recipe for a healthy dessert and include nutritional information.<s> <|assistant|>",
        #    "<|user|> Create a list of highly effective study tips for students in high school.<s> <|assistant|>",
        #    "<|user|> Write a poem about a special memory with your best friend.<s> <|assistant|>",
        #    "<|user|> Plan a list of self-care activities for a busy schedule.<s> <|assistant|>",
        #    "<|user|> Write a descriptive paragraph about the colors of a butterfly sanctuary.<s> <|assistant|>",
        #    "<|user|> Write a poem about the sound of a thunderstorm at night.<s> <|assistant|>",
        #    "<|user|> Can you recommend a unique cocktail recipe featuring gin as the main ingredient?<s> <|assistant|>",
           ]

def get_correct_tokens(ground_truth_seq, speculative_seqs, n_matches, starting_idx=0):
    
    ground_truth_seq = np.squeeze(ground_truth_seq)

    for i in range(len(speculative_seqs)):
        if speculative_seqs[0].shape != speculative_seqs[i].shape:
            speculative_seqs = speculative_seqs[:i]
            n_matches = n_matches[:i]
            break
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

    for i in range(len(speculative_seqs)):
        if speculative_seqs[0].shape != speculative_seqs[i].shape:
            speculative_seqs = speculative_seqs[:i]
            n_matches = n_matches[:i]
            break
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

    filename = 'out_speculative.csv'
    with open(filename, 'a') as csvfile:
        # Create a csv writer object
        csvwriter = csv.writer(csvfile)
        for llm_path in llm_paths:
            llm = AutoModelForCausalLM.from_pretrained(Path(llm_path))
            tokenizer = LlamaTokenizer.from_pretrained(llm_path)
            for ssm, ssm_path in zip(ssms, ssm_paths):
                
                print(f"llm: {llm_path}")
                print(f"ssm: {ssm_path}")

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

                        acceptance_rate = np.sum(n_matches) / ssm.generation_config.num_assistant_tokens / len(n_matches)
                        correct_rate_bc = np.sum(n_correct) / ssm.generation_config.num_assistant_tokens / len(n_correct)
                        correct_rate_gt = get_correct_tokens(outputs["sequences"], aux_metrics["candidate_tokens"], aux_metrics["n_matches"], len(np.squeeze(inputs["input_ids"])))
                        divergence_index = get_diverge_index(outputs["sequences"], aux_metrics["selected_tokens"], aux_metrics["n_matches"], len(np.squeeze(inputs["input_ids"])))

                        print(tokenizer.batch_decode(outputs["sequences"]))
                        print(f"acceptance_rate: {acceptance_rate}")
                        print(f"correct_rate (BC): {correct_rate_bc}")
                        print(f'correct_rate (GT): {correct_rate_gt}')
                        print(f'GT BC avg divergence point: {divergence_index}')

                        row = [llm_path, ssm_path, llm_generation_config.num_assistant_tokens, prompt, acceptance_rate, correct_rate_gt, correct_rate_bc, divergence_index]
                        csvwriter.writerow(tuple(row))

