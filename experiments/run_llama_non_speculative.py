import sys
sys.path
sys.path.append('..')
from src.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, GenerationConfig
from huggingface_hub import HfApi
from pathlib import Path
import csv

import numpy as np
from huggingface_hub import login


def get_correct_tokens(ground_truth_seq, speculative_seqs, n_matches, starting_idx=0):
    ground_truth_seq = np.squeeze(ground_truth_seq)

    if speculative_seqs[0].shape != speculative_seqs[-1].shape:
       speculative_seqs = speculative_seqs[:-1]
       n_matches = n_matches[:-1]
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

    if speculative_seqs[0].shape != speculative_seqs[-1].shape:
       speculative_seqs = speculative_seqs[:-1]
       n_matches = n_matches[:-1]
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

    login("=== your huggingface access key here ===")

    llm_paths = ["/cb/home/abhay/extra-storage/sparse_inference/sharing/qc/spec_infer_demo/llama2_134M_ift", 
                #  "/cb/home/abhay/extra-storage/sparse_inference/sharing/qc/sparsity_demo/llama2_13B_sparse85_mxfp6_ift"
                ]

    llms = []
    for llm_path in llm_paths:
        llm = AutoModelForCausalLM.from_pretrained(Path(llm_path))
        tokenizer = LlamaTokenizer.from_pretrained(llm_path)
        llms.append((llm, tokenizer, llm_path))

    configs = []
    llm_generation_config = GenerationConfig(return_dict_in_generate = True, eos_token_id = 2)
    configs.append(llm_generation_config)

    prompts = [
        "<|user|> You're a GPT which does not repeat keywords. Can you explain the role of the conductor in orchestral music?<s> <|assistant|>",
        "<|user|> You're a GPT which does not repeat keywords. Can you recommend a good place to learn how to make fresh pasta in Tuscany?<s> <|assistant|>",
        "<|user|> You're a GPT which does not repeat keywords. How does the gravitational pull of the Moon impact the Earth's atmosphere?<s> <|assistant|>"
            #    "<|user|> Can you explain the role of the conductor in orchestral music?<s> <|assistant|>",
            #    "<|user|> What role did ancient Egyptian mythology play in the creation of hieroglyphics?<s> <|assistant|>",
            #    "<|user|> Can you recommend a good place to learn how to make fresh pasta in Tuscany?<s> <|assistant|>",
            #    "<|user|> Can you suggest some local restaurants to try out in Limassol?<s> <|assistant|>",
            #    "<|user|> How does the gravitational pull of the Moon impact the Earth's atmosphere?<s> <|assistant|>",
            #    "<|user|> Write a list of camping essentials for a first-time camper.<s> <|assistant|>",
            #    "<|user|> What are some effective meditation techniques for reducing stress?<s> <|assistant|>",
            #    "<|user|> Can you provide examples of iconic film scores that have stood the test of time?<s> <|assistant|>",
            #    "<|user|> How do ants communicate and organize themselves in their colonies?<s> <|assistant|>",
            #    "<|user|> Are there any good urban parks in the city of Northglenn ideal for cycling and picnicking?<s> <|assistant|>",
            #    "<|user|> How does the climate of the chaparral biome contribute to the spread of wildfire?<s> <|assistant|>",
            #    "<|user|> What industries have experienced the most growth in Twente in the past decade?<s> <|assistant|>",
            #    "<|user|> Can algae thrive in environments with varying salinity levels?<s> <|assistant|>",
            #    "<|user|> Write a poem about the tranquility of a peaceful river.<s> <|assistant|>",
            #    "<|user|> Write a program in Bash that counts the number of files in a directory.<s> <|assistant|>",
            #    "<|user|> Can you explain the purpose of the hypogeum in ancient amphitheatres?<s> <|assistant|>",
            #    "<|user|> How does the chitin exoskeleton of insects protect them from predators?<s> <|assistant|>",
            #    "<|user|> What is unique about the reproductive strategies of animals like sea turtles and kangaroos?<s> <|assistant|>",
            #    "<|user|> What are the key ingredients used in Mexican mole sauce?<s> <|assistant|>",
            #     "<|user|> What is the best way to travel between Sibiu's historical landmarks?<s> <|assistant|>",
            #     "<|user|> What is the relationship between electrical conductivity and temperature in metals?<s> <|assistant|>",
            #     "<|user|> Are there any iconic desserts that are unique to Casablanca?<s> <|assistant|>",
            #     "<|user|> How has the average temperature in New York City fluctuated over the past century?<s> <|assistant|>",
            #     "<|user|> How do mussels feed on plankton and other small organisms?<s> <|assistant|>",
            #     "<|user|> What is the history of Labrador Retrievers as hunting dogs?<s> <|assistant|>",
            #     "<|user|> What is the average GPA required for the biology program at University of Maryland?<s> <|assistant|>",
            #     "<|user|> Create a Facebook post promoting a pop-up shop event.<s> <|assistant|>",
            #     "<|user|> Write a descriptive paragraph about a character's quirks.<s> <|assistant|>",
            #     "<|user|> Develop a recipe for a healthy dessert and include nutritional information.<s> <|assistant|>",
            #     "<|user|> List of ingredients for making a warming chicken tortilla soup.<s> <|assistant|>",
            #     "<|user|> Write a program in Python that finds the k smallest elements of an array.<s> <|assistant|>",
            #     "<|user|> Create a Spotify music playlist to promote positive work-life balance vibes.<s> <|assistant|>",
            #     "<|user|> Create a list of highly effective study tips for students in high school.<s> <|assistant|>",
            #     "<|user|> Write a poem about a special memory with your best friend.<s> <|assistant|>",
            #     "<|user|> What are the most common ingredients featured in Madeiran culinary dishes?<s> <|assistant|>",
            #     "<|user|> Can you suggest any scenic bike trails in the Netherlands?<s> <|assistant|>",
            #     "<|user|> Plan a list of self-care activities for a busy schedule.<s> <|assistant|>",
            #     "<|user|> Can you explain the process of how a car engine ignites fuel to create power?<s> <|assistant|>",
            #     "<|user|> Could you recommend any traditional food dishes to try while in Rome, Italy?<s> <|assistant|>",
            #     "<|user|> Design a shopping bag that promotes environmental conservation.<s> <|assistant|>",
            #     "<|user|> What are the mechanisms used by salmon to migrate upstream to their breeding grounds?<s> <|assistant|>",
            #     "<|user|> What is the symbolism behind the Lotus Temple in Delhi, India?<s> <|assistant|>",
            #     "<|user|> What is the maximum range of Bluetooth technology and how does it differ from Wi-Fi?<s> <|assistant|>",
            #     "<|user|> What are some of the traditional musical instruments used by the people of the Andean mountains in South America?<s> <|assistant|>",
            #     "<|user|> Write a descriptive paragraph about the colors of a butterfly sanctuary.<s> <|assistant|>",
            #     "<|user|> Can you provide some examples of famous Renaissance artists and their most notable works?<s> <|assistant|>",
            #     "<|user|> Write a poem about the sound of a thunderstorm at night.<s> <|assistant|>",
            #     "<|user|> What is the current state of the job market in Miami-Dade County?<s> <|assistant|>",
            #     "<|user|> How long does it take for a penguin egg to hatch, and how do penguin parents care for their young?<s> <|assistant|>",
            #     "<|user|> Can you recommend a unique cocktail recipe featuring gin as the main ingredient?<s> <|assistant|>",
            ]

    filename = 'out_inference.csv'
    with open(filename, 'a') as csvfile:
        # Create a csv writer object
        csvwriter = csv.writer(csvfile)

        for prompt in prompts:
            print(f"**********************************************")
            print(prompt)
            
            row = [prompt]

            for llm, tokenizer, llm_path in llms:    
                print(f"***  Decoding for LLM at path: {llm_path}  ***")
                for llm_generation_config in configs:
                    
                    inputs = tokenizer(prompt, return_tensors="pt")

                    # generation_config.return_dict_in_generate = True
                    outputs = llm.generate(**inputs, 
                                        generation_config=llm_generation_config,
                                        max_length = 128,
                                        )
                    response = tokenizer.batch_decode(outputs["sequences"])
                    print(response)

                    row.append(llm_path)
                    row.append(response)
            csvwriter.writerow(tuple(row))

        
