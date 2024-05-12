# python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
# python test_in_batch.py

import time
import random
import string
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import sglang as sgl
from sglang import set_default_backend, Runtime
from sglang.backend.runtime_endpoint import RuntimeEndpoint


def generate_random_string(token_length: int) -> str:
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10000))
    tokenized_output = tokenizer.encode(random_string, add_special_tokens=False)[:token_length]

    if len(tokenized_output) < token_length:
        tokenized_output = tokenized_output + [tokenizer.pad_token_id] * (token_length - len(tokenized_output))

    decoded_string = tokenizer.decode(tokenized_output, skip_special_tokens=False)
    return decoded_string


def generate_unique_prefix(base_text, index):
    return str(index) + base_text[len(str(index)):]


@sgl.function
def text_qa(s, question):
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0)


def prepare_prompts(num_prefix, num_samples_per_prefix, prefix_length, suffix_length):
    base_prefix = generate_random_string(prefix_length)
    
    tot_time = 0
    tot_input_len = 0
    all_prompts = []
    for i in tqdm(range(num_prefix)):
        unique_prefix = generate_unique_prefix(base_prefix, i)
        prompt_list = []
        for j in range(num_samples_per_prefix):
            suffix = generate_random_string(suffix_length)
            prompt = unique_prefix + suffix
            prompt_list.append({"question": prompt})
            tot_input_len += len(tokenizer.encode(prompt))
        all_prompts.append(prompt_list)
    return all_prompts, tot_input_len
 
    
def test_prefix_caching(num_prefix, num_samples_per_prefix, prefix_length, suffix_length):
    all_prompts, tot_input_len = prepare_prompts(num_prefix, num_samples_per_prefix, prefix_length, suffix_length)

    tot_time = 0
    tot_output_len = 0
    for i in tqdm(range(num_prefix)):
        tic = time.time()
        output = text_qa.run_batch(
            all_prompts[i],
            progress_bar=True
        )
        tot_time += time.time() - tic

        for i, out in enumerate(output):
            answer = out["answer"]
            output_token_length = len(tokenizer.encode(answer))
            tot_output_len += output_token_length
    
    print(f"Total input token length: {tot_input_len}, Total output token length: {tot_output_len}")
    rps = num_prefix * num_samples_per_prefix / tot_time
    tps = (tot_input_len + tot_output_len) / tot_time
    print(f"Throughput: {rps:.2f} requests/s, {tps:.2f} tokens/s")


def test_prefix_caching_hint(num_prefix, num_samples_per_prefix, prefix_length, suffix_length):
    all_prompts, tot_input_len = prepare_prompts(num_prefix, num_samples_per_prefix, prefix_length, suffix_length)

    tot_time = 0
    tot_output_len = 0
    for i in tqdm(range(num_prefix)):
        tic = time.time()
        text_qa.run_batch(all_prompts[i][:1])
        output = text_qa.run_batch(
            all_prompts[i],
            progress_bar=True
        )
        tot_time += time.time() - tic

        for i, out in enumerate(output):
            answer = out["answer"]
            output_token_length = len(tokenizer.encode(answer))
            tot_output_len += output_token_length
    
    print(f"Total input token length: {tot_input_len}, Total output token length: {tot_output_len}")
    rps = num_prefix * num_samples_per_prefix / tot_time
    tps = (tot_input_len + tot_output_len) / tot_time
    print(f"Throughput: {rps:.2f} requests/s, {tps:.2f} tokens/s")


def test_prefix_caching_send_all(num_prefix, num_samples_per_prefix, prefix_length, suffix_length):
    all_prompts, tot_input_len = prepare_prompts(num_prefix, num_samples_per_prefix, prefix_length, suffix_length)
    all_prompts = [x for prompt_list in all_prompts for x in prompt_list]

    # run
    tic = time.time()
    output = text_qa.run_batch(
        all_prompts,
        progress_bar=True
    )
    tot_time = time.time() - tic

    # print
    tot_output_len = 0
    for i, out in enumerate(output):
        answer = out["answer"]
        output_token_length = len(tokenizer.encode(answer))
        tot_output_len += output_token_length
    
    print(f"Total input token length: {tot_input_len}, Total output token length: {tot_output_len}")
    rps = num_prefix * num_samples_per_prefix / tot_time
    tps = (tot_input_len + tot_output_len) / tot_time
    print(f"Throughput: {rps:.2f} requests/s, {tps:.2f} tokens/s")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    backend = RuntimeEndpoint("http://127.0.0.1:30000")
    set_default_backend(backend)
    
    random.seed(0)
    num_prefix = 2
    num_samples_per_prefix = 32
    prefix_length = 500 
    suffix_length = 100
    test_prefix_caching(num_prefix, num_samples_per_prefix, prefix_length, suffix_length)
    test_prefix_caching_hint(num_prefix, num_samples_per_prefix, prefix_length, suffix_length)
    # test_prefix_caching_send_all(num_prefix, num_samples_per_prefix, prefix_length, suffix_length)
