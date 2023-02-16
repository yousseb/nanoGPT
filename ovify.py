"""
Sample from a trained model
"""
from contextlib import nullcontext
import torch
import tiktoken
from model import GPT
from pathlib import Path
from openvino.runtime import serialize
from openvino.tools import mo
import logging as log
import sys

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

# -----------------------------------------------------------------------------
init_from = 'gpt2'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

onnx_model_path = (str(Path(init_from).with_suffix('.onnx')))


def pytorch_to_onnx():
    global init_from
    global out_dir
    global start
    global num_samples
    global max_new_tokens
    global temperature
    global top_k
    global seed
    global device
    global dtype
    global compile

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    model.eval()
    model.to(device)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation once
    with torch.no_grad():
        with ctx:
            pass
            # y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # print(decode(y[0].tolist()))
            # print('---------------')

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,                                          # model input (or a tuple for multiple inputs)
                      onnx_model_path,                            # destination
                      export_params=True,                         # store the trained parameter weights inside the model
                      opset_version=10,                           # the ONNX version to export the model to
                      do_constant_folding=True,                   # whether to execute constant folding for optimization
                      input_names=['input'],                      # the model's input names
                      output_names=['output'],                    # the model's output names
                      dynamic_axes={'input':  {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def onnx_to_ir():
    import onnx

    global init_from
    global onnx_model_path

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # convert model to openvino
    ov_model = mo.convert_model(onnx_model_path, compress_to_fp16=True)

    # serialize openvino model
    serialize(ov_model, str(Path(init_from).with_suffix('.xml')))


#---------------------- Try OV
import logging as log
import time
from pathlib import Path
import numpy as np
from openvino.runtime import Core, get_version, PartialShape, Dimension
from transformers import GPT2Tokenizer
from numba import njit


class OVGPTConfig:
    model = str(Path(init_from).with_suffix('.xml'))
    device = 'CPU'               # 'CPU', 'GPU', 'Auto', etc.. If you use GPU, better not use dynamic shape
    top_k = 40                   # Number of tokens with the highest probability which will be kept for generation
    top_p = 0.9                  # Maximum probability, tokens with such a probability and lower will be kept for generation
    max_seq_len = 1024           # Maximum sequence length for processing.
    max_sequence_length = 128
    temperature = 0.8


class OVGPT:
    def __init__(self, config: OVGPTConfig):
        self.config = config

        # # create tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.eos_token_id = self.tokenizer.eos_token_id
        log.debug('Tokenizer configured')

        log.info('OpenVINO Runtime build: {}'.format(get_version()))
        self.core = Core()

        Path('cl_cache').mkdir(parents=True, exist_ok=True)
        Path('cache').mkdir(parents=True, exist_ok=True)
        self.core.set_property({'CACHE_DIR': Path('cache')})

        # read model
        log.info('Reading model {}'.format(config.model))
        self.model = self.core.read_model(config.model)

        self.input_tensor = self.model.inputs[0].any_name

        # validate model
        self._validate_model()

        # assign dynamic shapes to every input layer
        for input_layer in self.model.inputs:
            input_shape = input_layer.partial_shape
            input_shape[0] = -1
            input_shape[1] = -1
            self.model.reshape({input_layer: input_shape})

        # load model to the device
        self.compiled_model = self.core.compile_model(self.model, config.device)
        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()
        log.info('Model {} is loaded to {}'.format(config.model, config.device))

    def _validate_model(self):
        # check number inputs and outputs
        if len(self.model.inputs) != 1:
            raise RuntimeError('Expected model with single input, while provided {}'.format(
                len(self.model.inputs)))
        if len(self.model.outputs) != 1:
            raise RuntimeError('Expected model with single output, while provided {}'.format(
                len(self.model.outputs)))

    def _tokenize(self, text):
        """
        tokenize input text using GPT2 tokenizer

        Parameters:
          text, str - input text
        Returns:
          input_ids - np.array with input token ids
          attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model
        """
        inputs = self.tokenizer(text, return_tensors="np")
        return inputs["input_ids"], inputs["attention_mask"]

    def _get_top_k_logits(self, scores):
        filter_value = -float("Inf")
        top_k = min(max(self.config.top_k, 1), scores.shape[-1])
        top_k_scores = -np.sort(-scores)[:, :top_k]
        indices_to_remove = scores < np.min(top_k_scores)
        filtred_scores = np.ma.array(scores, mask=indices_to_remove, fill_value=filter_value).filled()
        return filtred_scores

    # def _softmax(self, x):
    #     e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    #     sum = e_x.sum(axis=-1, keepdims=True)
    #     return e_x / sum

    # https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-activation-function-and-its-derivative-jacobian-in-python/
    @staticmethod
    @njit(cache=True, fastmath=True)  # Best implementation (VERY FAST)
    def _softmax(x):
        '''
        Performs the softmax activation on a given set of inputs
        Input: x (N,k) ndarray (N: no. of samples, k: no. of nodes)
        Returns:
        Note: Works for 2D arrays only(rows for samples, columns for nodes/outputs)
        '''
        max_x = np.zeros((x.shape[0], 1), dtype=x.dtype)
        for i in range(x.shape[0]):
            max_x[i, 0] = np.max(x[i, :])
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum(axis=1).reshape((-1, 1))  # Alternative of keepdims=True for Numba compatibility

    def _get_top_p_logits(self, scores):
        top_p = self.config.top_p
        filter_value = -float("Inf")
        sorted_indices = np.argsort(-scores)
        sorted_logits = -np.sort(-scores)
        cumulative_probs = np.cumsum(self._softmax(sorted_logits), axis=-1)
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
        sorted_indices_to_remove[:, 0] = 0
        np.put_along_axis(sorted_indices_to_remove, sorted_indices, sorted_indices_to_remove, axis=1)
        filtred_scores = np.ma.array(scores, mask=sorted_indices_to_remove, fill_value=filter_value).filled()
        return filtred_scores

    def _process_logits(self, cur_length, scores, min_length=0):
        """
        reduce probability for padded indicies

        Parameters:
          cur_length - current length of input sequence
          scores - model output logits
          eos_token_id - index of end of string token in model vocab
          min_length - minimum length for appling postprocessing
        """
        if cur_length < min_length:
            scores[:, self.eos_token_id] = -float("inf")
        return scores

    def _stop_criteria(self, input_ids, max_length, eos_token_id):
        if input_ids[0][-1] == eos_token_id:
            return True
        return input_ids.shape[-1] >= max_length

    def _generate(self, input_ids):
        output_key = self.compiled_model.output(0)

        while True:
            cur_input_len = len(input_ids[0])
            model_input_ids = input_ids

            outputs = self.compiled_model({"input": model_input_ids})[output_key]
            next_token_logits = outputs[:, 0, :]

            # pre-process distribution
            next_token_logits = next_token_logits / self.config.temperature

            next_token_scores = self._process_logits(cur_input_len, next_token_logits)

            if self.config.top_k > 0:
                next_token_scores = self._get_top_k_logits(next_token_scores)

            if self.config.top_p < 1.0:
                next_token_scores = self._get_top_p_logits(next_token_scores)

            # get next token id
            probs = self._softmax(next_token_scores)
            next_tokens = np.random.choice(probs.shape[-1], 1, p=probs[0], replace=True)
            # break the loop if max length or end of text token is reached
            if cur_input_len == self.config.max_sequence_length or next_tokens == self.eos_token_id:
                break
            else:
                input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
        return input_ids

    def infer(self, prompt: str) -> str:
        input_ids, attention_mask = self._tokenize(prompt)

        t0 = time.perf_counter()
        output_ids = self._generate(input_ids)
        t1 = time.perf_counter()
        output_text = ""
        # Convert IDs to words and make the sentence from it
        for i in output_ids[0]:
            output_text += self.tokenizer.convert_tokens_to_string(self.tokenizer._convert_id_to_token(i))

        log.debug(f'OUTPUT: {output_text}')
        log.info(f'Generation took {t1 - t0:.3f} s')
        return f'{output_text}'


#pytorch_to_onnx()
#onnx_to_ir()

config = OVGPTConfig()
gpt = OVGPT(config)

for i in range(9):
    ov_result = gpt.infer("Deep learning is a type of machine learning that uses neural networks")

    print(ov_result)
    print('-' * 70)
