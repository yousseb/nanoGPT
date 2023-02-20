"""
Convert nanoGPT PyTorch model with pre-trained gpt2 to OpenVino IR
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
from transformers.onnx import export, FeaturesManager
from transformers import GPT2LMHeadModel, GPT2Tokenizer

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

# -----------------------------------------------------------------------------
init_from = 'gpt2'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 1  # number of tokens generated in each sample
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
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    model.eval()
    print(model)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,                                          # model input (or a tuple for multiple inputs)
                      onnx_model_path,                            # destination
                      export_params=True,                         # store the trained parameter weights inside the model
                      opset_version=10,                           # the ONNX version to export the model to
                      do_constant_folding=True,                   # whether to execute constant folding for optimization
                      input_names=['input_ids'],                  # the model's input names
                      output_names=['output'],                             # the model's output names
                      dynamic_axes={'input_ids':  {0: 'batch_size'},       # variable length axes
                                    'output': {0: 'batch_size'}})


def onnx_to_ir():
    import onnx

    global init_from
    global onnx_model_path

    #onnx_model = onnx.load(onnx_model_path)
    #onnx.checker.check_model(onnx_model)

    # # convert model to openvino
    ov_model = mo.convert_model(onnx_model_path, compress_to_fp16=True, input='input_ids[1,1..128]')

    # serialize openvino model
    serialize(ov_model, str(Path(init_from).with_suffix('.xml')))


if __name__ == '__main__':
    pytorch_to_onnx()
    onnx_to_ir()

