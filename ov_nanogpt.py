import logging as log
import time
from pathlib import Path
import numpy as np
from openvino.runtime import Core, get_version, PartialShape, Dimension, Type
from openvino.preprocess import PrePostProcessor
from openvino.inference_engine import IECore
from transformers import GPT2Tokenizer
from numba import njit
import tiktoken


class OVNanoGPTConfig:
    model: str = str(Path('gpt2.xml'))
    device: str = 'CPU'          # 'CPU', 'GPU', 'Auto', 'MULTI:GPU,CPU', 'HETERO:GPU,CPU', etc..
                                 # If you use GPU, better *not* use dynamic shape
    dynamic_shape: bool = True
    top_k: int = 40              # Number of tokens with the highest probability which will be kept for generation
    top_p: float = 0.5           # Maximum probability, tokens with such a probability and lower will be kept for generation
    max_seq_len: int = 500       # Maximum sequence length for processing.
    temperature: float = 0.3
    block_size: int = 2048       # If the sequence context is growing too long we must crop it at block_size


class OVNanoGPT:
    def __init__(self, config: OVNanoGPTConfig):
        self.config = config

        self.enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: np.array(self.enc.encode(s, allowed_special={"<|endoftext|>"}))
        self.decode = lambda l: self.enc.decode(l)
        self.eos_token_id = self.enc.n_vocab - 1
        log.debug('Tokenizer configured')

        log.info('OpenVINO Runtime build: {}'.format(get_version()))
        self.core = Core()

        Path('cl_cache').mkdir(parents=True, exist_ok=True)
        Path('cache').mkdir(parents=True, exist_ok=True)
        self.core.set_property({'CACHE_DIR': Path('cache')})

        if config.device == 'CPU':
            ie = IECore()
            cpu_caps = ie.get_metric(metric_name="OPTIMIZATION_CAPABILITIES", device_name="CPU")
            log.info('Available CPU Optimizations: {}'.format(cpu_caps))
            if 'BF16' in cpu_caps:
                self.core.set_property({'ENFORCE_BF16': 'YES'})

        # read model
        log.info('Reading model {}'.format(config.model))
        self.model = self.core.read_model(config.model)

        self.input_tensor = self.model.inputs[0].any_name

        # validate model
        self._validate_model()

        if not config.dynamic_shape and (
                self.model.inputs[0].partial_shape.is_dynamic or self.model.inputs[0].shape[1] != config.max_seq_len):
            self.model.reshape({self.input_tensor: PartialShape([Dimension(1), Dimension(config.max_seq_len)])})

        if config.dynamic_shape:
            # assign dynamic shapes to every input layer
            for input_layer in self.model.inputs:
                input_shape = input_layer.partial_shape
                input_shape[0] = -1
                input_shape[1] = -1
                self.model.reshape({input_layer: input_shape})

        # TODO: See if we can
        # ppp = PrePostProcessor(self.model)
        # input = ppp.input(tensor_name)
        # input.tensor().set_element_type(Type.u8)
        # input.model().set_element_type(Type.u8)
        # # layout and precision conversion is inserted automatically,
        # # because tensor format != model input format
        # self.model = ppp.build()

        # load model to the device
        self.compiled_model = self.core.compile_model(self.model, config.device)
        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()
        log.info('Model {} is loaded to {}'.format(config.model, config.device))

    def _validate_model(self) -> None:
        # check number inputs and outputs
        if len(self.model.inputs) != 1:
            raise RuntimeError('Expected model with single input, while provided {}'.format(
                len(self.model.inputs)))
        if len(self.model.outputs) != 1:
            raise RuntimeError('Expected model with single output, while provided {}'.format(
                len(self.model.outputs)))

    def _get_top_k_logits(self, scores):
        filter_value = -float("Inf")
        top_k = min(max(self.config.top_k, 1), scores.shape[-1])
        top_k_scores = -np.sort(-scores)[:, :top_k]
        indices_to_remove = scores < np.min(top_k_scores)
        filtred_scores = np.ma.array(scores, mask=indices_to_remove, fill_value=filter_value).filled()
        return filtred_scores

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

    def _stop_criteria(self, input_ids, max_length: int, eos_token_id: int) -> bool:
        if input_ids[0][-1] == eos_token_id:
            return True
        return input_ids.shape[-1] >= max_length

    def _generate(self, input_ids):
        output_key = self.compiled_model.output(0)

        # maximum number of tokens that can be processed by network at once
        input_ids = np.array([input_ids])
        while True:
            cur_input_len = len(input_ids[0])
            model_input_ids = input_ids

            if cur_input_len > self.config.block_size:
                model_input_ids = input_ids[:, -self.config.block_size:]

            if not self.config.dynamic_shape:
                # pad the rest of the request
                pad_len = self.config.max_seq_len - cur_input_len
                model_input_ids = np.concatenate(([[self.eos_token_id] * pad_len], input_ids), axis=-1)

            outputs = self.infer_request.infer({"input": model_input_ids})[output_key]
            next_token_logits = outputs[:, -1, :]

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
            if cur_input_len == self.config.max_seq_len or next_tokens[0] == self.eos_token_id:
                break
            else:
                input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
        return input_ids

    def infer(self, prompt: str) -> str:
        input_ids = self.encode(prompt)

        t0 = time.perf_counter()
        output_ids = self._generate(input_ids)
        t1 = time.perf_counter()
        output_text = ""
        # Convert IDs to words and make the sentence from it
        output_text += self.decode(output_ids[0].tolist())

        log.debug(f'OUTPUT: {output_text}')
        log.info(f'Generation took {t1 - t0:.3f} s')
        return f'{output_text}'
