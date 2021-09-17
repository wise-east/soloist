from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange, tqdm
import json

import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('.')
sys.path.append('./transformers')
sys.path.append('./transformers/')

from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(150)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, token_type_ids, system_token_id, eos_token_id,  num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,  device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)

    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    token_type_ids = token_type_ids.unsqueeze(0).repeat(num_samples, 1)
    system_token_id = torch.tensor(system_token_id, dtype=torch.long, device=device)
    system_token_id = system_token_id.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):

            inputs = {'input_ids': generated, 'token_type_ids':token_type_ids}
            # print(inputs)
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            token_type_ids = torch.cat((token_type_ids, system_token_id), dim=1)

            # import pdb; 
            # pdb.set_trace() 
            if next_token == eos_token_id:
                break  
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='gpt2', type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default='<|endoftext|>', help="Token at which text generation is stopped")
    parser.add_argument('--input_file', type=str, default=None, help="input json file to decoding")
    parser.add_argument('--output_file', type=str, default=None, help="save path")
    parser.add_argument('--max_turn', type=int, default=15, help="number of turns used as context")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--do_batch', action='store_true', help="do batch inference")
    parser.add_argument("--batch_test", action='store_true', help="for testing batch decoding approach")


    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    inputs = json.load(open(args.input_file))
    output_tests = []
    system_token_id = tokenizer.convert_tokens_to_ids(['system'])
    user_token_id = tokenizer.convert_tokens_to_ids(['user'])
    
    count = 0 
    if args.do_batch: 
        for idx in tqdm(range(0, len(inputs), args.batch_size)): 
            count += args.batch_size 

            # form batches
            input_context_tokens = [] 
            input_token_type_ids = [] 
            context_lengths =[] 
            max_length = -1 
            for i in range(idx, min(idx+args.batch_size, len(inputs))): 
                example = inputs[i]
                history = example['history']
                context = history[-args.max_turn:]
                context_ids = [] 
                token_ids_for_context = [] 
                for cxt in context:
                    ids = tokenizer.encode(cxt)
                    context_ids += ids
                    if 'user :' in cxt:
                        token_ids_for_context += user_token_id * len(ids)
                    else:
                        token_ids_for_context += system_token_id * len(ids)
                
                response = '=>'
                response_id = tokenizer.encode(response)

                context_tokens = context_ids + response_id
                token_type_ids = token_ids_for_context  + system_token_id

                context_lengths.append(len(context_tokens))

                # import pdb; pdb.set_trace()
                assert( len(context_tokens) == len(token_type_ids))
                input_context_tokens.append(context_tokens)
                max_length = max(max_length, len(context_tokens))
                input_token_type_ids.append(token_type_ids)

            # add padding for batch processing
            input_context_tokens = [ict + [tokenizer.eos_token_id]*(max_length - len(ict)) for ict in input_context_tokens] 
            input_token_type_ids = [itt + system_token_id*(max_length - len(itt)) for itt in input_token_type_ids]
            input_attention_masks =[[1 if ct!=tokenizer.eos_token_id else 0 for ct in ict] for ict in input_context_tokens] 

            # import pdb; pdb.set_trace()
            input_context_tokens = torch.LongTensor(input_context_tokens).to(args.device)
            input_token_type_ids=torch.LongTensor(input_token_type_ids).to(args.device)
            input_attention_masks=torch.LongTensor(input_attention_masks).to(args.device)

            out_ids = model.generate(
                input_ids = input_context_tokens, 
                temperature=args.temperature, 
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                max_length=args.length,
                num_return_sequences=args.num_samples,
                num_beams=args.num_samples,
                repetition_penalty=args.repetition_penalty, 
                token_type_ids=input_token_type_ids,
                attention_mask = input_attention_masks                 
            )

            out_ids = out_ids.reshape(len(input_context_tokens), args.num_samples, -1)
            out_ids_trimmed = out_ids[:,:,max_length:]
            for j, out in enumerate(out_ids_trimmed):
                examples = [inputs[idx+j]]
                # examples = [] 
                # out = out[:, context_lengths[j]:].tolist()
                for o in out:
                    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                    text = text[: text.find(args.stop_token) if args.stop_token else None]
                    examples.append(text)

                    if text.strip() == "" or "!!!!!" in text: 
                        import pdb; 
                        pdb.set_trace() 
                output_tests.append(examples)

            if args.batch_test and count >= args.batch_size: 
                break 
        json.dump(output_tests, open(args.output_file,'w'), indent=2)

    # keep to show that I get the same results. 
    set_seed(args)
    output_tests = []
    # non parallel generation
    model.eval()
    for idx in tqdm(range(len(inputs))):
        # logger.info(f"PROGRESS: {int(idx/len(inputs)*100)}%")
        example = inputs[idx]
        history = example['history']
        context = history[-args.max_turn:]
        context_ids = []
        token_ids_for_context = []
        for cxt in context:
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cxt))
            context_ids += ids
            if 'user :' in cxt:
                token_ids_for_context += user_token_id * len(ids)
            else:
                token_ids_for_context += system_token_id * len(ids)
        
        response = '=>'
        response_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))

        context_tokens = context_ids + response_id
        token_type_ids = token_ids_for_context  + system_token_id

        assert( len(context_tokens) == len(token_type_ids))
        
        out = sample_sequence(
            model=model,
            context=context_tokens,
            token_type_ids=token_type_ids,
            system_token_id=system_token_id,
            eos_token_id = tokenizer.eos_token_id, 
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )
        out = out[:, len(context_tokens):].tolist()
        example = [example] 
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            # text = text[: text.find(args.stop_token) if args.stop_token else None]
            example.append(text)
        
        output_tests.append(example)
        # print(examples)
        if args.batch_test and len(output_tests) ==count: 
            break

    json.dump(output_tests, open(args.output_file.replace(".json","")+"_nonparallel.json",'w'), indent=2)
    return text


if __name__ == '__main__':
    main()