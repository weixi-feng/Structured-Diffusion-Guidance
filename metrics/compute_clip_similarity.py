import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")
from rich.progress import track

from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates

import os
import sng_parser
import stanza
from nltk.tree import Tree
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
# import argparse, os
# parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", type=str, required=True, help='the dir of your generated images.')
# parser.add_argument("--eval_partial", action='store_true')
# opt = parser.parse_args()
# print(opt)

def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError



def get_token_alignment_map(tree, tokens):
    if tokens is None:
        return {i:[i] for i in range(len(tree.leaves())+1)}
        
    def get_token(token):
        return token[:-4] if token.endswith("</w>") else token

    idx_map = {}
    j = 0
    max_offset = np.abs(len(tokens) - len(tree.leaves()))
    mytree_prev_leaf = ""
    for i, w in enumerate(tree.leaves()):
        token = get_token(tokens[j])
        idx_map[i] = [j]
        if token == mytree_prev_leaf+w:
            mytree_prev_leaf = ""
            j += 1
        else:
            if len(token) < len(w):
                prev = ""
                while prev + token != w:
                    prev += token
                    j += 1
                    token = get_token(tokens[j])
                    idx_map[i].append(j)
                    # assert j - i <= max_offset
            else:
                mytree_prev_leaf += w
                j -= 1
            j += 1
    idx_map[i+1] = [j]
    return idx_map


def get_all_nps(tree, full_sent, tokens=None, highest_only=False, lowest_only=False):
    start = 0
    end = len(tree.leaves())

    idx_map = get_token_alignment_map(tree, tokens)

    def get_sub_nps(tree, left, right):
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []
        sub_nps = []
        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
        assert right - left == n_leaves
        if tree.label() == 'NP' and n_leaves > 1:
            sub_nps.append([" ".join(tree.leaves()), (int(min(idx_map[left])), int(min(idx_map[right])))])
            if highest_only and sub_nps[-1][0] != full_sent: return sub_nps
        for i, subtree in enumerate(tree):
            sub_nps += get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
        return sub_nps
    
    all_nps = get_sub_nps(tree, left=start, right=end)
    lowest_nps = []
    for i in range(len(all_nps)):
        span = all_nps[i][1]
        lowest = True
        for j in range(len(all_nps)):
            if i == j: continue
            span2 = all_nps[j][1]
            if span2[0] >= span[0] and span2[1] <= span[1]:
                lowest = False
                break
        if lowest:
            lowest_nps.append(all_nps[i])

    if lowest_only:
        all_nps = lowest_nps

    if len(all_nps) == 0:
        all_nps = []
        spans = []
    else:
        all_nps, spans = map(list, zip(*all_nps))
    if full_sent not in all_nps:
        all_nps = [full_sent] + all_nps
        spans = [(min(idx_map[start]), min(idx_map[end]))] + spans

    return all_nps, spans, lowest_nps


@dataclass
class EvalConfig:
    # exp_name: str = "default"
    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")
    input_dir: Path = Path("./input_dir") 
    eval_partial: bool = False
    truncate: bool = False

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)
    


@pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    exp_name = str(config.input_dir).split('/')[-1]
    print(f"running on exp {exp_name}")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    print("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    files = os.listdir(config.input_dir)
    print(f"Running on {len(files)} prompts...")

    results_per_prompt = {}

    gt_prompts = list()
    with open('/home/zhlu6105/Projects/decompose/Structured-Diffusion-Guidance/ABC-6K.txt', 'r') as f:
       gt_prompts = f.read().splitlines()
       
    for img in track(files):

        prompt = img.replace('.jpg', '').split('-')[-1]
        prompt_idx = int(img.split('-')[0])
        gt_prompt = gt_prompts[prompt_idx]
        # print(prompt, ' | ', gt_prompt, " | ", prompt_idx, " | ", img)
        if config.truncate:
            prompt = prompt.split('|')[0]

        # get all images for the given prompt
        # image_paths = [p for p in (config.output_path / prompt).rglob('*') if p.suffix in ['.png', '.jpg']]
        image_paths = [os.path.join(config.input_dir, img)]
        images = [Image.open(p) for p in image_paths]
        # image_names = [p.name for p in image_paths]
        image_names = [img]
        queries = [preprocess(image).unsqueeze(0).to(device) for image in images]
        prompt = gt_prompt

        with torch.no_grad():

            # split prompt into first and second halves
            # if config.eval_partial:
            #     if ' and ' in prompt:
            #         prompt_parts = prompt.split(' and ')
            #     elif ' with ' in prompt:
            #         prompt_parts = prompt.split(' with ')
            #     else:
            #         print(f"Unable to split prompt: {prompt}. "
            #               f"Looking for 'and' or 'with' for splitting! Skipping!")
            #         continue

            prompt = prompt.strip()
            if config.eval_partial:
                try:
                    doc = nlp(prompt)
                    mytree = Tree.fromstring(str(doc.sentences[0].constituency))
                    # tokens = model.cond_stage_model.tokenizer.tokenize(prompts[0])
                    tokens = None
                    prompt_parts, spans, noun_chunk = get_all_nps(mytree, prompt, tokens)
                    prompt_parts = prompt_parts[1:] # remove the first sentence as it is the full sentence
                except:
                    print(f"failed to parse: {prompt}")
                    continue
            # extract texture features
            full_text_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)

            # extract image features
            images_features = [model.encode_image(image) for image in queries]
            images_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in images_features]

            # compute similarities
            full_text_similarities = [(feat.float() @ full_text_features.T).item() for feat in images_features]
            partial_sim = list() 
            if config.eval_partial:
                for partial_prompt in prompt_parts:
                    partial_feature = get_embedding_for_prompt(model, partial_prompt, templates=imagenet_templates)
                    partial_similarities = [(feat.float() @ partial_feature.T).item() for feat in images_features]
                    partial_sim.append(partial_similarities)

            if config.eval_partial and len(partial_sim) != 0:
                results_per_prompt[prompt] = {
                    'full_text': full_text_similarities,
                    # 'first_half': first_half_similarities,
                    # 'second_half': second_half_similarities,
                    'partial_text': partial_sim,
                    'image_names': image_names,
                }
            elif not config.eval_partial:
                results_per_prompt[prompt] = {
                    'full_text': full_text_similarities,
                    # 'first_half': first_half_similarities,
                    # 'second_half': second_half_similarities,
                    'partial_text': partial_sim,
                    'image_names': image_names,
                }

    # aggregate results
    aggregated_results = {
        'full_text_aggregation': aggregate_by_full_text(results_per_prompt),
        'min_partial_aggregation': aggregate_by_min_partial(results_per_prompt) if config.eval_partial else 0.,
        'partial_aggregation': aggregate_by_partial(results_per_prompt) if config.eval_partial else 0.,
    }


    with open(config.metrics_save_path / f"clip_raw_metrics_{exp_name}.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    with open(config.metrics_save_path / f"clip_aggregated_metrics_{exp_name}.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)

def aggregate_by_min_half(d):
    """ Aggregate results for the minimum similarity score for each prompt. """
    min_per_half_res = [[min(a, b) for a, b in zip(d[prompt]["first_half"], d[prompt]["second_half"])] for prompt in d]
    min_per_half_res = np.array(min_per_half_res).flatten()
    return np.average(min_per_half_res)

def aggregate_by_min_partial(d):
    min_per_partial_res = [min(d[prompt]['partial_text']) for prompt in d]
    min_per_half_res = np.array(min_per_partial_res).flatten()
    return np.average(min_per_half_res)

def aggregate_by_partial(d):
    temp = list() 
    min_per_partial_res = [d[prompt]['partial_text'] for prompt in d]
    for prompt in d:
        for x in d[prompt]['partial_text']:
            temp.append(x)
    min_per_half_res = np.array(temp).flatten()
    return np.average(min_per_half_res)

def aggregate_by_full_text(d):
    """ Aggregate results for the full text similarity for each prompt. """
    full_text_res = [v['full_text'] for v in d.values()]
    full_text_res = np.array(full_text_res).flatten()
    return np.average(full_text_res)


if __name__ == '__main__':
    run()
