import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from rich.progress import track
sys.path.append(".")
sys.path.append("..")
import os
from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates


@dataclass
class EvalConfig:
    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")
    input_dir: Path = Path("./input_dir") 
    eval_partial: bool = False
    truncate: bool = True

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    exp_name = str(config.input_dir).split('/')[-1]
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    print("Done.")

    print("Loading BLIP model...")
    blip_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco",
                                                              is_eval=True, device=device)
    print("Done.")

    # prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    files = os.listdir(config.input_dir)
    print(f"Running on {len(files)} prompts...")

    gt_prompts = list()
    with open('/home/zhlu6105/Projects/decompose/Structured-Diffusion-Guidance/ABC-6K.txt', 'r') as f:
       gt_prompts = f.read().splitlines()

    results_per_prompt = {}
    for img in track(files):
        prompt = img.replace('.jpg', '').split('-')[-1]
        prompt_idx = int(img.split('-')[0])
        gt_prompt = gt_prompts[prompt_idx]
        if config.truncate:
            prompt = prompt.split('|')[0]

        # print(f'Running on: "{prompt}"')

        # get all images for the given prompt
        image_paths = [os.path.join(config.input_dir, img)]
        images = [Image.open(p) for p in image_paths]
        # image_names = [p.name for p in image_paths]
        image_names = [img]
        # print(prompt, '|',  gt_prompt)
        # prompt = gt_prompt
        with torch.no_grad():
            # extract prompt embeddings
            prompt_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)

            # extract blip captions and embeddings
            blip_input_images = [vis_processors["eval"](image).unsqueeze(0).to(device) for image in images]
            blip_captions = [blip_model.generate({"image": image})[0] for image in blip_input_images]
            texts = [clip.tokenize([text]).cuda() for text in blip_captions]
            caption_embeddings = [model.encode_text(t) for t in texts]
            caption_embeddings = [embedding / embedding.norm(dim=-1, keepdim=True) for embedding in caption_embeddings]

            text_similarities = [(caption_embedding.float() @ prompt_features.T).item()
                                 for caption_embedding in caption_embeddings]

            results_per_prompt[prompt] = {
                'text_similarities': text_similarities,
                'captions': blip_captions,
                'image_names': image_names,
            }

    # aggregate results
    total_average, total_std = aggregate_text_similarities(results_per_prompt)
    aggregated_results = {
        'average_similarity': total_average,
        'std_similarity': total_std,
    }

    with open(config.metrics_save_path / f"blip_raw_metrics_{exp_name}.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    with open(config.metrics_save_path / f"blip_aggregated_metrics_{exp_name}.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_text_similarities(result_dict):
    all_averages = [result_dict[prompt]['text_similarities'] for prompt in result_dict]
    all_averages = np.array(all_averages).flatten()
    total_average = np.average(all_averages)
    total_std = np.std(all_averages)
    return total_average, total_std


if __name__ == '__main__':
    run()
