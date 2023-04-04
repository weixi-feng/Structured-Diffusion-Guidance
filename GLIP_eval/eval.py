import matplotlib.pyplot as plt

import requests
from io import BytesIO
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import os
from collections import defaultdict
import json
from tqdm import tqdm
import sys


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def load(dir):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(dir).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    image_dir = args.image_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    image_names = sorted(os.listdir(image_dir))

    config_file = "configs/pretrain/glip_Swin_L.yaml"
    weight_file = "MODEL/glip_large_model.pth" # NOTE

    # update the config options with the config file
    # manual override some options
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )

    plus = 1 if glip_demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD" else 0

    grounding_results = {}
    blockPrint()
    for file in tqdm(image_names):
        image = load(os.path.join(image_dir, file))
        caption = os.path.splitext(file.split("-")[-1])[0] # NOTE
        result, top_predictions = glip_demo.run_on_web_image(image, caption, args.thresh)
        fig = plt.figure(figsize=(5,5))
        plt.imshow(result[:, :, [2, 1, 0]])
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/{file}")
        plt.close()

        scores = top_predictions.get_field("scores")
        labels = top_predictions.get_field("labels")
        bbox = top_predictions.bbox
        entities = glip_demo.entities
        
        new_labels = []
        for i in labels:
            if i <= len(entities):
                new_labels.append(entities[i-plus])
            else:
                new_labels.append("object")

        bbox_by_entities = defaultdict(list)
        for l, score, coord in zip(new_labels, scores, bbox):
            bbox_by_entities[l].append((score.item(), coord.tolist()))
        grounding_results[file] = bbox_by_entities

    with open(f"{output_dir}/glip_results.json", "w") as file:
        json.dump(grounding_results, file, indent=4, separators=(",",":"), sort_keys=True)