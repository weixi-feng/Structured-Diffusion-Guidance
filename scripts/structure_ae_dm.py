import argparse, os, sys, glob
from collections import defaultdict
from ossaudiodev import SNDCTL_SEQ_CTRLRATE
from ast import parse
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from structured_stable_diffusion.util import instantiate_from_config
from structured_stable_diffusion.models.diffusion.ddim import DDIMSampler
from structured_stable_diffusion.models.diffusion.plms import PLMSSampler



from attentd_and_excite.utils import vis_utils
import sng_parser
import stanza
from nltk.tree import Tree
nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
import pdb
import json

from diffusers import LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, PNDMScheduler
from composable_stable_diffusion.pipeline_composable_stable_diffusion import \
    ComposableStableDiffusionPipeline

from pipeline_attend_and_excite_structure import StableDiffusionAttendAndExcitePipeline


def preprocess_prompts(prompts):
    if isinstance(prompts, (list, tuple)):
        return [p.lower().strip().strip(".").strip() for p in prompts]
    elif isinstance(prompts, str):
        return prompts.lower().strip().strip(".").strip()
    else:
        raise NotImplementedError


def get_all_nps(tree, full_sent, tokens=None, highest_only=False, lowest_only=False):
    start = 0
    end = len(tree.leaves())

    idx_map = get_token_alignment_map(tree, tokens)

    
    def get_sub_nps(tree, left, right):
        sub_nps = []
        sub_nouns = []
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return sub_nps, sub_nouns

        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
        assert right - left == n_leaves
        if tree.label() == 'NP' and n_leaves > 1:
            sub_nps.append([" ".join(tree.leaves()), (int(min(idx_map[left])), int(min(idx_map[right])))])
            sub_nouns.append([[subtree.leaves()[0] for subtree in tree if subtree.label()[:2] == "NN"], (int(min(idx_map[left])), int(min(idx_map[right])))])
            if highest_only and sub_nps[-1][0] != full_sent: return sub_nps
        for i, subtree in enumerate(tree):
            # sub_nps += get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
            partial_sub_nps, partial_sub_nouns = get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
            sub_nps += partial_sub_nps
            sub_nouns += partial_sub_nouns
        return sub_nps, sub_nouns
    
    all_nps, all_nouns = get_sub_nps(tree, left=start, right=end)

    def find_lowest(all):
        lowest = []
        for i in range(len(all)):
            span = all[i][1]
            is_lowest = True
            for j in range(len(all)):
                if i == j: continue
                span2 = all[j][1]
                if span2[0] >= span[0] and span2[1] <= span[1]:
                    is_lowest = False
                    break
            if is_lowest:
                lowest.append(all[i])
        return lowest

    lowest_nps = find_lowest(all_nps)
    lowest_nouns = find_lowest(all_nouns)

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

    return all_nps, spans, lowest_nps, lowest_nouns


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


def get_all_spans_from_scene_graph(caption):
    caption = caption.strip()
    graph = sng_parser.parse(caption)
    nps = []
    spans = []
    words = caption.split()
    for e in graph['entities']:
        start, end = e['span_bounds']
        if e['span'] == caption: continue
        if end-start == 1: continue
        nps.append(e['span'])
        spans.append(e['span_bounds'])
    for r in graph['relations']:
        start1, end1 = graph['entities'][r['subject']]['span_bounds']
        start2, end2 = graph['entities'][r['object']]['span_bounds']
        start = min(start1, start2)
        end = max(end1, end2)
        if " ".join(words[start:end]) == caption: continue
        nps.append(" ".join(words[start:end]))
        spans.append((start, end))
    
    return [caption] + nps, [(0, len(words))] + spans, None


def single_align(main_seq, seqs, spans, dim=1):
    main_seq = main_seq.transpose(0, dim)
    for seq, span in zip(seqs, spans):
        seq = seq.transpose(0, dim)
        start, end = span[0]+1, span[1]+1
        seg_length = end - start
        main_seq[start:end] = seq[1:1+seg_length]

    return main_seq.transpose(0, dim)


def multi_align(main_seq, seq, span, dim=1):
    seq = seq.transpose(0, dim)
    main_seq = main_seq.transpose(0, dim)
    start, end = span[0]+1, span[1]+1
    seg_length = end - start

    main_seq[start:end] = seq[1:1+seg_length]

    return main_seq.transpose(0, dim)


def align_sequence(main_seq, seqs, spans, dim=1, single=False):
    aligned_seqs = []
    if single:
        return [single_align(main_seq, seqs, spans, dim=dim)]
    else:
        for seq, span in zip(seqs, spans):
            aligned_seqs.append(multi_align(main_seq.clone(), seq, span, dim=dim))
        return aligned_seqs


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--parser_type",
        type=str,
        choices=['constituency', 'scene_graph'],
        default='constituency'
    )
    parser.add_argument(
        "--conjunction",
        action='store_true',
        help='If True, the input prompt is a conjunction of two concepts like "A and B"'
    )
    parser.add_argument(
        "--save_attn_maps",
        action='store_true',
        help='If True, the attention maps will be saved as a .pth file with the name same as the image'
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help='resume generation',
    )


    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # pipe = ComposableStableDiffusionPipeline.from_pretrained(
    #         "CompVis/stable-diffusion-v1-4"
    # ).to(device)

    pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4").to(device)

    generator = torch.Generator("cuda").manual_seed(opt.seed)




    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        opt.from_file = ""
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            try:
                opt.end_idx = len(data) if opt.end_idx == -1 else opt.end_idx
                data = data[:opt.end_idx]
                data, filenames = zip(*[d.strip("\n").split("\t") for d in data])
                data = list(chunk(data, batch_size))
            except:
                data = [batch_size * [d] for d in data]




    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    if opt.resume:
        idx = len(os.listdir(sample_path))
    else:
        idx = 0
    
    start_idx = batch_size * idx

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            # with model.ema_scope():
            if True:

                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for bid, prompts in enumerate(tqdm(data, desc="data")):
                        if bid < start_idx:
                            continue
                        prompts = preprocess_prompts(prompts)

                        # uc = None
                        # if opt.scale != 1.0:
                        #     uc = model.get_learned_conditioning(batch_size * [""])

                        # c = model.get_learned_conditioning(prompts)

                        try:
                            if opt.parser_type == 'constituency':
                                doc = nlp(prompts[0])
                                mytree = Tree.fromstring(str(doc.sentences[0].constituency))

                                # tokens = model.cond_stage_model.tokenizer.tokenize(prompts[0])
                                tokens = None
                                nps, spans, noun_chunk, nouns = get_all_nps(mytree, prompts[0], tokens, lowest_only=True)
                                # we need noun_chunk for our implementaiton
                                # print(mytree, nps, spans, noun_chunk)
                            elif opt.parser_type == 'scene_graph':
                                nps, spans, noun_chunk = get_all_spans_from_scene_graph(prompts[0].split("\t")[0])
                            else:
                                raise NotImplementedError
                        except:
                            print(f"{prompts[0]} parsing failed")
                            raise
                            continue
                        
                        nps = [[np]*len(prompts) for np in nps]
                        
                        # if opt.conjunction:
                        #     c = [model.get_learned_conditioning(np) for np in nps]
                        #     k_c = [c[0]] + align_sequence(c[0], c[1:], spans[1:])
                        #     v_c = align_sequence(c[0], c[1:], spans[1:], single=True)
                        #     c = {'k': k_c, 'v': v_c}
                        # else:
                        #     c = [model.get_learned_conditioning(np) for np in nps]
                        #     k_c = c[:1]
                        #     v_c = [c[0]] + align_sequence(c[0], c[1:], spans[1:])
                        #     c = {'k': k_c, 'v': v_c}
                        prompts_list = [i[0] for i in nps]
                        prompts = " | ".join(prompts_list) 
                        # prompts = [prompts]
                        # weights = [opt.scale] * len(nps) 
                        token_indices = [chunk[-1] for _, chunk in noun_chunk]
                        # image = pipe(prompts, guidance_scale=opt.scale, num_inference_steps=opt.ddim_steps, 
                        #              weights=weights, generator=generator).images[0]
                        token_indices = [token_indices]

                        image = pipe(prompt=prompts,
                                     token_indices=token_indices,
                                     noun_chunks = noun_chunk,
                                     nouns = nouns,
                                     # attention_res=RunConfig.attention_res,
                                     guidance_scale=opt.scale,
                                     generator=generator,
                                     num_inference_steps=opt.ddim_steps,).images[0]


                        attn_img = vis_utils.show_cross_attention(attention_store=pipe.attention_store,
                                   prompt=prompt,
                                   tokenizer=pipe.tokenizer,
                                   res=16,
                                   from_where=("up", "down", "mid"),
                                   indices_to_alter=token_indices,
                                   orig_image=None)
                        print(attn_img)

                        x_checked_image_torch = [image]
                        # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        # samples_ddim, intermediates = sampler.sample(S=opt.ddim_steps,
                        #                                  conditioning=c,
                        #                                  batch_size=opt.n_samples,
                        #                                  shape=shape,
                        #                                  verbose=False,
                        #                                  unconditional_guidance_scale=opt.scale,
                        #                                  unconditional_conditioning=uc,
                        #                                  eta=opt.ddim_eta,
                        #                                  x_T=start_code,
                        #                                  save_attn_maps=opt.save_attn_maps)

                        # x_samples_ddim = model.decode_first_stage(samples_ddim)
                        # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        # x_checked_image = x_samples_ddim

                        # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for sid, img in enumerate(x_checked_image_torch):
                                # x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                # img = Image.fromarray(x_sample.astype(np.uint8))          
                                try:
                                    count = bid * opt.n_samples + sid
                                    safe_filename = f"{n}-{count}-" + (filenames[count][:-4])[:150] + ".jpg"
                                except:
                                    safe_filename = f"{base_count:05}-{n}-{prompts}"[:100] + ".jpg"
                                img.save(os.path.join(sample_path, f"{safe_filename}"))
                                
                                if opt.save_attn_maps:
                                    raise NotImplemented
                                    # torch.save(sampler.attn_maps, os.path.join(sample_path, f'{safe_filename}.pt'))
                                base_count += 1  
# 
#                         if not opt.skip_grid:
#                             all_samples.append(x_checked_image_torch)
# 
#                 if not opt.skip_grid:
#                     # additionally, save as grid
#                     grid = torch.stack(all_samples, 0)
#                     grid = rearrange(grid, 'n b c h w -> (n b) c h w')
#                     grid = make_grid(grid, nrow=n_rows)
# 
#                     # to image
#                     grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
#                     img = Image.fromarray(grid.astype(np.uint8))
#                     img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
#                     grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
