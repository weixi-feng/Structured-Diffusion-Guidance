import os 
import json 
import random
from collections import defaultdict

from numpy import single
from torch import numel


src = './ae_prompt.txt' 


with open(src, 'r') as f:
    ae_prompt = json.load(f) 


connect = [' , ', ' and ', ' with ']

num_samples = 500

level = 10


# break down 
single_object = defaultdict(list)
for k, vs in ae_prompt.items():

    for v in vs:
        single = v.split('and') if 'and' in v else v.split('with')
        for s in single:
            single_object[k].append(s.strip())

single = dict()
for k, v in single_object.items():
    single[k] = set(v)


level_prompts = dict()
for l in range(1, level):
    prompts = list() 

    for _ in range(num_samples):
        p = list() 
        for i in range(l):
            key = random.choice(list(single.keys()))
            p.append(random.choice(list(single[key])))
            if i != l-1:
                p.append(random.choice(connect))
        p = "".join(p)
        prompts.append(p)

    with open(f'./level_{l}.txt', 'w') as f:
        f.write('\n'.join(prompts))





