import json 
import os 
from collections import defaultdict
import numpy as np

src = './mine_method_blip'
clips_result = [os.path.join(src, f) for f in os.listdir(src)]

clips_json = defaultdict(list)

for clip in clips_result:
    with open(clip, 'r') as f:
        clip = json.load(f) 
        for k, v in clip.items():
            clips_json[k].append(v)

for k, v in clips_json.items():
    print(k, np.asarray(v).mean())
