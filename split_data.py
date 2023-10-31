import os 
import math
src = './ABC-6K.txt' 
dest = 'split_data_dir'

split = 8


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

with open(src, 'r') as f:
    lines = f.readlines()
    chunk_size = math.ceil(len(lines) / split)
    split_lines = list(chunks(lines, chunk_size))


    for i in range(split):
        # print(len(split_lines[i]))

        with open(os.path.join(dest, f'{str(i)}.txt'), 'w') as writer:
            writer.writelines(split_lines[i])

