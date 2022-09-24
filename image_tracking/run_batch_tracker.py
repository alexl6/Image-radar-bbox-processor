from email.mime import base
from new_tracker import runner
import os
import re

if __name__ == "__main__":
    seq_file = "D:\\sequences2+3.txt"
    base_dir = "D:\\UWCR Data2"

    date_pattern = re.compile("\d{4}_\d{2}_\d{2}")
    seqs = []

    with open(seq_file, 'r') as f:
        seqs = f.read().splitlines()

    i = 0
    for seq in seqs:
        date = date_pattern.search(seq).group()
        seq = os.path.join(base_dir, date, seq, "images_0")
        runner(seq)
        i += 1
        print("%d / %d \t %s" % (i, len(seqs), seq))
