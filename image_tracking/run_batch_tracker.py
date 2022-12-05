from new_tracker import runner
import os
import re

def run_by_seq_name():
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


def run_by_seq_date():
    date_file = "E:\\UWCR dataset\\seqs.txt"
    base_dir = "E:\\UWCR dataset"

    date_pattern = re.compile("\d{4}_\d{2}_\d{2}_.+")
    dates = []

    with open(date_file, 'r') as f:
        dates = f.read().splitlines()


    for i, date in enumerate(dates):
        seqs_name = os.listdir(os.path.join(base_dir, date))
        print("%d/%d\t%s" % (i+1, len(dates), date))

        for j, seq in enumerate(seqs_name):
            seq_path = os.path.join(base_dir, date, seq, "images_0")
            runner(seq_path)
            print("\t%d / %d \t %s"%(j+1, len(seqs_name), seq))

if __name__ == "__main__":
    run_by_seq_date()
