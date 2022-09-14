import glob
import os
import csv


if __name__ == '__main__':
    # dir_path = "E:\\UWCR Data\\2019_04_30\\2019_04_30_mlms001\\images_0\\Cyclists"
    dir_path ="D:\\UWCR Data2\\2019_04_09\\2019_04_09_pms3000\\images_0\\Trim"

    files = glob.glob("*.txt", root_dir=dir_path)
    for fname in files:
        if fname == "labels.txt":
            continue
        raw_bbox = []
        with open(os.path.join(dir_path, fname), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                # Don't trim if already trimmed
                if len(line) == 5:
                    print("Already trimmed")
                    f.close()
                    exit()
                # strip the confidence
                raw_bbox.append(line[:-1])

        with open(os.path.join(dir_path, fname), 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            for box in raw_bbox:
                writer.writerow(box)
