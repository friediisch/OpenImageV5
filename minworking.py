from fastai.vision import *
from pathlib import Path
PATH = Path('path')

paths = [PATH/'to/train/file1.jpg', PATH/'to/train/file2.jpg', PATH/'to/train/file3.jpg']
bboxes = [
    [[[32, 1, 4, 5], [89, 5, 37, 5]], ['Footwear', 'Person']],
    [[[32, 2, 14, 35]], ['Table']],
    [[[2, 132, 41, 51], [9, 25, 39, 3]], ['Cup', 'Plate']],
]

paths = [str(item) for item in paths]
img2bbox = dict(zip(paths, bboxes))
img2bbox.keys()

get_y_func = lambda o: img2bbox[str(o)]

df = pd.DataFrame({"Path": ['to/train/file1.jpg', 'to/train/file2.jpg', 'to/train/file3.jpg'], 
                   "ID": [item.split('/')[-1].split('.')[0] for item in paths]}, columns=["Path", "ID"])

def get_data(bs, size):
    src = ImageList.from_df(df, path=PATH)
    src = src.split_none()
    src = src.label_from_func(get_y_func)
    data = data.transform(get_transforms(), size=128, tfm_y=True)
    data = data.databunch(path=PATH, bs=16, collate_fn=bb_pad_collate)
    data = data.normalize(imagenet_stats)
    return src.databunch(path=PATH, bs=bs, collate_fn=bb_pad_collate)

data = get_data(bs=1, size=128)