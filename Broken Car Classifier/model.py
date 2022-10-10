# %%
#! [ -e /content ] && pip install -Uqq fastbook

# %%
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.widgets import *

# %%
# use duckduckgo for image search
search_images_ddg

# %%
ims = search_images_ddg('Broken Car')

# %%
ims[6]

# %%
types = 'Broken Car', 'Normal Car'
path = Path ('cars')

# %%
if not path.exists():
    path.mkdir()
for o in types:
    dest = (path/o)
    dest.mkdir(exist_ok=True)
    results = search_images_ddg(f'{o} cars')
    download_images(dest,urls=results)

# %%
fns = get_image_files(path)
fns

# %%
failed = verify_images(fns)
failed

# %%
# delete the corrupted downloaded images
failed.map(Path.unlink);

# %%
# creating the Datablock
brokencars = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))

# %%
brokencars = brokencars.new(
    item_tfms=RandomResizedCrop(128, min_scale=0.6),
    batch_tfms=aug_transforms())
dls = brokencars.dataloaders(path)

# %%
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(4)

# %%
cleaner = ImageClassifierCleaner(learn)
cleaner

# %%
# apply the changes that was made with UI
for idx in cleaner.delete():
    print(idx)
    cleaner.fns[idx].unlink()
for idx,cat in cleaner.change():
    print(idx)
    shutil.move(str(cleaner.fns[idx]),path/cat)

# %%
learn.export()
path = Path()
path.ls(file_exts='.pkl')


