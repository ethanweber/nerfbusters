"""Script to download objaverse dataset.
# https://objaverse.allenai.org/docs/download
"""

import objaverse
# print(objaverse.__version__)
import multiprocessing
import random

random.seed(42)

uids = objaverse.load_uids()

# uids = ["7561ffb58b7e4cd9a25965a08f1d476b"]
# selected_uids = uids[:10]
# annotations = objaverse.load_annotations(selected_uids)

target_categories = ["furniture-home"]
target_tags = ["house"]

annotations = objaverse.load_annotations()
selected_annotations = {}
for uid, ann in annotations.items():
    tags = set()
    cats = set()
    for tag in ann["tags"]:
        tags.add(tag["name"])
    for cat in ann["categories"]:
        cats.add(cat["name"])
    # check if the object has the target categorys and tags
    if target_categories and not cats.intersection(target_categories):
        continue
    if target_tags and not tags.intersection(target_tags):
        continue
    selected_annotations[uid] = ann
    
num_anns = len(selected_annotations)
print(f"Found {num_anns} annotations")

uids = list(selected_annotations.keys())
print(uids)
# save the uids
with open("objaverse/uids.txt", "w") as f:
    for uid in uids:
        f.write(f"{uid}\n")

# download the objects
processes = multiprocessing.cpu_count()
objects = objaverse.load_objects(
    uids=uids,
    download_processes=processes
)