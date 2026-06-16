# Get results of evaluation

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("output_json", type=str)
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results

df = pd.read_json(args.filename, orient="records", lines=True)

# Measure overall success

print("Summary")
print("=======")
print(f"Total images: {len(df)}")
print(f"Total prompts: {len(df.groupby('metadata'))}")
print(f"% correct images: {df['correct'].mean():.2%}")
print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
print()

# By group

task_scores = []

print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
print()

print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")

# save to a json file
import json
output = {
    "total_images": len(df),
    "total_prompts": len(df.groupby('metadata')),
    "percent_correct_images": df['correct'].mean(),
    "percent_correct_prompts": df.groupby('metadata')['correct'].any().mean(),
    "task_scores": {tag: task_df['correct'].mean() for tag, task_df in df.groupby('tag', sort=False)},
    "overall_score": np.mean(task_scores),
}
with open(args.output_json, "w") as f:
    json.dump(output, f, indent=2)