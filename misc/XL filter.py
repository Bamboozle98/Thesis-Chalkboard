import pandas as pd
import os
from Models.SuperPixel_Transformer.config import (
    g_dir,      # your WSI folder
    g_ann       # your Excel annotations path
)

# paths
ann_path = g_ann
wsi_dir  = g_dir
out_tsv  = r"E:\Geradt\rs_filtered.tsv"

# load your Excel sheet
df = pd.read_excel(ann_path, sheet_name=0, dtype={"Case#": str})
df = df.rename(columns=lambda c: c.strip())

# get the case IDs from your .tif/.tiff files, normalized as "1","2",...
tifs = []
for f in os.listdir(wsi_dir):
    if f.lower().endswith(('.tif', '.tiff')):
        stem = os.path.splitext(f)[0]
        try:
            norm = str(int(stem))
        except ValueError:
            # skip any files whose name isn't an integer
            continue
        tifs.append(norm)
tifs = set(tifs)

# filter the DataFrame to only those Case# in our slide set
df_filt = df[df["Case#"].isin(tifs)].copy()
print(f"Keeping {len(df_filt)} of {len(df)} annotations")

# write out a clean TSV
df_filt.to_csv(out_tsv, sep='\t', index=False)
print(f"Wrote filtered annotations to {out_tsv}")

