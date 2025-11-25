"""
    Downloading Dataset from HuggingFace
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def download_dataset(args):
    """
        Downloading dataset from HuggingFace
    """
    os.environ["HF_TOKEN"] = args.hf_token  # or rely on huggingface-cli login

    if not args.per_task_download:

        local_dir = os.path.join(args.local_dir_root, args.repo_id)

        print("downloading: {}".format(args.repo_id))
        print("saving under: {}".format(local_dir))

        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            max_workers=1,
        )

    else:
        print("Downloading per-task lerobot v3 LIBERO dataset from HuggingFace zak1030. Use with caution.")
        task_ids = ["libero_spatial_image_v3", "libero_object_image_v3", "libero_10_image_v3", "libero_goal_image_v3"]
        for task_id in task_ids:
            
            repo_id = "zak1030/{}".format(task_id)
            local_dir = os.path.join(args.local_dir_root, args.repo_id)

            print("downloading: {}".format(repo_id))
            print("saving under: {}".format(local_dir))

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                max_workers=1,
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", 
                        help="HuggingFace Token",
                        dtype=str,
                        required=True
                        )
    parser.add_argument("--repo_id",
                        help="Dataset Repo ID. By default it downloads the HuggingFaceVLA/libero datset.",
                        dtype=str,
                        default="HuggingFaceVLA/libero",
                        )
    parser.add_argument("--per_task_download",
                        help="Whether to download lerobot v3.0 LIBERO dataset per-task.",
                        dtype=bool,
                        default=False
                        )
    parser.add_argument("--local_dir_root",
                        help=r"Local Directory Root for saving the datset. \
                            By default it is stored under lerobot/data. Dataset will be stored under {local_dir_root}/{repo_id}.",
                        dtype=str,
                        default="../data"
                        )
    args = parser.parse_args()

    download_dataset(args)

if __name__ == "__main__":
    main()
