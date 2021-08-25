import tarfile
from pathlib import Path

from tqdm import tqdm

submission_tar_path = "submission_compressed.tar.gz"
with tarfile.open(submission_tar_path, "w:gz") as tar:
    # iterate over files and add each to the TAR
    files = list(Path("submission_converted_cm_compressed_uint16").glob("*"))
    for file in tqdm(files, total=len(files)):
        tar.add(file, arcname=file.name)