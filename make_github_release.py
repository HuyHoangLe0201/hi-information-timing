r"""Copy the released analysis into the GitHub working tree, ../github_repo.

The release is the code/ folder of the submission package (make_mssp_submission.py)
less the underscore-prefixed files, which are the helpers that edited the
manuscript and their captured output rather than analysis; _source.py stays,
because the audit layers import it.  Nothing is deleted from the working tree
that this script did not put there: files it copied before and no longer
selects are removed, anything else (the .git folder, a LICENSE the authors add)
is left alone.

    python make_github_release.py   ->  ../github_repo/  (then commit and push)
"""
import io
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.path.dirname(HERE), "github_repo")
MANIFEST = os.path.join(DEST, ".release_manifest")

SKIP_DIRS = {".backup", "__pycache__", ".git"}
SKIP_EXT = {".npz", ".pyc", ".pdf", ".png", ".tex", ".log", ".aux"}
SKIP_NAMES = {"Cover_letter.md", "OUTLINE.md", "SPLIT.md"}
KEEP_UNDERSCORE = {"_source.py"}

GITIGNORE = """# feature caches rebuilt from the public datasets by the extraction scripts
*.npz
__pycache__/
*.pyc
.backup/
"""


def selected():
    out = []
    for name in sorted(os.listdir(HERE)):
        p = os.path.join(HERE, name)
        if os.path.isdir(p) or name in SKIP_DIRS or name in SKIP_NAMES:
            continue
        if os.path.splitext(name)[1].lower() in SKIP_EXT:
            continue
        if name.startswith("_") and name not in KEEP_UNDERSCORE:
            continue
        out.append(name)
    return out


os.makedirs(DEST, exist_ok=True)
files = selected()
old = []
if os.path.exists(MANIFEST):
    old = [x for x in io.open(MANIFEST, encoding="utf-8").read().split("\n") if x]
for name in set(old) - set(files):
    p = os.path.join(DEST, name)
    if os.path.exists(p):
        os.remove(p)
for name in files:
    shutil.copy2(os.path.join(HERE, name), os.path.join(DEST, name))
io.open(os.path.join(DEST, ".gitignore"), "w", encoding="utf-8",
        newline="\n").write(GITIGNORE)
io.open(MANIFEST, "w", encoding="utf-8", newline="\n").write("\n".join(files) + "\n")
size = sum(os.path.getsize(os.path.join(HERE, n)) for n in files) / 1048576.0
print("%d files, %.1f MB -> %s" % (len(files), size, DEST))
