import argparse
import os
import re

_EXLUDE_NAME = ("__pycache__",)
_CHECK_FILE_EXTENSIONs = (".cpp", ".c", ".py")

_HEADER = (
    r"((//)|#) Copyright \(c\) 20[0-9]{2} Helixon Limited.",
    r"((//)|#)",
    r"((//)|#) This file is a part of ProtMyth and is released under the MIT License.",
    r"((//)|#) Thanks for using ProtMyth!",
)


def check_header(file_path: str):
    if file_path.endswith(_CHECK_FILE_EXTENSIONs):
        with open(file_path) as f:
            for h in _HEADER:
                line = f.readline()[:-1]
                match = re.match(h, line)
                if match is None:
                    return False
                st, ed = match.span()
                if st != 0 or ed != len(line):
                    return False
            if f.readline().strip():
                return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("check_dirs", nargs="+")
    parser.add_argument("-o", "--output", default="copyright-check.log")
    args = parser.parse_args()

    has_missing = False
    logfile = open(args.output, "w")

    for d in args.check_dirs:
        for root, dirs, files in os.walk(d):
            dirs[:] = [_ for _ in dirs if _ not in _EXLUDE_NAME]
            for f in files:
                fpath = os.path.join(root, f)
                if not check_header(fpath):
                    message = f"{fpath}:1: [E999] Copyright header is missing."
                    has_missing = True
                    print(message)
                    logfile.write(message + "\n")
    logfile.close()
    if has_missing:
        exit(1)
