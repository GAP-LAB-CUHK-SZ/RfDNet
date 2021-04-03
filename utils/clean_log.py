from pathlib import Path
from shutil import rmtree
import subprocess
import re
import os

if __name__ == '__main__':
    log_dir = Path('out/iscnet')
    for subdir in log_dir.iterdir():
        if not subdir.is_dir():
            continue

        remove_flag = 0
        logfile = subdir.joinpath('log.txt')
        if not os.path.exists(logfile):
            continue
        line = subprocess.check_output(['tail', '-100', logfile]).decode('ascii')
        if 'finished' not in line:
            epoch_info = re.findall(r'Epoch (\d*):', line)
            last_epoch = int(epoch_info[-1]) if len(epoch_info) else 1e5
            if (not len(list(subdir.glob('*.pth')))) or last_epoch<=10:
                if len(list(subdir.joinpath('visualization').iterdir()))<5 :
                    remove_flag = 1

        if remove_flag:
            rmtree(subdir)


