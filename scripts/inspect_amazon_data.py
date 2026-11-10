"""Count TFRecord examples without loading tensors or model weights."""
import gzip
import json
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def count_file(p):
    count = 0
    with gzip.open(p, 'rb') as f:
        while header := f.read(12):
            if len(header) != 12:
                raise ValueError(f'Truncated header: {p}')
            length = struct.unpack('<Q', header[:8])[0]
            if len(f.read(length + 4)) != length + 4:
                raise ValueError(f'Truncated record: {p}')
            count += 1
    return count

if __name__ == '__main__':
    result = {}
    for category in sorted(Path('data/amazon_data').iterdir()):
        result[category.name] = {}
        for split in sorted(category.iterdir()):
            files = sorted(split.glob('*.tfrecord.gz'))
            with ThreadPoolExecutor(max_workers=8) as pool:
                counts = list(pool.map(count_file, files))
            result[category.name][split.name] = {'records': sum(counts), 'files':len(files), 'bytes':sum(p.stat().st_size for p in files)}
        print(category.name, result[category.name], flush=True)
    Path('data/dataset_stats.json').write_text(json.dumps(result, indent=2)+'\n')
