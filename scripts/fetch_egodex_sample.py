"""Read a few training-set members from Apple's ZIP using bounded HTTP ranges."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class RangeFile(io.RawIOBase):
    def __init__(self, url, budget=256 * 1024**2):
        self.url = url
        with urlopen(Request(url, method='HEAD'), timeout=30) as response:
            self.size = int(response.headers['Content-Length'])
            self.etag = response.headers.get('ETag')
            self.modified = response.headers.get('Last-Modified')
        self.position = 0
        self.downloaded = 0
        self.budget = budget

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        new = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if new < 0 or new > self.size:
            raise ValueError('Archive seek outside bounds')
        self.position = new
        return new

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if not size:
            return b''
        if size > 64 * 1024**2 or self.downloaded + size > self.budget:
            raise ValueError('Refusing oversized archive request')
        start, end = self.position, self.position + size - 1
        headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}
        if self.etag:
            headers['If-Match'] = self.etag
        with urlopen(Request(self.url, headers=headers), timeout=60) as response:
            expected = f'bytes {start}-{end}/{self.size}'
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Server ignored the exact byte range; refusing full download')
            body = response.read(size + 1)
        if len(body) != size:
            raise ValueError('Incorrect HTTP range length')
        self.position += size
        self.downloaded += size
        return body


def fetch(part, members, destination):
    if part not in ('part1', 'part2', 'part3', 'part4', 'part5'):
        raise ValueError('Only official training partitions are allowed')
    url = f'https://ml-site.cdn-apple.com/datasets/egodex/{part}.zip'
    remote = RangeFile(url)
    with zipfile.ZipFile(remote) as archive:
        entries = archive.infolist()
        if not members:
            index = [{'name': i.filename, 'size': i.file_size, 'compressed_size': i.compress_size}
                     for i in entries]
            cache = ROOT / '.cache' / 'egodex'
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f'{part}-index.json').write_text(json.dumps(index))
            counts = Counter(str(PurePosixPath(i.filename).parent) for i in entries if i.filename.endswith('.hdf5'))
            print(json.dumps({'tasks': counts, 'downloaded_bytes': remote.downloaded,
                              'license_members': [i.filename for i in entries if any(s in i.filename.lower() for s in ('license','terms','readme'))]}, indent=2))
            return
        destination = Path(destination)
        for member in members:
            name = PurePosixPath(member)
            if name.is_absolute() or '..' in name.parts or name.suffix not in ('.hdf5', '.mp4'):
                raise ValueError('Only safe HDF5/MP4 member paths are permitted')
            info = archive.getinfo(member)
            if info.file_size > 64 * 1024**2 or info.compress_size > 64 * 1024**2:
                raise ValueError(f'Member too large for sample probe: {member}')
            path = destination / name
            if path.exists():
                raise FileExistsError(f'Preserving existing sample: {path}')
            payload = archive.read(member)  # ZIP CRC is verified by zipfile.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            provenance = {
                'dataset': 'EgoDex', 'publisher': 'Apple', 'official_split': 'train',
                'archive_url': url, 'archive_etag': remote.etag,
                'archive_last_modified': remote.modified, 'member': member,
                'member_crc32': f'{info.CRC:08x}', 'sha256': hashlib.sha256(payload).hexdigest(),
                'bytes': len(payload), 'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
                'license_statement': 'CC-by-NC-ND; version not specified in publisher README',
                'license_source': 'https://github.com/apple-aiml-research/ml-egodex#license',
                'use_scope': 'Local noncommercial research probe; no source or adapted material redistribution authorized by this workflow.',
            }
            path.with_suffix(path.suffix + '.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
            print(f'{path}: {len(payload)} bytes; sha256 {provenance["sha256"]}')
        print(f'Total network bytes this invocation: {remote.downloaded}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', default='part2')
    parser.add_argument('--member', action='append', default=[])
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/egodex')
    args = parser.parse_args()
    fetch(args.part, args.member, args.destination)
