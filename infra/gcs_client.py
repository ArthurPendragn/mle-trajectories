"""Use the selected maintainer gcloud profile, preserving the agent's ADC."""
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import subprocess
import uuid

import google_crc32c
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY
from google.api_core.exceptions import PreconditionFailed
from google.oauth2.credentials import Credentials


def client():
    def refresh(request, scopes):
        token = subprocess.check_output(['gcloud', 'auth', 'print-access-token'], text=True).strip()
        expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=45)
        return token, expiry

    credentials = Credentials(token=None, scopes=['https://www.googleapis.com/auth/devstorage.full_control'],
                              refresh_handler=refresh)
    credentials.refresh(None)
    return storage.Client(project='mle-agents', credentials=credentials)


def upload_file(bucket, name, path, workers=8, resume_prefix=None):
    """Upload large files in parallel multipart parts, compose, then verify CRC."""
    destination = bucket.blob(name)
    if destination.exists():
        return False
    size = path.stat().st_size
    if size <= 8 * 1024 * 1024:
        destination.upload_from_filename(str(path), if_generation_match=0, timeout=120, checksum='crc32c')
        return True
    temporary = resume_prefix or f"_uploads/{uuid.uuid4().hex}"
    print(f"Staging {name} at {temporary}; use --resume-train-parts for interrupted training uploads", flush=True)
    part_size = 4 * 1024 * 1024
    offsets = list(range(0, size, part_size))
    created = []
    verified = False
    def upload_part(offset):
        blob = bucket.blob(f"{temporary}/part-{offset:020d}")
        blob.storage_class = 'STANDARD'
        with path.open('rb') as stream:
            stream.seek(offset)
            data = stream.read(min(part_size, size-offset))
        if resume_prefix:
            existing = bucket.get_blob(blob.name)
            if existing is not None:
                crc = base64.b64encode(google_crc32c.Checksum(data).digest()).decode()
                if existing.size != len(data) or existing.crc32c != crc:
                    raise ValueError(f'Resumed part differs from local bytes: {blob.name}')
                return existing
        try:
            blob.upload_from_string(data, if_generation_match=0, timeout=300,
                                    retry=DEFAULT_RETRY.with_deadline(600), checksum='crc32c')
        except PreconditionFailed:
            # A lost response can leave a successful upload behind. Accept only
            # the same bytes, even when the retry sees the creation precondition.
            existing = bucket.get_blob(blob.name)
            crc = base64.b64encode(google_crc32c.Checksum(data).digest()).decode()
            if existing is None or existing.size != len(data) or existing.crc32c != crc:
                raise
            return existing
        return blob
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for count, blob in enumerate(pool.map(upload_part, offsets), 1):
                created.append(blob)
                if count % 25 == 0 or count == len(offsets):
                    print(f"{name}: {count}/{len(offsets)} parts uploaded", flush=True)
        sources = created.copy()
        level = 0
        while len(sources) > 32:
            merged = []
            for index in range(0, len(sources), 32):
                group = sources[index:index+32]
                signature = hashlib.sha256('\n'.join(
                    f'{source.name}:{source.generation}' for source in group
                ).encode()).hexdigest()
                blob = bucket.blob(f"{temporary}/compose-{level}-{index}-{signature}")
                blob.storage_class = 'STANDARD'
                existing = bucket.get_blob(blob.name) if resume_prefix else None
                if existing is not None:
                    if existing.size != sum(source.size for source in group):
                        raise ValueError(f'Resumed composition has incorrect size: {blob.name}')
                    blob = existing
                else:
                    blob.compose(group, if_generation_match=0, timeout=120)
                created.append(blob)
                merged.append(blob)
            sources = merged
            level += 1
        destination.compose(sources, if_generation_match=0, timeout=120)
        checksum = google_crc32c.Checksum()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                checksum.update(chunk)
        if destination.size != size or destination.crc32c != base64.b64encode(checksum.digest()).decode():
            raise ValueError(f'Composed file failed checksum/size validation: {name}')
        verified = True
        return True
    finally:
        # Delete only generation-specific temporary objects from this invocation.
        # Listing also catches intermediate objects retained by an earlier attempt.
        # Keep interrupted parts for a checksum-validated resume.
        if verified:
            parts = list(bucket.client.list_blobs(bucket, prefix=temporary + '/'))
            for offset in range(0, len(parts), 100):
                with bucket.client.batch():
                    for blob in parts[offset:offset+100]:
                        blob.delete(if_generation_match=blob.generation)
