import hashlib

def sha256_checksum(path):
    """Return SHA-256 checksum of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        sha.update(f.read())
    return sha.hexdigest()
