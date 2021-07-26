import os
import logging


logger = logging.getLogger(__name__)


def download(url, path, save_file=None, md5=None):
    from six.moves.urllib.request import urlretrieve

    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
    save_file = os.path.join(path, save_file)

    if not os.path.exists(save_file) or compute_md5(save_file) != md5:
        logger.info("Downloading %s to %s" % (url, save_file))
        urlretrieve(url, save_file)
    return save_file


def extract(zip_file, member=None):
    import gzip
    import shutil
    import zipfile
    import tarfile

    zip_name, extension = os.path.splitext(zip_file)
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]

    if member is None:
        save_file = zip_name
    else:
        save_file = os.path.join(os.path.dirname(zip_name), os.path.basename(member))
    if os.path.exists(save_file):
        return save_file

    if member is None:
        logger.info("Extracting %s to %s" % (zip_file, save_file))
    else:
        logger.info("Extracting %s from %s to %s" % (member, zip_file, save_file))

    if extension == ".gz":
        with gzip.open(zip_file, "rb") as fin, open(save_file, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    elif extension == ".tar.gz" or extension == ".tar":
        if member is None:
            with tarfile.open(zip_file, "r") as fin:
                fin.extractall(save_file)
        else:
            with tarfile.open(zip_file, "r").extractfile(member) as fin, open(save_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    elif extension == ".zip":
        if member is None:
            with zipfile.ZipFile(zip_file) as fin:
                fin.extractall(save_file)
        else:
            with zipfile.ZipFile(zip_file).open(member, "r") as fin, open(save_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    else:
        raise ValueError("Unknown file extension `%s`" % extension)

    return save_file


def compute_md5(file_name, chunk_size=65536):
    import hashlib

    md5 = hashlib.md5()
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            md5.update(chunk)
            chunk = fin.read(chunk_size)
    return md5.hexdigest()


def get_line_count(file_name, chunk_size=8192*1024):
    count = 0
    with open(file_name, "rb") as fin:
        chunk = fin.read(chunk_size)
        while chunk:
            count += chunk.count(b"\n")
            chunk = fin.read(chunk_size)
    return count