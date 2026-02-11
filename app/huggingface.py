from huggingface_hub import (
    create_repo,
    upload_file,
    hf_hub_download,
    list_repo_files
)
import os
import logging
from glob import glob
from typing import Dict
from .utils import _get_home

logger = logging.getLogger(__name__)

HUGGINGFACE_USERNAME = os.environ.get('HUGGINGFACE_USERNAME', 'huseinzol05')

hf_hub_download_parameters = hf_hub_download.__code__.co_varnames


def download_files(repository, s3_file, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in hf_hub_download_parameters}
    files = {}
    for k, file in s3_file.items():
        base_path = repository
        logger.info(f'downloading frozen {base_path}/{file}')
        files[k] = hf_hub_download(base_path, file, **kwargs)
    return files


def download_from_dict(
    file,
    s3_file,
    package,
    quantized=False,
    **kwargs,
):
    kwargs = {k: v for k, v in kwargs.items() if k in hf_hub_download_parameters}
    home, _ = _get_home(package=package)
    if quantized:
        if 'quantized' not in file:
            f = file['model'].replace(home, '').split('/')
            raise ValueError(
                f'Quantized model for {f[1]} module is not available, please load normal model.'
            )
        model = 'quantized'
        logger.warning('Load quantized model will cause accuracy drop.')
    else:
        model = 'model'

    files = {}
    for k, file in s3_file.items():
        base_path, file = os.path.split(file)
        base_path = base_path.replace('/', '-')
        base_path = f'{HUGGINGFACE_USERNAME}/{base_path}'
        logger.info(f'downloading frozen {base_path}/{file}')
        files[k] = hf_hub_download(base_path, file, **kwargs)
    return files


def download_from_string(
    path,
    module,
    keys,
    package,
    quantized=False,
    **kwargs,
):
    model = path
    repo_id = f'{HUGGINGFACE_USERNAME}/{module}-{model}'
    kwargs = {k: v for k, v in kwargs.items() if k in hf_hub_download_parameters}

    if quantized:
        repo_id = f'{repo_id}-quantized'
        try:
            list_repo_files(repo_id)
        except BaseException:
            raise ValueError(
                f'Quantized model for `{os.path.join(module, model)}` is not available, please load normal model.'
            )
        logger.warning('Load quantized model will cause accuracy drop.')

    files = {}
    for k, file in keys.items():
        if '/' in file:
            splitted = os.path.split(file)
            repo_id_ = splitted[0].replace('/', '-')
            repo_id_ = f'{HUGGINGFACE_USERNAME}/{repo_id_}'
            file = splitted[1]
        else:
            repo_id_ = repo_id
        files[k] = hf_hub_download(repo_id_, file, **kwargs)

    return files


def check_file(
    file,
    package,
    base_url,
    s3_file=None,
    module=None,
    keys=None,
    quantized=False,
    **kwargs,
):
    """
    path = check_file(
        file=model,
        module=module,
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized=quantized,
        **kwargs,
    )

    or,

    check_file(path['multinomial'], s3_path['multinomial'], **kwargs)
    """
    if isinstance(file, dict) and isinstance(s3_file, dict):
        files = download_from_dict(
            file=file,
            s3_file=s3_file,
            package=package,
            quantized=quantized,
            **kwargs,
        )
    else:
        files = download_from_string(
            path=file,
            module=module,
            keys=keys,
            package=package,
            quantized=quantized,
            **kwargs,
        )
    return files


def upload(model: str, directory: str, username: str = HUGGINGFACE_USERNAME):
    """
    Upload to huggingface repository, make sure already login using CLI.

    Parameters
    ----------
    model: str
        it will become repository name.
    directory: str
        local directory with files in it.
    username: str, optional (default=os.environ.get('HUGGINGFACE_USERNAME', 'huseinzol05'))
    """
    try:
        create_repo(name=model)
    except Exception as e:
        logger.warning(e)

    repo_id = f'{username}/{model}'

    for file in glob(os.path.join(directory, '*')):
        file_remote = os.path.split(file)[1]
        upload_file(path_or_fileobj=file,
                    path_in_repo=file_remote,
                    repo_id=repo_id)
        logger.info(f'Uploading from {file} to {repo_id}/{file_remote}')


def upload_dict(model: str, files_mapping: Dict[str, str], username: str = HUGGINGFACE_USERNAME):
    """
    Upload to huggingface repository, make sure already login using CLI.

    Parameters
    ----------
    model: str
        it will become repository name.
    files_mapping: Dict[str, str]
        {local_file: target_file}
    username: str, optional (default=os.environ.get('HUGGINGFACE_USERNAME', 'huseinzol05'))
    """
    repo_id = f'{username}/{model}'

    try:
        create_repo(repo_id)
    except Exception as e:
        logger.warning(e)

    for file, file_remote in files_mapping.items():
        upload_file(path_or_fileobj=file,
                    path_in_repo=file_remote,
                    repo_id=repo_id)

        logger.info(f'Uploaded from {file} to {repo_id}/{file_remote}')
