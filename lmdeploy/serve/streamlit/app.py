# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from dataclasses import asdict
from typing import Literal, Optional, Union

from mmengine import Config

from lmdeploy.archs import get_task
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.model import ChatTemplateConfig


def run(model_path_or_server: str,
        server_name: str = '0.0.0.0',
        server_port: int = 6006,
        batch_size: int = 32,
        backend: Literal['turbomind', 'pytorch'] = 'turbomind',
        backend_config: Optional[Union[PytorchEngineConfig,
                                       TurbomindEngineConfig]] = None,
        chat_template_config: Optional[ChatTemplateConfig] = None,
        tp: int = 1,
        model_name: str = None,
        **kwargs):
    """chat with AI assistant through web ui.

    Args:
        model_path_or_server (str): the path of the deployed model or
        restful api URL. For example:
            - huggingface hub repo_id
            - http://0.0.0.0:23333
        server_name (str): the ip address of streamlit server
        server_port (int): the port of stream server
        batch_size (int): batch size for running Turbomind directly
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        tp (int): tensor parallel for Turbomind
    """
    if ':' in model_path_or_server:
        import lmdeploy.serve.streamlit.api_server_backend as streamlit_app
    else:
        raise NotImplementedError('There are some unpredicable errors here.')
        pipeline_type, _ = get_task(model_path_or_server)
        if pipeline_type == 'vlm':
            import lmdeploy.serve.streamlit.vl as streamlit_app
            if backend_config is not None and \
                    backend_config.session_len is None:
                backend_config.session_len = 8192
        else:
            import lmdeploy.serve.streamlit.turbomind_coupled as streamlit_app
    # dump config
    if chat_template_config is not None:
        chat_template_content = chat_template_config.to_json()
    else:
        chat_template_content = None

    config = Config(
        dict(model_path_or_server=model_path_or_server,
             server_name=server_name,
             server_port=server_port,
             batch_size=batch_size,
             backend=backend,
             backend_config_name=backend_config.__class__.__name__,
             backend_config_content=asdict(backend_config),
             chat_template_content=chat_template_content,
             tp=tp,
             model_name=model_name,
             kwargs=kwargs))
    temp_file = tempfile.NamedTemporaryFile(suffix='.py', delete=False)
    config.dump(temp_file.name)
    print('Streamlit config has been dumped to:', temp_file.name)

    # run streamlit app
    cmd = (f'streamlit run {streamlit_app.__file__} {temp_file.name} '
           f'--server.address {server_name} --server.port {server_port}')
    os.system(cmd)


if __name__ == '__main__':
    import fire
    fire.Fire(run)
