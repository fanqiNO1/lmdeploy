# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import random
import sys

import streamlit as st
from mmengine import Config

from lmdeploy.messages import (GenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.async_engine import AsyncEngine


@st.cache_resource
def create_async_engine(_config: Config):
    # tell streamlit not to hash this argument by adding a leading underscore
    # prepare backend and chat template
    if _config.backend_config_name == 'TurbomindEngineConfig':
        backend_config = TurbomindEngineConfig(
            **_config.backend_config_content)
    elif _config.backend_config_name == 'PytorchEngineConfig':
        backend_config = PytorchEngineConfig(**_config.backend_config_content)
    if _config.chat_template_content is not None:
        chat_template_config = ChatTemplateConfig.from_json(
            _config.chat_template_content)
    else:
        chat_template_config = None
    # create async engine
    async_engine = AsyncEngine(model_path=_config.model_path_or_server,
                               backend=_config.backend,
                               backend_config=backend_config,
                               chat_template_config=chat_template_config,
                               model_name=_config.model_name,
                               tp=_config.tp,
                               **_config.kwargs)
    return async_engine


async def chat_stream_local(instruction: str, session_id: int, top_p: float,
                            temperature: float, request_output_len: int):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        session_id (int): the session id
    """
    st.session_state.chatbot = st.session_state.chatbot + [(instruction, None)]
    gen_config = GenerationConfig(max_new_tokens=request_output_len,
                                  top_p=top_p,
                                  top_k=40,
                                  temperature=temperature,
                                  random_seed=random.getrandbits(64) if len(
                                      st.session_state.chatbot) == 1 else None)
    async for outputs in st.session_state.async_engine.generate(
            instruction,
            session_id,
            gen_config=gen_config,
            stream_response=True,
            sequence_start=(len(st.session_state.chatbot) == 1),
            sequence_end=False):
        response = outputs.response
        if outputs.finish_reason == 'length' and \
                outputs.generate_token_len == 0:
            st.warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
        if outputs.generate_token_len < 0:
            st.warning('WARNING: running on the old session.'
                       ' Please restart the session by reset button.')
        if st.session_state.chatbot[-1][-1] is None:
            st.session_state.chatbot[-1] = (st.session_state.chatbot[-1][0],
                                            response)
        else:
            st.session_state.chatbot[-1] = (st.session_state.chatbot[-1][0],
                                            st.session_state.chatbot[-1][1] +
                                            response)
        yield st.session_state.chatbot[-1][1]


async def reset_local_func(session_id: int):
    """reset the session."""
    st.session_state.chatbot = []
    await st.session_state.async_engine.end_session(session_id)


async def cancel_local_func(session_id: int):
    """cancel the session."""
    await st.session_state.async_engine.stop_session(session_id)
    if st.session_state.async_engine.backend == 'pytorch':
        pass
    else:
        await st.session_state.async_engine.end_session(session_id)
        messages = []
        for qa in st.session_state.chatbot:
            messages.append(dict(role='user', content=qa[0]))
            if qa[1] is not None:
                messages.append(dict(role='assistant', content=qa[1]))
        gen_config = GenerationConfig(max_new_tokens=0)
        async for out in st.session_state.async_engine.generate(
                messages,
                session_id,
                gen_config=gen_config,
                stream_response=True,
                sequence_start=True,
                sequence_end=False):
            pass


async def main(config_path: str):
    """chat with AI assistant through web ui.

    Args:
        config_path (str): the path of the streamlit config file.
    """
    config = Config.fromfile(config_path)
    # init model
    if 'async_engine' not in st.session_state:
        st.session_state.async_engine = create_async_engine(config)
    # init states
    if 'session_id' not in st.session_state:
        st.session_state.session_id = 0
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = []
    # prepare ui
    st.title('LMDeploy Playground')
    with st.sidebar:
        st.text_area('Model', value=config.model_path_or_server)
        request_output_len = st.slider('Maximum new tokens',
                                       min_value=1,
                                       max_value=2048,
                                       value=512,
                                       step=1)
        top_p = st.slider('Top p',
                          min_value=0.01,
                          max_value=1.0,
                          value=0.8,
                          step=0.01)
        temperature = st.slider('Temperature',
                                min_value=0.01,
                                max_value=1.5,
                                value=0.7,
                                step=0.01)
        cancel_btn = st.button('Cancel', use_container_width=True)
        if cancel_btn:
            await cancel_local_func(st.session_state.session_id)
        reset_btn = st.button('Reset', use_container_width=True)
        if reset_btn:
            await reset_local_func(st.session_state.session_id)
    # display chatbot on app rerun
    for message in st.session_state.chatbot:
        with st.chat_message('user'):
            st.markdown(message[0])
        if message[1] is not None:
            with st.chat_message('assistant'):
                st.markdown(message[1])
    # chat with AI assistant
    if instruction := st.chat_input('Hello'):
        with st.chat_message('user'):
            st.markdown(instruction)
        # generate response
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            async for cur_response in chat_stream_local(
                    instruction, st.session_state.session_id, top_p,
                    temperature, request_output_len):
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)


if __name__ == '__main__':
    config_path = sys.argv[1]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(main(config_path))
    loop.run_forever()
