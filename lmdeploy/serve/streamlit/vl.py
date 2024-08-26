# Copyright (c) OpenMMLab. All rights reserved.
import sys
import time
from io import BytesIO

import streamlit as st
from mmengine import Config
from PIL import Image

from lmdeploy.messages import (EngineGenerationConfig, PytorchEngineConfig,
                               TurbomindEngineConfig)
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.pytorch.engine.request import _run_until_complete
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from lmdeploy.tokenizer import DetokenizeState
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


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
    async_engine = VLAsyncEngine(model_path=_config.model_path_or_server,
                                 backend=_config.backend,
                                 backend_config=backend_config,
                                 chat_template_config=chat_template_config,
                                 model_name=_config.model_name,
                                 tp=_config.tp,
                                 **_config.kwargs)
    return async_engine


def add_image(upload_image):
    """Append image to query."""
    image_bytes = upload_image.read()
    st.session_state.chatbot = st.session_state.chatbot + [(
        (image_bytes, ), None)]
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    history = st.session_state.message
    if len(history) == 0 or history[-1][-1] is not None:
        st.session_state.message.append([[image], None])
    else:
        st.session_state.message[-1][0].append(image)


def add_text(text):
    """User query."""
    st.session_state.chatbot = st.session_state.chatbot + [(text, None)]
    history = st.session_state.message
    if len(history) == 0 or history[-1][-1] is not None:
        st.session_state.message.append([text, None])
    else:
        st.session_state.message[-1][0].insert(0, text)


def chat(session_id, max_new_tokens, top_p, top_k, temperature):
    engine = st.session_state.async_engine
    generator = engine.engine.create_instance()
    history = st.session_state.message
    sequence_start = len(history) == 0

    if isinstance(history[-1][0], str):
        prompt = history[-1][0]
    else:
        prompt = history[-1][0][0]
        images = history[-1][0][1:]
        prompt = (prompt, images)

    logger.info('prompt: ' + str(prompt))
    prompt = engine.vl_prompt_template.prompt_to_messages(prompt)
    t0 = time.perf_counter()
    inputs = _run_until_complete(
        engine._get_prompt_input(prompt, True, sequence_start, ''))
    t1 = time.perf_counter()
    logger.info('preprocess cost %.3fs' % (t1 - t0))

    input_ids = inputs['input_ids']
    logger.info('input_ids: ' + str(input_ids))
    if len(input_ids
           ) + st.session_state._step + max_new_tokens > engine.session_len:
        st.warning('WARNING: exceed session max length.'
                   ' Please restart the session by reset button.')
    else:
        gen_config = EngineGenerationConfig(max_new_tokens=max_new_tokens,
                                            top_p=top_p,
                                            top_k=top_k,
                                            temperature=temperature,
                                            stop_words=engine.stop_words)
        step = st.session_state._step
        state = DetokenizeState(len(input_ids))
        for outputs in generator.stream_infer(session_id=session_id,
                                              **inputs,
                                              sequence_start=sequence_start,
                                              step=step,
                                              gen_config=gen_config,
                                              stream_output=True):
            res, tokens = input_ids + outputs.token_ids, outputs.num_token
            response, state = engine.tokenizer.detokenize_incrementally(
                res, state, skip_special_tokens=gen_config.skip_special_tokens)
            if st.session_state.chatbot[-1][1] is None:
                st.session_state.chatbot[-1] = (
                    st.session_state.chatbot[-1][0], response)
                st.session_state.message[-1][1] = response
            else:
                st.session_state.chatbot[-1] = (
                    st.session_state.chatbot[-1][0],
                    st.session_state.chatbot[-1][1] + response)
                st.session_state.message[-1][1] += response
            st.session_state._step = step + len(input_ids) + tokens
            yield st.session_state.chatbot[-1][1]


def stop(session_id):
    generator = st.session_state.async_engine.engine.create_instance()
    for _ in generator.stream_infer(session_id=session_id,
                                    input_ids=[0],
                                    request_output_len=0,
                                    sequence_start=False,
                                    sequence_end=False,
                                    stop=True):
        pass


def cancel(session_id):
    stop(session_id)


def reset(session_id):
    """Reset a new session."""
    stop(session_id)
    st.session_state.chatbot = []
    st.session_state.message = []
    st.session_state._step = 0


def main(config_path: str):
    """Chat with AI assistant."""
    config = Config.fromfile(config_path)
    # init model
    if 'async_engine' not in st.session_state:
        st.session_state.async_engine = create_async_engine(config)
    # init states
    if 'session_id' not in st.session_state:
        st.session_state.session_id = 0
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = []
    if 'message' not in st.session_state:
        st.session_state.message = []
    if '_step' not in st.session_state:
        st.session_state._step = 0
    # prepare ui
    st.title('LMDeploy Playground')
    with st.sidebar:
        st.text_area('Model', value=config.model_path_or_server)
        max_new_tokens = st.slider('Maximum new tokens',
                                   min_value=1,
                                   max_value=2048,
                                   value=512,
                                   step=1)
        top_p = st.slider('Top p',
                          min_value=0.01,
                          max_value=1.0,
                          value=0.8,
                          step=0.01)
        top_k = st.slider('Top k',
                          min_value=1,
                          max_value=100,
                          value=50,
                          step=1)
        temperature = st.slider('Temperature',
                                min_value=0.01,
                                max_value=1.5,
                                value=0.7,
                                step=0.01)
        uploaded_image = st.file_uploader('Upload Image',
                                          type=['png', 'jpg', 'jpeg'])
        if uploaded_image is not None:
            add_image(uploaded_image)
        cancel_btn = st.button('Cancel', use_container_width=True)
        if cancel_btn:
            cancel(st.session_state.session_id)
        reset_btn = st.button('Reset', use_container_width=True)
        if reset_btn:
            reset(st.session_state.session_id)
    # display chatbot on app rerun
    for message in st.session_state.chatbot:
        if isinstance(message[0], str):
            with st.chat_message('user'):
                st.markdown(message[0])
        else:
            for img in message[0]:
                st.image(img, caption='uploaded image')
        if message[1] is not None:
            with st.chat_message('assistant'):
                st.markdown(message[1])
    # chat with AI assistant
    if instruction := st.chat_input('Hello'):
        with st.chat_message('user'):
            st.markdown(instruction)
        add_text(instruction)
        # generate response
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            for cur_response in chat(st.session_state.session_id,
                                     max_new_tokens, top_p, top_k,
                                     temperature):
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)


if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)
