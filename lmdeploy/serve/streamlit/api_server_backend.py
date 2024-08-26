# Copyright (c) OpenMMLab. All rights reserved.
import sys

import streamlit as st
from mmengine import Config

from lmdeploy.serve.openai.api_client import (get_model_list,
                                              get_streaming_response)


def chat_stream_restful(instruction: str, session_id: int, top_p: float,
                        temperature: float, request_output_len: int):
    """Chat with AI assistant.

    Args:
        instruction (str): user's prompt
        session_id (int): the session id
    """
    st.session_state.chatbot = st.session_state.chatbot + [(instruction, None)]

    for response, tokens, finish_reason in get_streaming_response(
            instruction,
            f'{st.session_state.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=request_output_len,
            interactive_mode=True,
            top_p=top_p,
            temperature=temperature):
        if finish_reason == 'length' and tokens == 0:
            st.warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
        if tokens < 0:
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


def reset_restful_func(session_id: int):
    """reset the session."""
    st.session_state.chatbot = []
    for response, tokens, finish_reason in get_streaming_response(
            '',
            f'{st.session_state.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=0,
            interactive_mode=False):
        pass


def cancel_restful_func(session_id: int):
    """cancel the session."""
    # stop the session
    for out in get_streaming_response(
            '',
            f'{st.session_state.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=0,
            cancel=True,
            interactive_mode=True):
        pass
    # end the session
    for out in get_streaming_response(
            '',
            f'{st.session_state.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=0,
            interactive_mode=False):
        pass
    # resume the session
    messages = []
    for qa in st.session_state.chatbot:
        messages.append(dict(role='user', content=qa[0]))
        if qa[1] is not None:
            messages.append(dict(role='assistant', content=qa[1]))
    for out in get_streaming_response(
            messages,
            f'{st.session_state.api_server_url}/v1/chat/interactive',
            session_id=session_id,
            request_output_len=0,
            interactive_mode=True):
        pass


def main(config_path: str):
    """chat with AI assistant through web ui.

    Args:
        config_path (str): the path of the streamlit config file.
    """
    config = Config.fromfile(config_path)
    # init model
    if 'api_server_url' not in st.session_state:
        st.session_state.api_server_url = config.model_path_or_server
    model_names = get_model_list(
        f'{st.session_state.api_server_url}/v1/models')
    model_name = ''
    if isinstance(model_names, list) and len(model_names) > 0:
        model_name = model_names[0]
    else:
        raise ValueError(
            'streamlit cannot find a suitable model from restful-api')
    # init states
    if 'session_id' not in st.session_state:
        st.session_state.session_id = 0
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = []
    # prepare ui
    st.title('LMDeploy Playground')
    with st.sidebar:
        st.text_area('Model', value=model_name)
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
            cancel_restful_func(st.session_state.session_id)
        reset_btn = st.button('Reset', use_container_width=True)
        if reset_btn:
            reset_restful_func(st.session_state.session_id)
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
            for cur_response in chat_stream_restful(
                    instruction, st.session_state.session_id, top_p,
                    temperature, request_output_len):
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)


if __name__ == '__main__':
    config_path = sys.argv[1]
    main(config_path)
