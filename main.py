import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st
from streamlit_marquee import streamlit_marquee

st.set_page_config(
    page_title = 'Activation Infopedia',
    page_icon = '💡',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)

st.sidebar.image("af.png")

streamlit_marquee(**{
    # the marquee container background color
    'background': "black",
    # the marquee text size
    'fontSize': '20px',
    # the marquee text color
    "color": "cyan",
    # the marquee text content
    'content': 'In this app you can learn about all types of activation functions, go through their formulae and codes, play with the graphs and learn to implement your own Deep Learning model',
    # the marquee container width
    'width': '1000px',
    # the marquee container line height
    'lineHeight': "35px",
    # the marquee duration
    'animationDuration': '50s',
})

st.markdown(
    """
    <style>
        /* Change tab background color */
        div[data-baseweb="tab-list"] {
            background-color: #4CAF50 !important; /* Change this to your preferred color */
        }
        
        /* Change tab text color */
        div[data-baseweb="tab"] {
            color: #000000 !important; /* Text color */
            font-weight: bold !important;
        }
        
        /* Active tab customization */
        div[data-baseweb="tab"][aria-selected="true"] {
            background-color: #388E3C !important; /* Active tab background */
            color: #0000CC !important; /* Active tab text color */
        }
    </style>
    """,
    unsafe_allow_html=True
)


from tabs import home, linear, binary_step, elu, relu, prelu, leakyrelu, gelu, selu, sigmoid, tanh, softmax, swish, appendix



Tabs = {
    "Home": home,
    "Linear": linear,
    "Binary Step": binary_step,
    "ELU": elu,
    "RELU": relu,
    "Parametric RELU": prelu,
    "Leaky RELU": leakyrelu,
    "SELU": selu,
    "GELU": gelu,
    "Sigmoid": sigmoid,
    "Tanh":tanh,
    "Softmax":softmax,
    "Swish": swish,
    "Appendix": appendix
    }


# Add title to sidear
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(Tabs.keys()))

Tabs[page].app()

st.sidebar.info("Made by : [MAINAK CHAUDHURI](https://www.linkedin.com/in/mainak-chaudhuri-127898176/)")
