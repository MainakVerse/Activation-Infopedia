# Import necessary modules
import streamlit as st

def app():
    """This function create the home page"""
    
    st.title("Activation Functions")
 

    # Add brief describtion of your web app
    st.markdown(
    """<p style="font-size:20px; text-align:justify">
           Activation functions are essential components in deep learning neural networks, serving as the nonlinear transformation that introduces complexity and nonlinearity to the network's computations. They play a crucial role in enabling neural networks to model and learn complex, nonlinear relationships within data. Various activation functions are employed, each with its unique characteristics. The widely used sigmoid and hyperbolic tangent (tanh) functions squash input values into a bounded range, making them suitable for models where the output needs to be in a specific range. However, these functions suffer from vanishing gradient problems in deep networks. Rectified Linear Unit (ReLU) has emerged as the most popular activation function due to its simplicity and effectiveness in mitigating vanishing gradient issues. Leaky ReLU and Parametric ReLU (PReLU) variants were introduced to address the problem of "dying" ReLUs. Additionally, Exponential Linear Unit (ELU) and Swish activation functions have been proposed to further improve gradient flow and model performance. Choosing the right activation function depends on the specific problem, network architecture, and careful experimentation, as each function exhibits different behaviors with varying advantages and disadvantages. Activation functions are a fundamental component of deep learning, contributing significantly to the expressiveness and learning capabilities of neural networks.
        </p>
    """, unsafe_allow_html=True)

    