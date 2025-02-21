import streamlit as st
import matplotlib.pyplot as plt
import numpy as np



def app():
    # Streamlit App Title
    st.subheader("Activation Function Visualization")
    # Function Definitions
    with st.expander("Want to view all the activation function graphs at once?"):
        def linear(x):
            return x

        def binary_step(x):
            return np.where(x >= 0, 1, 0)

        def elu(x, alpha=1.0):
            return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

        def relu(x):
            return np.maximum(0, x)

        def leaky_relu(x, alpha=0.01):
            return np.where(x >= 0, x, alpha * x)

        def parametric_relu(x, alpha):
            return np.where(x >= 0, x, alpha * x)

        def selu(x, alpha=1.67326, scale=1.0507):
            return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))

        def gelu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def tanh(x):
            return np.tanh(x)

        def softmax(x):
            exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
            return exp_x / exp_x.sum(axis=0, keepdims=True)

        def swish(x, beta=1.0):
            return x * sigmoid(beta * x)

        # Range of x values
        x = np.linspace(-5, 5, 400)

        # Compute activation function values
        linear_values = linear(x)
        binary_step_values = binary_step(x)
        elu_values = elu(x)
        relu_values = relu(x)
        leaky_relu_values = leaky_relu(x)
        parametric_relu_values = parametric_relu(x, 0.01)
        selu_values = selu(x)
        gelu_values = gelu(x)
        sigmoid_values = sigmoid(x)
        tanh_values = tanh(x)
        softmax_values = softmax(x)
        swish_values = swish(x, 1.0)

        # Create subplots
        fig, axs = plt.subplots(4, 3, figsize=(12, 12))

        # Plot activation functions
        axs[0, 0].plot(x, linear_values)
        axs[0, 0].set_title('Linear')
        axs[0, 1].plot(x, binary_step_values)
        axs[0, 1].set_title('Binary Step')
        axs[0, 2].plot(x, elu_values)
        axs[0, 2].set_title('ELU')

        axs[1, 0].plot(x, relu_values)
        axs[1, 0].set_title('ReLU')
        axs[1, 1].plot(x, leaky_relu_values)
        axs[1, 1].set_title('Leaky ReLU')
        axs[1, 2].plot(x, parametric_relu_values)
        axs[1, 2].set_title('Parametric ReLU')

        axs[2, 0].plot(x, selu_values)
        axs[2, 0].set_title('SELU')
        axs[2, 1].plot(x, gelu_values)
        axs[2, 1].set_title('GELU')
        axs[2, 2].plot(x, sigmoid_values)
        axs[2, 2].set_title('Sigmoid')

        axs[3, 0].plot(x, tanh_values)
        axs[3, 0].set_title('Tanh')
        axs[3, 1].plot(x, softmax_values)
        axs[3, 1].set_title('Softmax')
        axs[3, 2].plot(x, swish_values)
        axs[3, 2].set_title('Swish')

        # Adjust layout
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)

    with st.expander("Want to test graphs with equation on your own?"):
       

        # Desmos embed HTML code
        desmos_html = """
        <iframe
        src="https://www.desmos.com/calculator"
        width="1000px"
        height="600px"
        style="border: 1px solid #ccc"
        frameborder="0"
        scrolling="no"
        ></iframe>
        """

        # Use the st.components function to embed the Desmos iframe
        st.components.v1.html(desmos_html, width=1000, height=600)

    
    st.success("Special thanks and acknowledgement to (graphics partner):")
    st.image("https://preview.redd.it/8l1kzoyzqyd61.png?width=7652&format=png&auto=webp&s=dd45ed1fbf251fa60475170d68b2fa9c1178762b")