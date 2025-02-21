import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def app():

    st.title("Leaky Rectified Linear Unit Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The Leaky Rectified Linear Unit (Leaky ReLU) is an activation function used in artificial neural networks. It is an extension of the Rectified Linear Unit (ReLU) activation function and is designed to address the "dying ReLU" problem, where some neurons can become inactive during training, leading to gradients of zero and preventing weight updates. Leaky ReLU introduces a small slope for negative inputs, which allows gradients to flow and helps mitigate this issue. In brief, the Leaky Rectified Linear Unit (Leaky ReLU) is an extension of the ReLU activation function that introduces a small slope for negative inputs, allowing gradients to flow and mitigating the dying ReLU problem. It is a widely used activation function in deep learning and is known for its ability to improve training stability and prevent the vanishing gradient problem.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The ReLU activation function can be mathematically defined as follows:")
        st.latex(r'''f(x) = \begin{cases}
        x, & \text{if } x > 0 \\
        0, & \text{if } αx \leq 0
        \end{cases}''')
                
        st.markdown('Where:<br><br> $x$ is the input to the activation function.<br> α is a small positive constant (typically a small fraction like 0.01) known as the "leakage" coefficient',unsafe_allow_html=True)

        st.markdown('''Here's how the Leaky RELU activation function works:<br><ol><li> $For$ $Positive$ $Inputs$ $(x>0)$: When the input $x$ is positive, Leaky ReLU behaves like the identity function, returning the input value $x$ as is. This property allows Leaky ReLU to capture linear relationships when the input is positive, similar to the ReLU activation function.</li><li> $For$ $Non-Positive$ $Inputs$ $(x≤0)$: When the input $x$ is non-positive (i.e., zero or negative), Leaky ReLU introduces a small slope by multiplying $x$ by the leakage coefficient α. This small slope ensures that gradients can flow for negative inputs, preventing neurons from becoming completely inactive during training.As $x$ approaches negative infinity, the function approaches $αx$, and $α$ controls the rate at which this happens. As $x$ approaches zero from the left, the function approaches $α⋅0=0$.</li><li>$Hyperparameter$ $α$: The value of $α$ is typically a small positive constant, such as 0.01. While this value can be manually set, it can also be tuned during training using techniques like gradient-based optimization or as a hyperparameter during model selection.</li></ol>''',unsafe_allow_html=True)

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # Leaky ReLU Activation Function Pseudocode
                function leaky_relu(x, alpha):
                    # Initialize an empty output list
                    output_list = []
                    
                    # Element-wise operation for input x
                    for element in x:
                        if element > 0:
                            output = element
                        else:
                            output = alpha * element
                        add output to output_list
                    
                    # Return the output list
                    return output_list

                # Example usage:
                input_data = [-1, 2, 3, -4, 0]
                alpha = 0.01  # Leaky coefficient
                output = leaky_relu(input_data, alpha)
                print(output)
                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Define a simple neural network with Leaky ReLU activation
            class SimpleNet(nn.Module):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc1 = nn.Linear(10, 20)  # Fully connected layer
                    self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # Leaky ReLU activation layer
                    self.fc2 = nn.Linear(20, 1)   # Fully connected layer (output layer)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.leaky_relu(x)
                    x = self.fc2(x)
                    return x

            # Create an instance of the neural network
            model = SimpleNet()

            # Define a sample input
            input_data = torch.randn(1, 10)  # Input tensor with shape (batch_size, input_size)

            # Forward pass to obtain the output
            output = model(input_data)

            # Print the output
            print("Output:", output)

            # Define a loss function and optimizer for training
            criterion = nn.MSELoss()       # Mean squared error loss
            optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

            # Training loop (not shown in this example)
            # Typically, you would train the model on a dataset using a loop.


            ''')
   
        st.subheader('Conclusion')
        st.markdown('''<p style="font-size:17px;text-align:justify;">We define a simple neural network (SimpleNet) with two fully connected layers and a Leaky ReLU activation layer.
        The forward method specifies the forward pass of the network, applying the Leaky ReLU activation after the first fully connected layer.
        We create an instance of the neural network (model).
        We define a sample input (input_data) with shape (1, 10).
        We perform a forward pass to obtain the output.
        We define a loss function (mean squared error) and an optimizer (Stochastic Gradient Descent) for training. Note that this example does not include a full training loop, which would involve iterating over a dataset, computing gradients, and updating the model's parameters.
        In practice, you would use Leaky ReLU as an activation function within deep learning models for various tasks such as classification, regression, or other machine learning tasks.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("Leaky ReLU Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
            st.divider()
            alpha = st.slider("Alpha", 0.01, 1.0, 0.01)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
            x = np.linspace(-5, 5, 400)

            # Compute the Leaky ReLU activation function values
            leaky_relu_values = np.where(x > 0, x, alpha * x)

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, leaky_relu_values, label=f'Leaky ReLU (Alpha={alpha})', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('Leaky ReLU(x)')
            ax.set_title('Leaky ReLU Activation Function')
            plt.grid()
            ax.legend()

            # Display the plot in Streamlit
            st.pyplot(fig)

        
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Mitigating the Dying ReLU Problem: Leaky ReLU helps mitigate the dying ReLU problem, which is a limitation of the traditional Rectified Linear Unit (ReLU) activation function. In deep networks, ReLU neurons can become inactive (output zero) for certain inputs, leading to gradients of zero and preventing weight updates. Leaky ReLU introduces a small slope for negative inputs, ensuring that gradients can flow for negative values, preventing neurons from becoming completely inactive during training.")
        st.write("- No Vanishing Gradient: Unlike some other activation functions like sigmoid or tanh, Leaky ReLU does not suffer from the vanishing gradient problem. The gradient is well-defined and nonzero for both positive and negative inputs, making it suitable for deep networks.")
        st.write("- Improved Training Stability: Leaky ReLU has been observed to provide improved training stability in some cases, especially when compared to sigmoid and tanh activations, which can suffer from saturation issues.")
        st.write("- Simple to Implement: Leaky ReLU is simple to implement and computationally efficient. It introduces only one hyperparameter, the leakage coefficient (α), which is typically set to a small positive value.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Lack of Uniqueness in Slope: The choice of the leakage coefficient (α) is somewhat arbitrary, and there is no universally optimal value. While small values like 0.01 or 0.001 are commonly used, the best value depends on the specific problem and dataset. This introduces an additional hyperparameter to tune during model development.")
        st.write("- Limited Nonlinearity: Leaky ReLU is still a piecewise-linear activation function and may not capture complex nonlinear relationships as effectively as some other activation functions like sigmoid, tanh, or more advanced ones like the Parametric ReLU (PReLU).")
        st.write("- While Leaky ReLU is a valuable activation function, it may not always outperform other activations in all scenarios. The choice of activation function often depends on the specific problem, architecture, and dataset. Variants like Parametric ReLU or Exponential Linear Unit (ELU) may perform better in some cases.")