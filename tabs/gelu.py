import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt

def app():

    st.title("Gaussian Error Linear Unit Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The Gaussian Error Linear Unit (GELU) activation function is a continuous and differentiable approximation of the rectifier linear unit (ReLU) activation function. It has gained popularity in deep learning for its smoothness and better handling of vanishing gradients compared to ReLU. In practice, you can use the GELU activation function within deep learning models for various tasks such as classification, regression, or other machine learning tasks. GELU is particularly popular in natural language processing (NLP) tasks and has been used in transformer architectures like BERT and GPT-2. In brief, the GELU activation function offers advantages in terms of smoothness, efficient computation, and effective modeling of both linear and nonlinear relationships. However, its performance can vary across different scenarios, and proper hyperparameter tuning may be required to leverage its full potential. It is a valuable option to consider in deep learning, particularly in natural language processing and computer vision tasks.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The GELU activation function can be defined as follows:")
        st.latex(r'''GELU(x) = \frac{1}{2}x\left(1 + \tanh\left(\sqrt{\frac{\pi}{2}}\left(x + 0.044715x^3\right)\right)\right)''')
                
        st.markdown('Where:<br><br><ul><li>$Transformation$ $of$ $Input$ $(x)$: The input $x$ is first transformed using a weighted combination of the input and a cubic polynomial function: $x+0.044715x^3$. This transformation introduces non-linearity and smoothness.</li><br><li>$Scaling$ $Factor$ : The transformed input is scaled by a factor of $ \sqrt{2/π}$, which helps control the slope of the activation function.</li><br><li>$Hyperbolic$ $Tangent$ $(tanh)$: The scaled, transformed input is then passed through the hyperbolic tangent $(tanh)$ function. The $tanh$ function introduces saturation and smoothness to the activation, making it differentiable and preventing the gradient from becoming zero for large inputs.</li><li><br>$Rescaling$ $and$ $Shifting$: After the tanh operation, the result is scaled by $1/2$ and multiplied by the original input $x$. This rescaling and shifting ensure that the GELU activation function has zero mean and unit variance for zero-centered inputs, which can be beneficial for training deep networks.</li></ul>',unsafe_allow_html=True)

        

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # GELU Activation Function Pseudocode
                function gelu(x):
                    # Define constants
                    alpha = 0.79788  # Constant for GELU approximation
                    sqrt_two_over_pi = 0.79788  # Constant for GELU approximation
                    
                    # Compute the cubic polynomial component
                    cubic_term = 0.044715 * x^3
                    
                    # Compute the exponential component
                    exp_term = exp(-x^2 / 2)
                    
                    # Compute the GELU function
                    gelu_value = 0.5 * x * (1 + tanh(sqrt_two_over_pi * (x + alpha * cubic_term))) * exp_term
                    
                    # Return the GELU value
                    return gelu_value

                # Example usage:
                input_data = 2.0  # Example input value
                output = gelu(input_data)
                print("GELU Output:", output)

                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np

            # Define a simple neural network with GELU activation
            class SimpleNet(nn.Module):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc1 = nn.Linear(10, 20)  # Fully connected layer
                    self.gelu = nn.GELU()         # GELU activation layer
                    self.fc2 = nn.Linear(20, 1)   # Fully connected layer (output layer)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.gelu(x)
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
        st.markdown('''<p style="font-size:17px;text-align:justify;">We define a simple neural network (SimpleNet) with two fully connected layers and a GELU activation layer.
        The forward method specifies the forward pass of the network, applying the GELU activation after the first fully connected layer.
        We create an instance of the neural network (model).
        We define a sample input (input_data) with shape (1, 10).
        We perform a forward pass to obtain the output.
        We define a loss function (mean squared error) and an optimizer (Stochastic Gradient Descent) for training. Note that this example does not include a full training loop, which would involve iterating over a dataset, computing gradients, and updating the model's parameters.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("GELU Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
        
            alpha = st.slider("Alpha", 0.01, 2.0, 0.79788)
            sqrt_two_over_pi = st.slider("Sqrt(2/π)", 0.01, 2.0, 0.79788)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
            # Create a range of x values
            x = np.linspace(-5, 5, 400)

            # Compute the GELU activation function values
            gelu_values = 0.5 * x * (1 + np.tanh(sqrt_two_over_pi * (x + alpha * (x ** 3))))

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, gelu_values, label=f'GELU (Alpha={alpha}, Sqrt(2/π)={sqrt_two_over_pi})', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('GELU(x)')
            ax.set_title('GELU Activation Function')
            ax.legend()

            # Display the plot in Streamlit
            st.pyplot(fig)
                    
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Smoothness and Continuity: GELU is a smooth and continuous activation function. This smoothness is beneficial for optimization because it ensures that gradients are well-behaved and avoids the zero gradient problem associated with some other activation functions, such as the standard ReLU.")
        st.write("- Approximation of Identity for Positive Inputs: For positive inputs, GELU behaves like the identity function, allowing it to capture linear relationships and preserve positive activations, similar to the behavior of ReLU.")
        st.write("- Nonlinear Behavior for Negative Inputs: For negative inputs, GELU introduces a nonlinear component through the hyperbolic tangent $(tanh)$ operation. This helps in modeling complex, nonlinear relationships and enables the network to learn more expressive representations.")
        st.write("- Efficient Computational Properties: GELU can be computed efficiently and is differentiable, making it suitable for deep learning frameworks and architectures.")
        st.write("- Effective for Deep Networks: GELU has been observed to perform well in deep neural networks, including transformer-based architectures used in natural language processing (NLP) tasks like BERT and GPT.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Complexity: While GELU is computationally efficient compared to some other activation functions like sigmoid or tanh, it is slightly more complex than the standard ReLU and may have a higher computational cost.")
        st.write("- Empirical Performance Variability: The performance of GELU can vary depending on the specific problem, architecture, and initialization. It may not always outperform other activation functions in all scenarios.")
        st.write("- Hyperparameter Tuning: GELU introduces additional hyperparameters, such as the alpha and sqrt(2/π) constants. Tuning these hyperparameters for optimal performance can be a challenge.")
        st.write("- Not Universally Applicable: While GELU has shown advantages in certain deep learning tasks, it may not be the best choice for all types of problems or network architectures. Its effectiveness can depend on the nature of the data and the specifics of the task.")