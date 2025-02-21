import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt

def app():

    st.title("Tanh Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The hyperbolic tangent activation function, often abbreviated as tanh, is a widely used activation function in neural networks and machine learning models. It is a smoothed version of the sigmoid function, and it maps any real-valued number to a value between -1 and 1.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("Here's a detailed explanation of the tanh activation function:")
        st.latex(r'''\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}''')
                
        st.markdown('Where:<br><br><ul><li>Input Transformation ($x$): The input to the tanh function, denoted as $x$, can be any real number. It represents the weighted sum of input features and biases in a neural network layer before the activation. </li><br><li>Exponential Components ($e^x and\; e^{-x})$: In the formula, the input $x$ is used in two exponentiated components: $e^x$ (positive exponential) and $e^{-x}$ (negative exponential). These components emphasize the relative difference between positive and negative inputs.</li><br><li>Numerator $(e^x - e^{-x})$: The numerator of the fraction computes the difference between the positive and negative exponentials. This difference highlights the polarity of the input $x$.</li><br><li>Denominator $(e^x + e^{-x})$: The denominator of the fraction computes the sum of the positive and negative exponentials. This sum serves as a normalization factor to ensure that the output of the tanh function falls within the range of -1 to 1.</li><br><li>Final Output $(tanh(x))$: The final output of the tanh function is obtained by taking the ratio of the numerator to the denominator. The tanh function maps the input $x$ to a value between -1 and 1. When $x$ is very large (positive or negative), the output approaches 1 or -1, respectively, and for inputs near zero, the output is close to zero.</li></ul>',unsafe_allow_html=True)

        

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # Tanh Activation Function Pseudocode
                function tanh(x):
                    # Calculate the numerator components
                    positive_exponential = exp(x)
                    negative_exponential = exp(-x)
                    
                    # Calculate the numerator (e^x - e^-x)
                    numerator = positive_exponential - negative_exponential
                    
                    # Calculate the denominator (e^x + e^-x)
                    denominator = positive_exponential + negative_exponential
                    
                    # Calculate the tanh value
                    tanh_value = numerator / denominator
                    
                    # Return the tanh value
                    return tanh_value

                # Example usage:
                input_data = 2.0  # Example input value
                output = tanh(input_data)
                print("Tanh Output:", output)

                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np

            # Define a simple neural network with tanh activation
            class SimpleNet(nn.Module):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc1 = nn.Linear(10, 20)  # Fully connected layer
                    self.tanh = nn.Tanh()         # Tanh activation layer
                    self.fc2 = nn.Linear(20, 1)   # Fully connected layer (output layer)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.tanh(x)
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
        st.markdown('''<p style="font-size:17px;text-align:justify;">We define a simple neural network (SimpleNet) with two fully connected layers and a tanh activation layer. The forward method specifies the forward pass of the network, applying the tanh activation after the first fully connected layer. We create an instance of the neural network (model). We define a sample input (input_data) with shape (1, 10). We perform a forward pass to obtain the output. We define a loss function (mean squared error) and an optimizer (Stochastic Gradient Descent) for training. Note that this example does not include a full training loop, which would involve iterating over a dataset, computing gradients, and updating the model's parameters.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("Tanh Activation Function Visualization")

        @st.cache_data
        def tanh(x, a=1.0):
            return np.tanh(a * x)


        # Create a range of x values
        
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
        
            a = st.slider("Parameter 'a'", 0.1, 5.0, 1.0)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
            x = np.linspace(-5, 5, 400)
            # Compute the tanh activation function values
            tanh_values = tanh(x, a)

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, tanh_values, label=f'Tanh (a={a})', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('Tanh(ax)')
            ax.set_title('Tanh Activation Function')
            ax.legend()
            plt.grid()

            # Display the plot in Streamlit
            st.pyplot(fig)
                    
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Smoothness and Continuity: The sigmoid function is smooth and continuous. This smoothness makes it suitable for optimization algorithms that rely on gradients, such as gradient descent, as it provides well-behaved gradients throughout most of its range.")
        st.write("- Output Range: Sigmoid maps the input to an output range between 0 and 1. This property makes it useful for binary classification problems where the output can be interpreted as a probability. It's often used in the output layer of such models.")
        st.write("- Squashing Property: Sigmoid has a squashing effect, which can be advantageous when you want to compress large input values into a smaller, interpretable range. This can help in modeling probabilities or squeezing activations within certain bounds.")
        
        st.divider()

        st.subheader("Demerits")
        st.write("- Vanishing Gradient Problem: One of the major disadvantages of the sigmoid function is the vanishing gradient problem. As the input moves away from zero (either positively or negatively), the derivative of the sigmoid approaches zero. This can cause issues during training, particularly in deep networks, as the gradients become too small to update the weights effectively.")
        st.write("- Not Zero-Centered: Sigmoid is not zero-centered. The output of the sigmoid function has a mean value of approximately 0.5, which can lead to slower convergence during training, especially when used in layers where the activations should be centered around zero (e.g., in conjunction with weight initialization methods like Xavier/Glorot).")
        st.write("- Limited Output Range: While the output range of 0 to 1 is suitable for binary classification, it may not be ideal for other tasks that require more diverse output values. This can limit the expressiveness of models using sigmoid activation.")
        st.write("- Efficiency: Computing the sigmoid function is computationally more expensive than some other activation functions like ReLU (Rectified Linear Unit), which makes it less favorable in large-scale deep learning models where computational efficiency is crucial.")