import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt

def app():

    st.title("Scaled Exponential Linear Unit Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The Scaled Exponential Linear Unit (SELU) is an activation function used in artificial neural networks. It is designed to address vanishing gradient problems and improve the performance of deep neural networks. The SELU activation function extends the Exponential Linear Unit (ELU) activation function by introducing a scale parameter to stabilize the activations. In brief, the SELU activation function has distinct advantages, especially in terms of self-normalization and improved training stability in deep networks. However, its practical benefits can vary depending on the specific problem, network architecture, and proper initialization. Researchers and practitioners often experiment with different activation functions to determine the most suitable one for their use case.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The SELU activation function can be defined as follows:")
        st.latex(r'''f(x) = \lambda \cdot \begin{cases}
        x & \text{if } x > 0 \\
        \alpha \cdot (e^x - 1) & \text{if } x \leq 0
        \end{cases}''')
                
        st.markdown('Where:<br><br> $x$ is the input to the activation function.<br> λ is a scale parameter (positive constant).<br>α is a scaling parameter (positive constant), typically set to 1.67326 for improved performance in deep networks.',unsafe_allow_html=True)

        st.markdown('''Here's how the Leaky RELU activation function works:<br><ol><li> $For$ $Positive$ $Inputs$ $(x>0)$: When the input $x$ is positive, SELU behaves like the identity function, returning the input value $x$. This property allows SELU to capture linear relationships when the input is positive, similar to the ReLU activation function.</li><li> $For$ $Non-Positive$ $Inputs$ $(x≤0)$: When the input $x$ is non-positive (i.e., zero or negative), SELU introduces a scaled exponential component. It computes the output as $α⋅(e^x −1)$, where α and λ are positive constants. This scaling introduces nonlinearity for negative inputs while preserving the exponential growth rate. The choice of α ensures that the activations maintain zero mean and unit variance under certain conditions. As x approaches negative infinity, the function approaches $−α$ with a rate controlled by $λ$. As $x$ approaches zero from the left, the function approaches $α⋅(e^0 −1)$=$α⋅0$=$0$.</li><li>$Hyperparameter$ $α$ $and$ $λ$: The values of $α$ and $λ$ are typically set to constants, with $α$ around 1.67326 and $λ$ around 1.0507. These values have been shown to work well in practice for deep networks, but they can be fine-tuned based on the specific problem and architecture.</li></ol>''',unsafe_allow_html=True)

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # SELU Activation Function Pseudocode
                function selu(x, alpha, lambda):
                    # Initialize an empty output list
                    output_list = []

                    # Element-wise operation for input x
                    for element in x:
                        if element > 0:
                            output = lambda * element
                        else:
                            output = lambda * alpha * (e^element - 1)
                        add output to output_list
                    
                    # Return the output list
                    return output_list

                # Example usage:
                input_data = [-1, 2, 3, -4, 0]
                alpha = 1.67326  # Scaling parameter (typically constant)
                lambda = 1.0507  # Scale parameter (typically constant)
                output = selu(input_data, alpha, lambda)
                print(output)

                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np

            # Define a simple neural network with SELU activation
            class SimpleNet(nn.Module):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc1 = nn.Linear(10, 20)  # Fully connected layer
                    self.selu = nn.SELU()         # SELU activation layer
                    self.fc2 = nn.Linear(20, 1)   # Fully connected layer (output layer)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.selu(x)
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
        st.markdown('''<p style="font-size:17px;text-align:justify;">We define a simple neural network (SimpleNet) with two fully connected layers and a SELU activation layer.
        The forward method specifies the forward pass of the network, applying the SELU activation after the first fully connected layer.
        We create an instance of the neural network (model).
        We define a sample input (input_data) with shape (1, 10).
        We perform a forward pass to obtain the output.
        We define a loss function (mean squared error) and an optimizer (Stochastic Gradient Descent) for training. Note that this example does not include a full training loop, which would involve iterating over a dataset, computing gradients, and updating the model's parameters.
        In practice, you can use SELU as an activation function within deep learning models for various tasks such as classification, regression, or other machine learning tasks.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("SELU Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
            st.divider()
            alpha = st.slider("Alpha", 0.01, 5.0, 1.67326)
            lambda_ = st.slider("Lambda", 0.01, 5.0, 1.0507)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
            x = np.linspace(-5, 5, 400)

            # Compute the SELU activation function values
            selu_values = np.where(x > 0, lambda_ * x, lambda_ * alpha * (np.exp(x) - 1))

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, selu_values, label=f'SELU (Alpha={alpha}, Lambda={lambda_})', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('SELU(x)')
            ax.set_ylim(-10,10)
            ax.set_title('SELU Activation Function')
            ax.legend()

            # Display the plot in Streamlit
            st.pyplot(fig)
        
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Self-Normalization: One of the key advantages of SELU is that it possesses a self-normalizing property. In deep networks, it can help stabilize the activations and gradients during training, reducing the likelihood of vanishing and exploding gradients. This self-normalization property can lead to faster convergence and improved training performance.")
        st.write("- Zero Mean and Unit Variance: Under certain conditions, the SELU activation function can ensure that the activations have a mean close to zero and a standard deviation close to one during both forward and backward passes. This is particularly useful for training deep networks as it mitigates issues related to internal covariate shift.")
        st.write("- Improved Gradient Flow: SELU introduces smooth non-linearities for both positive and negative inputs. This enables a more robust gradient flow throughout the network, which can lead to improved optimization and reduced sensitivity to weight initialization.")
        st.write("- No Need for Batch Normalization: In some cases, SELU can eliminate the need for batch normalization layers, which are commonly used to stabilize activations in deep networks. This can simplify the network architecture and reduce computational overhead.")
        st.write("- Empirical Performance: SELU has demonstrated improved training performance and generalization in certain deep learning tasks, especially when the self-normalization property is leveraged effectively.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Strict Assumptions: The self-normalizing property of SELU relies on specific assumptions, such as weights being initialized with a particular method (LeCun initialization) and feedforward networks without recurrent connections. These assumptions may not always hold in practice.")
        st.write("- Sensitivity to Initialization: Although SELU is designed to work well with specific weight initialization, if the network weights are not initialized properly, it may not exhibit the desired self-normalizing behavior. This makes weight initialization a critical consideration when using SELU.")
        st.write("- Limited Applications: SELU has shown promise in feedforward neural networks, particularly for deep architectures. However, its applicability to various types of neural networks, such as convolutional or recurrent networks, is still an active area of research.")
        st.write("- Computational Cost: While SELU does not require batch normalization in some cases, it may not always outperform other activation functions like ReLU with batch normalization in terms of computational efficiency.")