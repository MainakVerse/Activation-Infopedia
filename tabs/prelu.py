import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def app():

    st.title("Parametric Rectified Linear Unit Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The Parametric Rectified Linear Unit (PReLU) is an activation function used in artificial neural networks. It is an extension of the Rectified Linear Unit (ReLU) activation function and is designed to address some of the limitations of ReLU by introducing a learnable parameter. PReLU allows the network to adaptively learn the slope of the activation function during training, which can be beneficial in various scenarios.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The ReLU activation function can be mathematically defined as follows:")
        st.latex(r'''f(x) = \begin{cases}
        x, & \text{if } x > 0 \\
        0, & \text{if } α⋅x \leq 0
        \end{cases}''')
                
        st.markdown('Where:<br><br> $f(x)$ is the activation function.<br> α is a learnable parameter (scalar or channel-wise) that controls the slope of the function for negative values of $x$',unsafe_allow_html=True)

        st.markdown('''Here's how the RELU activation function works:<br><ol><li> $For$ $Positive$ $Inputs$ $(x>0)$: When the input $x$ is positive, PReLU behaves like the identity function, returning the input value $x$ as is. This property allows PReLU to capture linear relationships when the input is positive, similar to the ReLU activation function.</li><li> $For$ $Non-Positive$ $Inputs$ $(x≤0)$: When the input $x$ is non-positive (i.e., zero or negative), PReLU introduces a nonlinear component. It computes the output as a scaled version of $x$, where the scaling factor α is learned during training. The introduction of this learnable parameter allows PReLU to adapt the slope of the activation function for negative inputs, addressing the limitation of the fixed slope in traditional ReLU.</li><li>$Learnable$ $Parameter$ α: The key feature of PReLU is the learnable parameter α, which is unique to each neuron (or channel, in the case of convolutional neural networks). During training, the network learns the optimal α values alongside other parameters through gradient descent or similar optimization techniques. This adaptability allows PReLU to adjust the activation function's behavior for different neurons or channels, enhancing the model's expressiveness.</li></ol>''',unsafe_allow_html=True)

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # PReLU Activation Function Pseudocode
                function prelu(x, alpha):
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
                alpha = 0.1  # Learnable parameter
                output = prelu(input_data, alpha)
                print(output)
                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Define a simple neural network with PReLU activation
            class SimpleNet(nn.Module):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc1 = nn.Linear(10, 20)  # Fully connected layer
                    self.prelu = nn.PReLU()       # PReLU activation layer
                    self.fc2 = nn.Linear(20, 1)   # Fully connected layer (output layer)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.prelu(x)
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
        st.markdown('''<p style="font-size:17px;text-align:justify;">We define a simple neural network (SimpleNet) with two fully connected layers and a PReLU activation layer.
        The forward method specifies the forward pass of the network, applying the PReLU activation after the first fully connected layer.
        We create an instance of the neural network (model).
        We define a sample input (input_data) with shape (1, 10).
        We perform a forward pass to obtain the output.
        We define a loss function (mean squared error) and an optimizer (Stochastic Gradient Descent) for training. Note that this example does not include a full training loop, which would involve iterating over a dataset, computing gradients, and updating the model's parameters.
        In practice, you would use PReLU and other activation functions within deep learning models for tasks such as classification, regression, or other machine learning tasks.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("Parametric ReLU (PReLU) Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
            st.divider()
            alpha = st.slider("Alpha (\u03B1)", 0.01, 5.0, 1.0)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
            x = np.linspace(-5, 5, 400)

            # Compute the PReLU activation function values
            prelu_values = np.where(x > 0, x, alpha * x)

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, prelu_values, label=f'PReLU(\u03B1={alpha})', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('PReLU(x)')
            ax.set_title('PReLU Activation Function')
            ax.legend()
            plt.grid()
            # Display the plot in Streamlit
            st.pyplot(fig)

        
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Mitigating the Dying ReLU Problem: PReLU helps alleviate the dying ReLU problem, which is a limitation of the traditional Rectified Linear Unit (ReLU) activation function. In deep networks, ReLU neurons can become inactive (output zero) for certain inputs, leading to dead neurons that don't update their weights. PReLU, by allowing a non-zero slope for negative inputs, ensures that gradients can flow backward even for negative values, preventing neurons from becoming inactive during training.")
        st.write("- Learnable Slope: One of the significant advantages of PReLU is that it introduces a learnable parameter (α) for each neuron or channel. During training, the network learns the optimal α values alongside other parameters through gradient descent or similar optimization techniques. This adaptability allows PReLU to adjust the activation function's behavior for different neurons or channels, enhancing the model's expressiveness.")
        st.write("- Robustness to Noisy Inputs: PReLU is robust to noisy inputs because it can adapt its slope for negative inputs. This adaptability can help the network handle noisy or outlier data more effectively compared to activations like ReLU, which can be sensitive to extreme values.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Computational Complexity: Compared to traditional ReLU, PReLU is computationally more complex because it introduces an additional learnable parameter (α) for each neuron or channel. This increased complexity can le")
        st.write("- Risk of Overfitting: The learnable parameter α in PReLU introduces additional model complexity, and if not regularized properly, it may lead to overfitting, especially when the dataset is small or noisy. Careful hyperparameter tuning and regularization techniques may be necessary to prevent overfitting.")
        st.write("- Lack of Convexity: PReLU is not a convex activation function, which can make the optimization landscape more complex. This non-convexity might lead to challenges in finding the global minimum during training, but in practice, it often works well due to the effectiveness of stochastic gradient-based optimization methods.")
       