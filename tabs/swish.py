import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt

def app():

    st.title("Swish Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The Swish activation function is a relatively recent activation function introduced by researchers at Google in 2017. It has gained attention due to its potential advantages over traditional activation functions like ReLU (Rectified Linear Unit). The Swish activation function has been explored in various deep learning architectures, and it has shown promising results in improving training convergence and model performance in certain tasks. However, its adoption is not as widespread as ReLU and its variants, which are still the most commonly used activation functions. In practice, the choice of activation function depends on the specific problem, the architecture of the neural network, and empirical experimentation to determine which activation function works best for a given task. In brief, the Swish activation function combines the properties of the sigmoid function and adaptability through the scaling factor $β$, which can lead to improved training and convergence characteristics in certain neural network architectures.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("Here's a detailed explanation of the Swish activation function:")
        st.latex(r'''\text{Swish}(x) = x \cdot \sigma(\beta x)''')
                
        st.markdown('Where:<br><br><ul><li>$Linear$ $Transformation$ ($x⋅σ(βx)$): The Swish function starts with a linear transformation of the input $x$ by multiplying it with the output of the sigmoid function $σ(βx)$. The sigmoid function squashes the input $x$ into the range [0, 1], introducing non-linearity.</li><br><li>$Scaling$ $Factor$ ($β$): The scaling factor $β$ controls how quickly the output saturates (becomes very close to 0 or 1) as the input moves away from 0. A larger $β$ results in a steeper sigmoid curve, making the function closer to a linear transformation for small input values.</li></ul><br>',unsafe_allow_html=True)

        st.markdown('''<b>Properties:</b><br><li>$Smoothness$: The Swish activation function is smooth and differentiable, which is advantageous for gradient-based optimization algorithms used in training neural networks. Its smoothness helps mitigate some of the vanishing gradient problems encountered with ReLU.</li><li>$Zero-Centeredness$: Unlike ReLU, Swish is approximately zero-centered. This characteristic can help with optimization and training stability, as it avoids issues related to the mean-shift of activations.</li><li>$Adaptive$ $Activation$: The Swish function introduces an adaptiveness to the activation, controlled by the scaling factor $β$. It can make the function closer to ReLU-like behavior when $β$ is small and introduce saturation when $β$ is large.</li>''',unsafe_allow_html=True)

        

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # Swish Activation Function Pseudocode
                function swish(x, beta):
                    # Apply the sigmoid activation function to beta times the input 'x'
                    sigmoid_beta_x = sigmoid(beta * x)
                    
                    # Compute the Swish output as the product of 'x' and 'sigmoid_beta_x'
                    swish_output = x * sigmoid_beta_x
                    
                    # Return the Swish output
                    return swish_output

                # Sigmoid Activation Function
                function sigmoid(x):
                    return 1 / (1 + exp(-x))

                # Example usage:
                input_data = 2.0  # Example input value
                beta = 1.0       # Swish scaling factor (can be learned during training)
                output = swish(input_data, beta)
                print("Swish Output:", output)

                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import tensorflow as tf
            from tensorflow.keras.layers import Input, Dense
            from tensorflow.keras.models import Model
            from tensorflow.keras.optimizers import Adam
            import numpy as np

            # Define the Swish activation function
            def swish(x):
                return x * tf.sigmoid(x)

            # Generate random data for a toy regression problem
            np.random.seed(0)
            X = np.random.rand(1000, 1)
            y = 2 * X + 1 + 0.1 * np.random.randn(1000, 1)

            # Create a simple neural network with Swish activation
            input_layer = Input(shape=(1,))
            hidden_layer = Dense(32, activation=swish)(input_layer)
            output_layer = Dense(1)(hidden_layer)

            # Build the model
            model = Model(inputs=input_layer, outputs=output_layer)

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

            # Train the model
            model.fit(X, y, epochs=50, batch_size=32)

            # Evaluate the model
            loss = model.evaluate(X, y)
            print("Final Loss:", loss)

            ''')
   
        st.subheader('Conclusion')
        st.markdown('''<p style="font-size:17px;text-align:justify;">We define the Swish activation function as swish(x) = x * sigmoid(x) using TensorFlow operations. We generate random data for a toy regression problem. X is the input, and y is the target output. We create a simple feedforward neural network with one hidden layer using the Swish activation function. The model is compiled with the mean squared error loss and the Adam optimizer. We train the model on the generated data for 50 epochs. Finally, we evaluate the model's performance on the same dataset and print the final loss. You can use a similar approach with PyTorch or other deep learning frameworks to incorporate the Swish activation function into your neural network architectures.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("Swish Activation Function Visualization")

        @st.cache_data
        def swish(x, beta=1.0):
            return x * (1 / (1 + np.exp(-beta * x)))
            


        # Create a range of x values
        
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
        
            beta = st.slider("Beta (Scaling Factor)", 0.1, 5.0, 1.0)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
            x = np.linspace(-5, 5, 400)
            # Compute the Swish activation function values
            swish_values = swish(x, beta)

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, swish_values, label=f'Swish (β={beta})', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('Swish(x)')
            ax.set_title('Swish Activation Function')
            ax.legend()

            # Display the plot in Streamlit
            st.pyplot(fig)
                    
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Smoothness and Differentiability: The Swish function is a smooth and differentiable activation function, making it well-suited for gradient-based optimization methods such as stochastic gradient descent (SGD). The smoothness can help mitigate issues like vanishing gradients, which are sometimes encountered with the ReLU family of activation functions.")
        st.write("- Zero-Centered: Unlike ReLU, the Swish function is approximately zero-centered, which can be beneficial for training neural networks. Zero-centered activations can help prevent issues related to the mean-shift of activations and can lead to faster convergence during training.")
        st.write("- Adaptiveness with β: The Swish function introduces an adaptive component controlled by the scaling factor β. By adjusting β, you can control the sharpness of the activation function's curve. Smaller β values result in a curve closer to the ReLU-like behavior, while larger β values introduce saturation, making it closer to a sigmoid-like function. This adaptiveness can be useful in different network architectures and tasks.")
        
        st.divider()

        st.subheader("Demerits")
        st.write("- Computational Complexity: The Swish function involves additional operations compared to simpler activation functions like ReLU, particularly due to the inclusion of the sigmoid function. This can make it computationally more expensive, which may be a concern in resource-constrained environments or when designing highly efficient neural networks.")
        st.write("- Limited Adoption: While the Swish function has shown promise in research, it has not seen widespread adoption in practical applications as of my last knowledge update in September 2021. Commonly used activation functions like ReLU and its variants (e.g., Leaky ReLU) continue to dominate in many deep learning tasks.")
        st.write("- Lack of Theoretical Justification: The Swish function lacks a strong theoretical foundation explaining its superiority over other activation functions in practice. While it performs well empirically in some cases, it may not be the best choice for all types of neural networks and tasks.")
        st.write("- Increased Model Complexity: The inclusion of the β parameter introduces additional learnable parameters in the neural network, potentially increasing the model's complexity and the risk of overfitting, especially in small datasets.")