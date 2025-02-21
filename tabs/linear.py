import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def app():

    st.title("Linear Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The linear activation function, also known as the identity function, is one of the simplest activation functions used in neural networks. It's a straightforward mathematical function that performs a linear transformation on its input. The primary purpose of the linear activation function is to pass the input as it is, without introducing any non-linearity. It is typically used in certain regression problems or as a part of more complex neural network architectures. The linear activation function is rarely used in hidden layers of neural networks because it doesn't introduce non-linearity, and neural networks with only linear activations are equivalent to linear regression models. However, it can be useful in the output layer of regression models or for certain special cases where a linear transformation is explicitly required. In most practical applications, non-linear activation functions like ReLU (Rectified Linear Unit), sigmoid, or hyperbolic tangent (tanh) are preferred in the hidden layers of neural networks because they enable the network to learn complex, non-linear patterns in the data. The linear activation function is rarely used in hidden layers of neural networks because it doesn't introduce non-linearity, and neural networks with only linear activations are equivalent to linear regression models. However, it can be useful in the output layer of regression models or for certain special cases where a linear transformation is explicitly required. In most practical applications, non-linear activation functions like ReLU (Rectified Linear Unit), sigmoid, or hyperbolic tangent (tanh) are preferred in the hidden layers of neural networks because they enable the network to learn complex, non-linear patterns in the data.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The formula for this function is:")
        st.latex(r'''\ f(x)  =  x''')
        st.write("Here, x is the input to the function, and f(x) is the output. As you can see, f(x) equals x for all values of x. This means that the output is directly proportional to the input, with no change in the shape or characteristics of the data. Mathematically, this means that the weights and biases associated with the layer will simply scale and shift the input data without introducing any non-linearity. Therefore, the output of a layer with linear activation can be expressed as:")
        st.latex(r'''\ Output=W⋅Input+b''')
        st.markdown('Where:<br> $Output$ is the output of the layer.<br> $W$ represents the weight matrix.<br>$Input$ is the input data to the layer.<br>$b$ is the bias vector. <br><br>',unsafe_allow_html=True)

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                def linear_activation(x, a, b):
                    return a * x + b

                x = 2.0
                a = 0.5
                b = 1.0

                output = linear_activation(x, a, b)
                print(output)

                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import tensorflow as tf
            from tensorflow import keras
            import numpy as np

            # Generate some sample data
            np.random.seed(0)
            X = np.random.rand(100, 1)
            y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

            # Define a simple neural network model with a single dense layer using a linear activation function
            model = keras.Sequential([
                keras.layers.Dense(units=1, activation='linear', input_shape=(1,))
            ])

            # Compile the model
            model.compile(optimizer='sgd', loss='mean_squared_error')

            # Train the model
            model.fit(X, y, epochs=100, verbose=1)

            # Make predictions
            predictions = model.predict(X)

            # Print the trained weights and bias
            weights, bias = model.layers[0].get_weights()
            print("Trained Weight:", weights)
            print("Trained Bias:", bias)
            ''')
        st.subheader('Conclusion')
        st.markdown('''<p style="font-size:17px;text-align:justify;">Import the necessary libraries, including TensorFlow and NumPy.
            Generate some sample data (X and y) for a simple linear regression problem.
            Define a neural network model with a single dense (fully connected) layer. We specify the activation function as 'linear' when defining the layer, which means it will use the linear activation function.
            Compile the model with stochastic gradient descent ('sgd') as the optimizer and mean squared error ('mean_squared_error') as the loss function, which is commonly used for regression problems.
            Train the model on the sample data for 100 epochs.
            After training, we print the trained weights and bias of the layer, which should approximate the values of 2 and 1 (from the data generation process).
            This example demonstrates how to create a simple neural network with a linear activation function for a regression problem using TensorFlow in Python.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
        # Define the linear activation function
        def linear_activation(x, a, b):
            return a * x + b

        # Streamlit app
        st.subheader("Linear Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            a = st.slider("Slope (a)", min_value=-5.0, max_value=5.0, step=0.1, value=1.0)
            b = st.slider("Intercept (b)", min_value=-10.0, max_value=10.0, step=0.1, value=0.0)
            st.divider()
           
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))

        with col2:
            # Generate x values
            x = np.linspace(-10, 10, 400)
            
            # Calculate y values using the linear activation function
            y = linear_activation(x, a, b)

            # Plot the function
            fig, ax = plt.subplots()
            ax.plot(x, y,color=colour,linewidth=thickness)
            ax.set_xlabel("Input (x)")
            ax.set_ylabel("Output (f(x))")
            ax.set_ylim(-10,10)
            ax.set_title("Linear Activation Function")
            plt.grid()

            # Display the plot in Streamlit
            st.pyplot(fig)

        
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Simplicity: Linear activation is straightforward and computationally efficient. It involves a simple linear transformation of the input data, making it easy to implement and understand.")
        st.write("- No Saturation: Unlike some other activation functions like sigmoid or tanh, linear activation doesn't suffer from the vanishing gradient problem. This means that gradients don't become extremely small, which can make training more stable, especially in deep neural networks.")
        st.write("- Interpretability: Linear activation retains the interpretability of the input features since it's essentially a linear combination of those features. This can be advantageous in cases where interpretability and feature importance are critical.")
        st.write("- Use in Regression: Linear activation is well-suited for regression problems, where the network is tasked with predicting continuous numeric values. In regression, the model needs to approximate a linear relationship between input features and output.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Limited Expressiveness: Linear activation can only model linear relationships between input and output. It lacks the capacity to capture complex, nonlinear patterns in data. In many real-world problems, the relationships are nonlinear, which can limit the usefulness of linear activation.")
        st.write("- Not Suitable for Classification: For classification tasks, where the goal is to separate data into distinct classes, linear activation is not suitable. It can't create decision boundaries that separate classes effectively since it only performs linear transformations.")
        st.write("- Loss of Depth: In deep neural networks, stacking multiple layers with linear activation functions is essentially equivalent to having a single-layer network. This means that deep architectures may not be able to learn hierarchical or complex representations, which are often needed for tasks like image recognition or natural language processing.")
        st.write("- Output Range Limitation: The output of a linear activation function can cover a wide range of values (both positive and negative), which might not be desirable for some tasks. For example, when dealing with probabilities, it's common to use activation functions that restrict outputs to a specific range (e.g., sigmoid for [0, 1]).")