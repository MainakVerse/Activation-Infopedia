import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def app():

    st.title("Binary Step Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The binary step activation function is one of the simplest and most basic activation functions used in artificial neural networks. It's a piecewise linear function that produces binary outputs, typically 0 or 1, based on a certain threshold. The binary step function is used to introduce non-linearity into a neural network model.This function takes an input value x and a threshold (defaulting to 0) and returns 0 if x is less than the threshold and 1 if x is greater than or equal to the threshold. In practice, the binary step function is rarely used in modern deep learning because of its lack of differentiability, which makes it unsuitable for gradient-based optimization methods. More advanced activation functions like sigmoid, tanh, and ReLU are preferred because they allow for smoother training and convergence in deep neural networks.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The formula for this function is:")
        #st.markdown('''$0$ if $x < threshold$''')
        #st.markdown('''$1$ if $x > threshold$''')
        st.latex(r'''\ 0 \;\; if \ x < threshold''')
        st.latex(r'''\ 1 \;\; if \ x \geq threshold''')
        
        st.markdown('Where:<br><br> $f(x)$  represents the output of the binary step function for a given input $x$.<br> $threshold$ is a predefined value that determines the point at which the function switches from producing 0 to producing 1.  <br><br> Any input value less than the threshold results in an output of 0, and any input value greater than or equal to the threshold results in an output of 1.',unsafe_allow_html=True)

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # Define a function named binary_step that takes two arguments:
                # - x: the input value
                # - threshold: the activation threshold (default is set to 0)
                function binary_step(x, threshold=0)
                    # Check if the input x is greater than or equal to the threshold
                    if x >= threshold:
                        # If x is greater than or equal to the threshold, return 1
                        return 1
                    else:
                        # If x is less than the threshold, return 0
                        return 0
                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import tensorflow as tf
            from tensorflow import keras
            import numpy as np

            # Define the binary step function
            def binary_step(x):
                return tf.where(x >= 0, 1, 0)

            # Generate some sample data
            np.random.seed(0)
            X = np.random.randn(100, 1)
            y = np.random.randint(0, 2, size=(100, 1))

            # Define a simple neural network model with a single dense layer using the binary step activation function
            model = keras.Sequential([
                keras.layers.Dense(units=1, activation=binary_step, input_shape=(1,))
            ])

            # Compile the model
            model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

            # Train the model
            model.fit(X, y, epochs=100, verbose=1)

            # Evaluate the model
            accuracy = model.evaluate(X, y, verbose=0)[1]
            print("Model Accuracy:", accuracy)

            ''')
        st.subheader('Conclusion')
        st.markdown('''<p style="font-size:17px;text-align:justify;">We define the binary_step function, which takes an input x and uses TensorFlow's tf.where function to return 1 if x is greater than or equal to 0, and 0 otherwise. We generate some sample data, including random input data X and binary labels y.
        We define a simple neural network model with a single dense layer. We specify the binary_step function as the activation function for this layer. The model is compiled with stochastic gradient descent ('sgd') as the optimizer and binary cross-entropy ('binary_crossentropy') as the loss function. We also track the accuracy metric. The model is trained on the sample data for 100 epochs. After training, we evaluate the model's accuracy on the same data.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
        # Define the linear activation function
        # Binary step activation function
        def binary_step_activation(x, threshold):
            return np.where(x >= threshold, 1, 0)

        # Streamlit app
        st.subheader("Binary Step Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            threshold = st.slider("Threshold:", -5.0, 5.0, 0.0, 0.1)
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))

        with col2:
            # Generate x values
            # Generate x values
            x = np.linspace(-10, 10, 400)
            # Compute binary step activation values
            y = binary_step_activation(x, threshold)

            # Create a plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, y,color=colour,linewidth=thickness)
            ax.set_xlabel('Input (x)')
            ax.set_ylabel('Output (Binary Step)')
            ax.set_title('Binary Step Activation Function')
            plt.grid()

            st.pyplot(fig)

        
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Simplicity: The binary step activation function is one of the simplest activation functions. It's easy to understand and computationally efficient.")
        st.write("- Interpretability: The output of this function is binary, typically 0 or 1. This can be interpreted as a clear decision or classification, which is useful in binary classification problems.")
        st.write("- Non-linearity: While it is a highly non-linear function, it is also piecewise constant. In some cases, this piecewise nature can be useful for certain tasks.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Not Suitable for Gradient Descent: The binary step activation function is not differentiable at the point where it jumps from 0 to 1. This non-differentiability makes it unsuitable for gradient-based optimization algorithms like stochastic gradient descent (SGD). These algorithms rely on computing gradients to update neural network weights.")
        st.write("- Vanishing Gradient Problem: The function has zero gradients almost everywhere, which can lead to the vanishing gradient problem. This makes it challenging to train deep neural networks using this activation function.")
        st.write("- Lack of Sensitivity: The binary step function does not take into account the magnitude of the input; it only checks whether it's above or below the threshold. As a result, it lacks sensitivity to small changes in input values, making it unsuitable for many real-world problems.")
        st.write("- Discontinuous and Non-Smooth: The binary step function is discontinuous, making it difficult to use in some optimization algorithms and numerical computations. The lack of smoothness can slow down convergence during training.")
        st.write("- Limited Expressiveness: Due to its binary nature, the binary step activation function is limited in its ability to capture complex relationships in data. It cannot approximate complex functions as effectively as some other activation functions like ReLU or sigmoid.")