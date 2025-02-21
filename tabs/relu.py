import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def app():

    st.title("Rectified Linear Unit Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The Rectified Linear Unit (ReLU) activation function is one of the most widely used activation functions in deep learning and neural networks. It introduces non-linearity to the model by outputting zero for all negative inputs and passing positive inputs directly through.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The ReLU activation function can be mathematically defined as follows:")
        st.latex(r'''f(x) = \begin{cases}
        x, & \text{if } x > 0 \\
        0, & \text{if } x \leq 0
        \end{cases}''')
                
        st.markdown('Where:<br><br> $f(x)$ is the output to the ReLU activation function.<br> $x$ is the input for the activation function.',unsafe_allow_html=True)

        st.markdown('''Here's how the RELU activation function works:<br><ol><li> $For Positive Inputs (x>0)$: When the input x is greater than zero, ReLU behaves as an identity function, passing the input value directly to the output. In other words, it keeps all positive values unchanged.</li><li> $For Non-Positive Inputs (x≤0)$: When the input $x$ is zero or negative, ReLU outputs zero. It effectively "kills" or "zeros out" any negative values, setting them to zero. This thresholding behavior introduces non-linearity into the neural network model.</li></ol>''',unsafe_allow_html=True)

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                function relu(x):
                    if x > 0:
                        return x
                    else:
                        return 0
                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import tensorflow as tf
            from tensorflow import keras

            # Define a simple neural network with ReLU activation
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),  # Input layer (Flatten 28x28 images)
                keras.layers.Dense(128, activation='relu'),  # Hidden layer with ReLU activation
                keras.layers.Dense(10, activation='softmax') # Output layer with softmax activation
            ])

            # Compile the model
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

            # Load and preprocess your dataset (e.g., MNIST)
            # Replace this with your actual data loading and preprocessing code
            mnist = keras.datasets.mnist
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            train_images, test_images = train_images / 255.0, test_images / 255.0

            # Train the model
            model.fit(train_images, train_labels, epochs=5)

            # Evaluate the model
            test_loss, test_acc = model.evaluate(test_images, test_labels)
            print(f'Test accuracy: {test_acc}')

            ''')
   
        st.subheader('Conclusion')
        st.markdown('''<p style="font-size:17px;text-align:justify;">We create a simple neural network using the Sequential API from Keras.
        The hidden layer uses ReLU activation (activation='relu'), which introduces non-linearity into the network.
        The output layer uses softmax activation for multiclass classification.
        We compile the model with appropriate loss and optimizer.
        We load and preprocess the MNIST dataset (you should replace this with your own dataset loading code).
        We train the model on the training data.
        Finally, we evaluate the model's accuracy on the test data.
        The ReLU activation in the hidden layer allows the network to learn complex representations from the input data, which is important for its ability to classify images correctly.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("RELU Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
            st.divider()
            a = st.slider("Maximum Range", min_value=-10.0, max_value=10.0, step=0.2, value=1.0)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
            # Create a range of x values
            x = np.linspace(-5, a, 400)

            # Compute the ReLU activation function values
            relu_values = np.maximum(0, x)

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, relu_values, label='ReLU', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('ReLU(x)')
            ax.set_title('ReLU Activation Function')
            ax.legend()
            plt.grid()
            # Display the plot in Streamlit
            st.pyplot(fig)

        
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Simplicity: ReLU is computationally efficient and straightforward to implement. It involves a simple thresholding operation without any complex mathematical computations.")
        st.write("- Non-Linearity: Despite its simplicity, ReLU introduces non-linearity to the neural network, enabling it to learn complex and hierarchical representations of data. This non-linearity is crucial for the model's ability to capture intricate patterns in data.")
        st.write("- Prevents Vanishing Gradient: ReLU helps mitigate the vanishing gradient problem, which can occur with activation functions that squash their inputs into a small range. Since ReLU has a derivative of 1 for positive inputs, it allows gradients to flow through during backpropagation, facilitating training in deep networks.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Dead Neurons: A common issue with ReLU is the dying ReLU problem, where neurons can become inactive (output zero) for all inputs during training. If a large gradient flows through a ReLU neuron for a certain input, it might update its weights in a way that it always outputs zero for that input. This can lead to dead neurons that don't contribute to learning.")
        st.write("- Not Centered Around Zero: ReLU outputs only positive values or zeros, which means it is not centered around zero. This lack of symmetry can make optimization more challenging and result in a gradient update that shifts the model weights in one direction.")
        st.write("- Exploding Activation: Although ReLU helps mitigate the vanishing gradient problem, it can lead to an exploding gradient problem when the weights are not properly initialized or when the learning rate is too high. Large gradients can cause weight updates that result in unstable training.")
        st.write("- Lack of Smoothness: ReLU is not a smooth function at the point where it switches from zero to positive values (at $x=0$). This lack of smoothness can sometimes result in convergence issues during training.")
