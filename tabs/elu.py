import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def app():

    st.title("Exponential Linear Unit Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The Exponential Linear Unit (ELU) is an activation function used in artificial neural networks. It is designed to address some of the limitations of other activation functions like the Rectified Linear Unit (ReLU) and its variants. ELU introduces a nonlinear component that can help mitigate the "dying ReLU" problem and provide smoother gradients, which can facilitate more stable and faster convergence during training. In brief, the Exponential Linear Unit (ELU) is an activation function that combines linear behavior for positive inputs with a smooth, nonlinear component for negative inputs. It addresses some of the shortcomings of ReLU-based activations and can improve the training and generalization of deep neural networks.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("The formula for this function is:")
        st.latex(r'''\text{ELU}(x) = \begin{cases}
        x & \text{if } x > 0 \\
        \alpha \cdot (e^x - 1) & \text{if } x \leq 0
        \end{cases}''')
                
        st.markdown('Where:<br><br> x is the input to the ELU function.<br> α is a hyperparameter that controls the slope of the function for $x$ ≤ 0.<br> Typically, α is a small positive value, such as 1.0.',unsafe_allow_html=True)

        st.markdown('''Here's how the ELU activation function works:<br><ol><li>For positive values of (x>0), it behaves like the identity function, i.e., it returns x itself.</li><li>This means that if the input is positive, there is no activation, and the output is the same as the input.</li><li> For non-positive values of x ≤ 0, it uses the exponential function to smoothly transition from a negative value to zero.</li><li>The term α⋅($e^x$ −1) is responsible for this transition.</li><li> When x is very negative, $e^x$ approaches 0, and the function approaches −α.</li><li> As x approaches 0 from the left, the function approaches 0.</li><li>This characteristic helps prevent the "dying ReLU" problem, as the neuron can still have some gradient and learn even if it is not active.</li></ol>''',unsafe_allow_html=True)

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                function elu(x, alpha):
                    if x > 0:
                        return x
                    else:
                        return alpha * (exp(x) - 1)
                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Activation
            from tensorflow.keras.optimizers import Adam  # You can choose any optimizer here

            # Create a Sequential model
            model = Sequential()

            # Add layers to the model with ELU activation
            model.add(Dense(units=64, input_dim=input_dim))  # input_dim is the number of input features
            model.add(Activation('elu'))

            model.add(Dense(units=32))
            model.add(Activation('elu'))

            model.add(Dense(units=num_classes))  # num_classes is the number of output classes for classification tasks
            model.add(Activation('softmax'))  # Softmax activation for classification

            # Compile the model with an optimizer
            optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate as needed
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # Now, you can train the model using your dataset
            # model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

            ''')
   
        st.subheader('Conclusion')
        st.markdown('''<p style="font-size:17px;text-align:justify;">We import TensorFlow and the necessary modules from Keras. We create a Sequential model, which is a linear stack of layers. We add layers to the model using the Dense layer, which represents fully connected layers. Each Dense layer is followed by an ELU activation function using Activation('ELU'). We compile the model using the Adam optimizer, but you can replace it with any other optimizer of your choice. We also specify the loss function and metrics appropriate for your task (here, it's categorical cross-entropy for lassification). Finally, you can train the model using your training data with the model.fit() function. Make sure to adjust the input_dim, num_classes, and other hyperparameters according to your specific problem. Additionally, you should load your own dataset and preprocess it before training the model.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("ELU Activation Function Visualization")
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
            alpha = st.slider("Alpha (\u03B1)", 0.01, 5.0, 1.0)
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))

        with col2:
            # Create a range of x values
            x = np.linspace(-5, 5, 400)

            # Compute the ELU activation function values
            elu_values = np.where(x > 0, x, alpha * (np.exp(x) - 1))

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, elu_values, label=f'ELU(\u03B1={alpha})', color=colour,linewidth=thickness)
            ax.set_xlabel('x')
            ax.set_ylabel('ELU(x)')
            ax.set_title('ELU Activation Function')
            ax.legend()
            plt.grid()


            # Display the plot in Streamlit
            st.pyplot(fig)

        
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Smoothness and Continuity: ELU is a smooth and continuous activation function for all input values, including negative values. This smoothness ensures that its derivatives exist everywhere, which is particularly beneficial for gradient-based optimization during training. The absence of sharp transitions can lead to more stable convergence.")
        st.write("- Mitigating the Dying ReLU Problem: One of the primary advantages of ELU is its ability to mitigate the dying ReLU problem, which is a common issue with the Rectified Linear Unit (ReLU) and its variants. In deep neural networks, ReLU neurons can become inactive (output zero) for certain inputs during training, leading to dead neurons that don't update their weights. ELU does not have this issue because it can still carry gradients when the input is negative, preventing neurons from becoming completely inactive.")
        st.write("- Robust to Noisy Inputs: ELU is robust to noisy inputs due to its behavior in the negative saturation region. It can model both positive and negative saturation regions, making it less susceptible to noisy or outlier data compared to activations like ReLU.")
        st.write("- Flexibility: The ELU activation function introduces a hyperparameter α that controls the slope of the function for negative inputs. This allows model designers to fine-tune the activation function's behavior to suit their specific problem, potentially leading to improved model performance.")
        st.divider()
        st.subheader("Demerits")
        st.write("- Computational Complexity: ELU is more computationally expensive to compute compared to ReLU and its variants. The exponential term ($e^x$) in the negative region requires additional computation. While this may not be a significant issue for many applications, it can be a consideration for resource-constrained environments.")
        st.write("- Saturating Behavior for Large Negative Inputs: Although ELU is designed to mitigate the vanishing gradient problem in the negative region, it can still saturate for extremely large negative inputs. In such cases, the gradient becomes very small, potentially slowing down convergence.")
        st.write("- Not Universally Superior: While ELU addresses some issues associated with ReLU, it does not necessarily outperform ReLU in all situations. The choice between activation functions often depends on the specific problem, architecture, and dataset. ReLU variants like Leaky ReLU and Parametric ReLU can also be effective in certain cases.")
        