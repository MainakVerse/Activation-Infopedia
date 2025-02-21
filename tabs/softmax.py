import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt

def app():

    st.title("Softmax Activation Function")


    selected = option_menu(None, ["Description", "Explanation", "Implementation", 'Visualization','Inference'], 
    default_index=0, orientation="horizontal",styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    })
    selected
      
    if selected == "Description":
        st.markdown('''<p style="font-size:20px; text-align:justify">The softmax activation function is a commonly used activation function in neural networks, particularly in the output layer of multi-class classification problems. It transforms a vector of real numbers into a probability distribution over multiple classes. The softmax function takes as input a vector z of real numbers and returns a vector of the same dimension with values in the range [0, 1] that sum to 1. The softmax activation function is commonly used in multi-class classification problems, including image classification, natural language processing tasks (e.g., sentiment analysis and named entity recognition), and any task where the model needs to assign a probability to multiple mutually exclusive classes. In brief, the softmax activation function is a crucial component of neural networks for multi-class classification. It ensures that the model's output represents class probabilities, enabling the selection of the most likely class during inference.</p>''', unsafe_allow_html=True)
        
    if selected=="Explanation":
        st.write("Here's a detailed explanation of the softmax activation function:")
        st.latex(r'''p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}''')
        st.markdown('''Where:<br><b><li>$e$ is the base of the natural logarithm, also known as Euler's number (approximately 2.71828).</li>
         <li>$z_i$ is the raw score or logit associated with the ith class. </li><li>K is the total number of classes.</li>''',unsafe_allow_html=True)
        st.markdown('<ul><li>Exponentiation $(e^z)$: For each element $z_i$ in the input vector $z$, we compute the exponential $e^{zi}$. This exponentiation ensures that each element is non-negative, and larger values in $z$ will result in larger values in the numerator of the softmax formula.</li><br><li>Sum of Exponentials ($Σe^z$): We calculate the sum of the exponentials of all elements in the input vector $z$, denoted as $\sum_{j=1}^{K} e^{z_j}$. This sum serves as a normalization constant.</li><br><li>Probability Calculation ($p_i$): For each class $i$, we compute the probability $p_i$ by dividing $e^{zi}$  by the sum of exponentials. This step ensures that the probabilities for all classes sum up to 1, which is a crucial property for classification.</li>',unsafe_allow_html=True)

        

        #st.markdown('''<iframe
        #src="https://30days.streamlit.app/?embed=true"
        #height="450"
        #style="width:100%;border:none;"
        #></iframe>''',unsafe_allow_html=True)

    if selected == "Implementation":
        st.subheader("Pseudocode")
        st.code('''
                # Softmax Activation Function Pseudocode
                function softmax(z):
                    # Initialize an empty list to store the probabilities
                    probabilities = []

                    # Compute the sum of exponentials of the input vector 'z'
                    sum_of_exponentials = sum(exp(z_j) for z_j in z)

                    # Calculate the probability for each class 'i'
                    for z_i in z:
                        probability_i = exp(z_i) / sum_of_exponentials
                        probabilities.append(probability_i)

                    # Return the list of probabilities
                    return probabilities

                # Example usage:
                input_vector = [2.0, 1.0, 0.1]  # Example input vector
                output_probabilities = softmax(input_vector)
                print("Softmax Probabilities:", output_probabilities)
                ''')
        st.subheader("Example of Implementation")
        st.code('''
            import tensorflow as tf
            from tensorflow.keras.datasets import mnist
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Flatten
            from tensorflow.keras.utils import to_categorical

            # Load and preprocess the MNIST dataset
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

            # Convert labels to one-hot encoding
            y_train = to_categorical(y_train, num_classes=10)
            y_test = to_categorical(y_test, num_classes=10)

            # Build a simple neural network with softmax output
            model = Sequential([
                Flatten(input_shape=(28, 28)),     # Flatten the 28x28 input images
                Dense(128, activation='relu'),     # Fully connected layer with ReLU activation
                Dense(10, activation='softmax')    # Output layer with softmax activation
            ])

            # Compile the model
            model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

            # Train the model
            model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

            # Evaluate the model on the test set
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
            print(f"Test Accuracy: {test_accuracy*100:.2f}%")

            ''')
   
        st.subheader('Conclusion')
        st.markdown('''<p style="font-size:17px;text-align:justify;">We load the MNIST dataset, which consists of grayscale images of handwritten digits (0 to 9). We preprocess the data by normalizing pixel values to the range [0, 1] and converting class labels to one-hot encoding. We build a simple neural network model with a softmax activation function in the output layer. This model is designed for multi-class classification. We compile the model with the categorical cross-entropy loss function, which is commonly used for multi-class classification tasks. We train the model on the training data for 5 epochs, using the Adam optimizer and a batch size of 32. Finally, we evaluate the model on the test set and print the test accuracy. The softmax function in the output layer ensures that the model's predictions represent class probabilities for each of the 10 possible digits in the MNIST dataset.</p>''',unsafe_allow_html=True)
        

    if selected == "Visualization":
             

        # Streamlit app
        st.subheader("Softmax Activation Function Visualization")

        @st.cache_data
        def softmax(z, temperature=1.0):
            exp_z = np.exp(z / temperature)
            softmax_scores = exp_z / exp_z.sum()
            return softmax_scores
        
        col1,col2 = st.columns([1,2])
        # Sliders to adjust parameters
        with col1:
            st.subheader("Adjust Parameters")
            # Create a range of class scores (input values)
            class_scores = np.linspace(-5, 5, 400)
            temperature = st.slider("Temperature", 0.1, 5.0, 1.0)
            
            st.divider()
            thickness = st.slider("Select Line thickness", min_value=1, max_value=7, step=1, value=1)
            colour = st.selectbox('Choose a colour for line',('blue','red','green','black'))
        with col2:
                        
            # Compute the softmax probabilities for different class scores
            softmax_probs = softmax(class_scores, temperature)

            # Create a bar chart to visualize the probabilities
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(len(class_scores)), softmax_probs, tick_label=class_scores, color=colour,linewidth=thickness)
            ax.set_xlabel('Class Scores')
            ax.set_ylabel('Softmax Probabilities')
            ax.set_title(f'Softmax Activation (Temperature={temperature})')

            # Display the bar chart in Streamlit
            st.pyplot(fig)
                    
    if selected == "Inference":
        st.subheader("Merits:")
        st.write("- Probabilistic Interpretation: The softmax function naturally provides class probabilities as output, making it suitable for multi-class classification problems where you need to assign an input sample to one of several possible classes. These probabilities are useful for ranking and decision-making.")
        st.write("- Normalization: The softmax function ensures that the probabilities sum up to 1.0, which enforces a probabilistic interpretation. This property is crucial when you want to select the class with the highest probability as the predicted class.")
        st.write("- Differentiability: The softmax function is differentiable, which is essential for training neural networks using gradient-based optimization algorithms such as gradient descent. It allows for the calculation of gradients during backpropagation.")
        st.write("- Generalization: Softmax can handle any number of classes, making it suitable for problems with a large number of classes.")
        st.write("- Softmax Temperature: The introduction of the softmax temperature parameter allows for fine-tuning the output distribution's sharpness. Higher temperatures result in a more uniform distribution, while lower temperatures sharpen the distribution.")
        
        st.divider()

        st.subheader("Demerits")
        st.write("- Sensitivity to Scale: The softmax function is sensitive to the scale of the input scores (logits). Small changes in the input scores can result in significant changes in the probabilities, potentially leading to numerical instability, especially if the scores are large or small..")
        st.write("- Exponential Complexity: Computing the exponential values in the softmax function can be computationally expensive, particularly when dealing with a large number of classes or a large batch size.")
        st.write("- Overfitting Risk: In practice, the softmax function can be prone to overfitting, especially when applied to high-dimensional data. It may lead to overconfident predictions and difficulties in generalization.")
        st.write("- Lack of Interpretability: While the softmax function provides class probabilities, it may not inherently provide interpretability at the feature level. Understanding why a particular prediction was made may require additional techniques such as feature importance analysis.")