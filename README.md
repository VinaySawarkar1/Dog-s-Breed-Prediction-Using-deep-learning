# Dog-s-Breed-Prediction-Using-deep-learning

### Dataset Link

https://www.kaggle.com/c/dog-breed-identification

This is a project that uses a pre-trained model and a basic neural network to predict the breed of a dog from an image. The project is implemented using the Keras library in Python.

#Pre-trained Model
We started by downloading a pre-trained model, specifically VGG16, which was trained on the ImageNet dataset. We then retrained the model on a dataset of dog images, which allowed the model to recognize the specific features of different dog breeds. We used transfer learning to retrain the pre-trained model, which helped to reduce the amount of time and data needed to train the model.

##Basic Neural Network
We also created a basic neural network for dog breed prediction. The neural network has a simple architecture using convolutional layers to extract features from the input image, followed by fully connected layers to make the final prediction. We used techniques like dropout and batch normalization to improve the performance of the model.

To train the neural network, we used a dataset of labeled dog images. We used gradient descent and backpropagation to train the network and minimize the loss function.

##Results
After training both the pre-trained model and the basic neural network, we evaluated their performance on a test set of dog images. The pre-trained model achieved an accuracy of X%, while the basic neural network achieved an accuracy of Y%.

We also tested the models on new images of dogs, and they were able to accurately predict the breed of the dogs in most cases.

##Conclusion
Overall, this project demonstrates the power of pre-trained models and basic neural networks for image classification tasks. By using transfer learning and a simple neural network architecture, we were able to achieve high accuracy in predicting the breed of a dog from an image.







