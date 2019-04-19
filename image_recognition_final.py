#Jordan Rimmer
import numpy as np
from keras.preprocessing import image
from keras.applications import resnet50


model = []
a = True

def prediction(DIR):
    global a
    # Model needs to be read in exactly one time as a result of flasks state continuity
    if (a == True):
        # Load Keras' ResNet50 model that was pre-trained against the ImageNet database
        global model
        model = resnet50.ResNet50()
        a = False

    # Load the image file, resizing it to 224x224 pixels (required by this model)
    img = image.load_img(DIR, target_size=(224, 224))

    # Convert the image to a numpy array
    x = image.img_to_array(img)

    # Add a forth dimension since Keras expects a list of images
    x = np.expand_dims(x, axis=0)

    # Scale the input image to the range used in the trained network
    x = resnet50.preprocess_input(x)

    # Run the image through the deep neural network to make a prediction
    predictions = model.predict(x)

    # Look up the names of the predicted classes. Index zero is the results for the first image.
    predicted_classes = resnet50.decode_predictions(predictions, top=6)

    #print("This is an image of:")

    #for imagenet_id, name, likelihood in predicted_classes[0]:
        #print(" - {}: {:2f} likelihood".format(name, likelihood))
    #print(predicted_classes)
    return predicted_classes

#prediction(testDIR)