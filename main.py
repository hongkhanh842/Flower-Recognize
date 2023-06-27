import os
import cv2
import numpy as np
from keras.models import load_model
import sys

# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

categories = np.sort(os.listdir('data'))

#Build app_data and layout
class CamApp(App):
    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Predict", on_press = self.predict , size_hint=(1, .1))
        self.prediction = Label(text="Result: ", size_hint=(1, .1))
        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.prediction)
        layout.add_widget(self.button)
        # Load tensorflow/keras model
        self.model = load_model(os.path.join('models', 'flowerclassifier.h5'))
        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120 + 500, 200:200 + 500, :]
        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and conver to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        ##byte_img = tf.io.read_file(file_path)
        img = cv2.imread(file_path)
        # Load in the image
        #img = tf.io.decode_jpeg(byte_img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Scale image to be between 0 and 1
        #img = img / 255.0
        im = cv2.resize(img_rgb, (128, 128))
        # Return image
        return im

    def predict(self, *args):

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('app_data', 'input_image', 'input_image.jpeg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 500, 200:200 + 500, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []

        input_img = self.preprocess(os.path.join('app_data', 'input_image', 'input_image.jpeg'))

        # Make Predictions
        result = self.model.predict(np.expand_dims(input_img/255, 0))
        results.append(result)

        # Set verification text
        self.prediction.text = "Result: " + categories[np.argmax(result)]

        # Log out details
        Logger.info(results)

        # test
        while True:
            img = cv2.imread(os.path.join('app_data', 'input_image', 'input_image.jpeg'), cv2.IMREAD_ANYCOLOR)
            cv2.imshow(categories[np.argmax(result)], img)
            cv2.waitKey(0)
            break

        return results
if __name__ == '__main__':
    CamApp().run()

