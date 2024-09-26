from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from PIL import Image
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class_name = ['apple', 'avocado', 'banana', 'cucumber', 'dragonfruit', 'durian', 'grape', 'guava', 'kiwi',  'lychee', 'mango', 'papaya', 'pear', 'pineapple', 'pomegranate', 'strawberry', 'tomato', 'watermelon']

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    input = Input(shape=(150, 150, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(18, activation='softmax', name='predictions')(x)

    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

my_model = get_model()
my_model.load_weights('.\weights\weights-49-0.84.keras')


@app.route('/api/predict', methods=['POST'])
def predict():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    try:
        img = Image.open(file.stream)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = my_model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_name[predicted_class_index]
        confidence = prediction[0][predicted_class_index]  

        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence)  
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
