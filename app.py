# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model


import cv2

# Load the trained model
model_path = '/home/gesture-queen/Desktop/EmulsifyFinal/Completemodel1.pkl'
print(model_path)
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    def get_max_info(final_features):
        try:
            if final_features.size == 0:
                return None, None
                
            max_value = np.max(final_features)
            max_index = np.argmax(final_features)
            
            return max_value, max_index
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return None, None


    #extract image from Opencv-window
    from keras.preprocessing import image


    detection_model_path = '/home/gesture-queen/Desktop/EmulsifyFinal/emotionMusicBasedReccommendation/haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = '/home/gesture-queen/Desktop/EmulsifyFinal/emotionMusicBasedReccommendation/final_model.h5'
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["happy","sad"]


    def emotion_testing():
        cap=cv2.VideoCapture(0)
        while True:
            ret,test_img=cap.read()# captures frame and returns boolean value and captured image
            if not ret:
                continue
            gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            faces_detected = face_detection.detectMultiScale(gray_img, 1.32, 5)


            for (x,y,w,h) in faces_detected:
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
                roi_gray=cv2.resize(roi_gray,(48,48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = emotion_classifier.predict(img_pixels)

                #find max indexed array
                max_index = np.argmax(predictions[0])
                predicted_emotion = EMOTIONS[max_index]

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('Facial emotion analysis ',resized_img)



            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
        cap.release()
        cv2.destroyAllWindows
        return predicted_emotion

            # except KeyboardInterrupt:
            #     print("Emotion detection stopped by user")
            # finally:
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     return predicted_emotion
    
    # Make prediction

    def final():
        emotion_word=emotion_testing()
        max_val, max_idx = get_max_info(final_features)

        if max_val is not None:
            max_prob=max_val
            idx=max_idx

        if emotion_word=='sad'or max_idx==2:
            emotion_code=0
        else:
            emotion_code=1
            
        return emotion_code

    prediction = model.predict(final_features)
    output = 'Happy and Stress free' if prediction == 1 else 'Sad and Stressed'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True , port=5001)