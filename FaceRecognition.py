from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import dlib
import cv2
import numpy as np


app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:satya@localhost/facerecognition'
app.debug = True

db = SQLAlchemy(app)

class Faces(db.Model):
    __tablename__ = 'Faces'
    username = db.Column(db.String(40), primary_key=True)
    imagedata = db.Column(db.String())

    def __init__(self, username, imagedata):
        self.username = username
        self.imagedata = imagedata



@app.route("/registration", methods=["POST"])
def facerecog():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    user_name = request.values["username"]
    image_input = request.files["image"].read()
    image_input = np.frombuffer(image_input, np.uint8)
    image_input = cv2.imdecode(image_input, cv2.IMREAD_COLOR)
    img1_detection = detector(image_input, 1)
    print(len(img1_detection))
    img1_shape = sp(image_input,img1_detection[0])
    img1_aligned = dlib.get_face_chip(image_input, img1_shape)
    img1_representation = model.compute_face_descriptor(img1_aligned)
    img1_representation = np.array(img1_representation)

    list = []
    for i in img1_representation:
        list.append(i)
    print(type(list))
    input = Faces(username=user_name, imagedata=list)
    db.session.add(input)
    db.session.commit()
    return f"{user_name},{list}"

@app.route("/login", methods=['POST'])
def authentication():
    input_image = request.files["input"].read()
    input_image = np.frombuffer(input_image, np.uint8)
    input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    img2_detection = detector(input_image, 1)
    print(len(img2_detection))
    img2_shape = sp(input_image, img2_detection[0])
    img2_aligned = dlib.get_face_chip(input_image, img2_shape)
    img2_representation = model.compute_face_descriptor(img2_aligned)
    img2_representation = np.array(img2_representation)
    print(type(img2_representation))

    def findEuclideanDistance(source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    user_check = Faces.query.all()
    l1 = []
    l2 = []
    for i,j in enumerate(user_check):
        for k in j.imagedata.split(","):
            if "{" in k:
                k = k.replace("{", " ")
                k = float(k)
                l1.append(k)
            elif "}" in k:
                k = k.replace("}", " ")
                k = float(k)
                l1.append(k)
            elif "{" and "}" not in k:
                k = float(k)
                l1.append(k)
        l2.append(l1)
        l1 = []
    for e, f in enumerate(l2):
        print(type(l2[e]))
        array = np.array(f, dtype="float64")
        print(len(array))
        distance = findEuclideanDistance(array, img2_representation)
        threshold = 0.6
        if distance < threshold:
            return f"Authentication Successful"

    return "Authentication Failed. Try creating an account first."

if __name__=="__main__":
    db.create_all()
    app.run(debug=True, port=7000)