from flask import render_template, Flask, request, url_for
from keras.models import load_model
import pickle
import tensorflow as tf

# Disable eager execution if using TensorFlow 2.x
tf.compat.v1.disable_eager_execution()

app = Flask(__name__)

# Load the count vectorizer
with open(r'C:\Users\3525 R7\Desktop\major project\model.pkl', 'rb') as file:
    cv = pickle.load(file)

# Load the model
cla = load_model('C:\\Users\\3525 R7\\Desktop\\major project\\cellphone.h5')
cla.compile(optimizer='adam', loss='binary_crossentropy')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def page2():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        topic = request.form['tweet']
        print("Hey " + topic)
        topic = cv.transform([topic])
        print("\n" + str(topic.shape) + "\n")

        # Prediction block
        y_pred = cla.predict(topic)
        print("pred is " + str(y_pred))

        if y_pred > 0.5:
            topic = "Positive Tweet"
        else:
            topic = "Negative Tweet"

        return render_template('index.html', ypred=topic)

if __name__ == "__main__":
    app.run(debug=True)