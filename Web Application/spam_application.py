from flask import Flask, render_template, request
import joblib as jl
app = Flask(__name__)
@app.route('/')
def get_input():
    return render_template('input.html')
@app.route('/classify', methods=['POST'])
def classify():
    # load the vectorizer and model
    VECT = open("vectorizer.pkl", 'rb')
    MODEL = open("model.pkl", 'rb')

    # instantiating the vectorizer and svm model
    cv = jl.load(VECT) # count vectorizer
    clf = jl.load(MODEL) # svm model

    if request.method == 'POST':
        sms = request.form['message']
        data = [sms]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)


