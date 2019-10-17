import tensorflow as tf
import PIL
import flask
import io
import numpy
import base64

App = flask.Flask(__name__)
model = tf.keras.models.load_model("mnist.h5")  # type: tf.keras.models.Sequential


def debug(arr):
    str = ""
    for i in arr:
        for j in i:
            if j == 0:
                str += " "
                print(" ", end="")
            else:
                str += "*"
                print("*", end="")
        print()
        str += "\n"
    return str


@App.route("/predict", methods=["POST", "GET"])
def predict():
    if flask.request.files.get("image"):

        response = {}

        # Read image from HTTP request
        buf = flask.request.files["image"].read()

        # Convert to image object
        image = PIL.Image.open(io.BytesIO(buf))
        image = image.resize((28, 28))

        # Convert from RGB to Grayscale
        image = image.convert("L")

        # Convert image to a numpy array
        nparr = numpy.array(image)

        # Reshape numpy array from 28x28x1 (28x28 by 0-255 color) to a 1-dimensional array
        nparr_reshaped = nparr.reshape(1, 784)

        # Convert from 0-255, to 0-1
        nparr_reshaped = nparr_reshaped.astype('float32') / 255

        predictions = model.predict_proba(nparr_reshaped, verbose=1)
        answer = predictions.argmax(axis=-1)
        response["answer"] = str(answer[0])

        debugString = debug(nparr)
        debugBuf = bytearray(debugString.encode("utf-8"))
        response["debug"] = base64.b64encode(debugBuf).decode("utf-8")
        return flask.jsonify(response)
    else:
        flask.abort(404)


@App.route("/", methods=["GET"])
def home():
    return serve("index.html")


@App.route("/<path:path>", methods=["GET"])
def serve(path):
    return flask.send_from_directory('./static/', path)


if __name__ == "__main__":
    App.run(debug=False, threaded=True, host='0.0.0.0')
