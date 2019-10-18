
# keras-number-ocr

Prototype to get end to end components working for a number recognition software.  It is using Keras, mnist, numpy, and Flask.

Training portion of the software is very rudimentary at this point and requires a lot of further tuning.

# Usage

## Docker

```
docker run -p 5000:5000 --detach josephkiok/keras-number-ocr:latest
```

## Running Locally

```
# Trainer
python3 trainer.py
```

```
# Run the HTTP server
python3 server.py
```

Then, launch http://localhost:5000/

```
# To submit an image via curl:
curl -X POST -F image=@1.png 'http://localhost:5000/predict'
```

# References

* https://my.safaribooksonline.com/book/programming/python/9781617294433/chapter-2dot-before-we-begin-the-mathematical-building-blocks-of-neural-networks/ch02lev1sec5_html
* http://my.safaribooksonline.com/book/programming/python/9781617294433/chapter-5dot-deep-learning-for-computer-vision/ch05lev1sec1_html
