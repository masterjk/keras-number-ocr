
# keras-number-ocr

Prototype to get end to end components working for a number recognition software.  It is using Keras, mnist, numpy, and Flask.

Training portion of the software is very rudimentary at this point and requires a lot of tuning.

# Usage

## Docker

## Running Locally

```
# Trainer
python3 trainer.py
```

```
# Run the HTTP server
python3 server.py
```

```
# Submit a request via curl
curl -X POST -F image=@1.png 'http://localhost:5000/predict'
```

# References

* https://my.safaribooksonline.com/book/programming/python/9781617294433/chapter-2dot-before-we-begin-the-mathematical-building-blocks-of-neural-networks/ch02lev1sec5_html
