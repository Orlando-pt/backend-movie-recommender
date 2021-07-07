# Movie Recommender

This project consists of a simple Flask API that allows the retrieving of movie recommendations.

## Technologies used

- Flask
- Gunicorn
- TensorFlow
- Collaborative Filtering algorithms

## Run the project

1. Create virtual environment (for development python3.8 was used)

    ```bash
    $ python3 -m venv venv
    ```

2. Install dependencies

    ```bash
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    ```

3. Run for testing

    ```bash
    $ python main.py
    ```

4. Run with Gunicorn (WSGI)

    ```bash
    $ gunicorn -b 127.0.0.1:8080 -w 1 wsgi:app
    ```

5. Check the api at [http://127.0.0.1:8080](http://127.0.0.1:8080).