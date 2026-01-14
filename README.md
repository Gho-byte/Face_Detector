# Face Detection and Recognition Application

This project is a desktop application developed using Python that provides real-time face detection and recognition capabilities. It utilizes computer vision techniques to train custom models on user faces and subsequently recognize them in live video feeds. The application features a graphical user interface (GUI) built with Tkinter for easy interaction.

## Features

* **Real-time Face Detection:** Detects faces in the live video stream using Haar Cascade classifiers.
* **Custom Model Training:** Allows users to create datasets of their own faces by capturing multiple images.
* **Face Recognition:** Recognizes trained faces in real-time using the LBPH (Local Binary Patterns Histograms) Face Recognizer.
* **User-Friendly GUI:** Simple interface for switching between detection and recognition modes, managing models, and viewing the camera feed.
* **Threaded Performance:** Uses threading to ensure the GUI remains responsive while processing video frames.

## Prerequisites

To run this application, you need to have Python installed along with the following libraries:

* opencv-python (cv2)
* numpy
* Pillow (PIL)
* tkinter (usually included with Python)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Gho-byte/Face_Detector
    cd Face_Detector
    ```

2.  Install the required dependencies:
    ```bash
    pip install opencv-python numpy Pillow
    ```

3.  Ensure you have the `haarcascade_frontalface_default.xml` file in the project directory. If not, download it from the official OpenCV repository.

## Usage

1.  Run the application:
    ```bash
    python app.py
    ```

2.  **Training a New Model:**
    * Navigate to the "Train Model" tab.
    * Enter the number of images to capture (between 100 and 900) in the input field.
    * Enter the name of the person in the "Name" field.
    * Click "Start" to begin capturing face images. The application will automatically train and save the model in the `Models` directory.

3.  **Recognizing Faces:**
    * Navigate to the "Use Model" tab.
    * Select a trained model from the dropdown list.
    * Click the button to load the model.
    * The application will display the live feed and label recognized faces with their names and confidence scores.

## Project Structure

* `app.py`: The main entry point of the application containing the GUI and logic.
* `Models/`: Directory where trained `.yml` models are stored.
* `haarcascade_frontalface_default.xml`: The pre-trained Haar Cascade classifier for face detection.

## Technical Details

* **Language:** Python
* **GUI Framework:** Tkinter
* **Computer Vision Library:** OpenCV
* **Face Recognition Algorithm:** LBPH (Local Binary Patterns Histograms)
* **Concurrency:** Python `threading` module for non-blocking UI operations.

## Author

Mohamed Ait Lafkih
