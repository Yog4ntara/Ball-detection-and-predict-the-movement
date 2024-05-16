'''
This program was developed for the purpose of a Laboratiom Teknik Fisika II project at the Institut Teknologi Bandung.
This program essentially detects and predicts the direction of a ball heading towards a goal. 
The program's output provides prediction data to the client via a Wi-Fi network.
'''

import cv2
import supervision as sv
import numpy as np
import requests
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression

class RingBuffer:

  """
  A circular buffer to store a fixed number of (x, y) coordinate pairs.

  Attributes:
    size (int): The maximum number of pairs to store in the buffer.
    buffer (list): A list to store the (x, y) coordinate pairs.
    index (int): The current index in the buffer to add the next pair.
  """
  
  def __init__(self, size):
    self.size = size
    self.buffer = [(None, None)] * size
    self.index = 0

  def add(self, x, y):
    """
    Add a new (x, y) coordinate pair to the buffer.

    Args:
      x (float): The x-coordinate.
      y (float): The y-coordinate.
    """
    self.buffer[self.index] = (x, y)
    self.index = (self.index + 1) % self.size

  def get(self):
    """
      Get the current buffer as a list of (x, y) coordinate pairs.

      Returns:
        list: A list of (x, y) coordinate pairs.
    """
    return self.buffer

  def get_linear_regression(self):
    """
      Perform linear regression on the (x, y) coordinate pairs in the buffer.

      Returns:
        LinearRegression: A fitted linear regression model.
    """
    x_values = [x for x, _ in self.buffer]
    y_values = [y for _, y in self.buffer]
    x_values = np.array(x_values).reshape(-1, 1)
    y_values = np.array(y_values)
    model = LinearRegression()
    model.fit(x_values, y_values)
    return model
  
class RangeCalculator:

  """
  Calculate the expected range based on the ball's position and goal parameters.

  Attributes:
    goalSize (float): The size of the goal.
    nearPost (float): The position of the near goal post.
    farPost (float): The position of the far goal post.
    goalPosition (float): The desired position of the ball relative to the goal.
  """
  
  def __init__(self, goalSize, nearPost, farPost, goalPosition):
    self.goalSize = goalSize
    self.nearPost = nearPost
    self.farPost = farPost
    self.goalPosition = goalPosition

  def moveRange(self, x):
    """
    Calculate the expected range based on the ball's x-coordinate.

    Args:
      x (float): The x-coordinate of the ball.

    Returns:
      float: The expected range.
    """
    coef = self.goalSize / (self.farPost - self.nearPost)
    if x < self.nearPost:
      expectedRange = 0
    elif x > self.farPost:
      expectedRange = 30
    else:
      expectedRange = (x - self.nearPost) * coef
    return expectedRange

class WebcamCapturer:

  """
    Open and manage the webcam for capturing frames.

    Attributes:
      resolution (tuple): The desired resolution of the webcam frames.
      fps (int): The desired frames per second of the webcam.
      cap (cv2.VideoCapture): The VideoCapture object for the webcam.
  """

  def __init__(self, resolution=(1280, 720), fps=60): # Change resolution and fps to use the desired settings
    self.resolution = resolution
    self.fps = fps

  def open_webcam(self, cam_index=1): # Change cam_index to use the desired webcam
    """
    Open the webcam and set the desired resolution and FPS.

    Args:
      cam_index (int, optional): The index of the webcam to open. Default is 0.
    """
    self.cap = cv2.VideoCapture(cam_index)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
    self.cap.set(cv2.CAP_PROP_FPS, self.fps)

  def get_frame(self):
    """
    Capture a single frame from the webcam.

    Returns:
      numpy.ndarray: The captured frame as a NumPy array.
    """
    ret, frame = self.cap.read()
    return frame

  def release(self):
    """
    Release the webcam resources.
    """
    self.cap.release()

class HTTPClient:

  """
  Send HTTP GET requests to a specified URL.

  Attributes:
    url (str): The URL to send the requests to.
  """

  def __init__(self, url):
    self.url = url

  def send_request(self, data):
    """
    Send an HTTP GET request with the provided data.

    Args:
      data (any): The data to send in the request.
    """
    params = {"data": data} # Change the parameter name to match the server's expected parameter name
    try:
      response = requests.get(self.url, params=params)
      response.raise_for_status()
    except requests.exceptions.RequestException as e:
      pass

class BallDetector:

  """
  Detect the ball in a frame using the YOLO model and annotate the frame.

  Attributes:
    model (YOLO): The YOLO model for object detection.
    box_annotator (sv.BoundingBoxAnnotator): The annotator for drawing bounding boxes.
    label_annotator (sv.LabelAnnotator): The annotator for drawing labels.
    dot_annotator (sv.DotAnnotator): The annotator for drawing dots.
    circle_annotator (sv.CircleAnnotator): The annotator for drawing circles.
  """

  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.box_annotator = sv.BoundingBoxAnnotator()
    self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
    self.dot_annotator = sv.DotAnnotator()
    self.circle_annotator = sv.CircleAnnotator()

  def detect_ball(self, frame):

    """
    Detect the ball in the given frame and annotate the frame.

    Args:
      frame (numpy.ndarray): The frame to detect the ball in.

    Returns:
      tuple: A tuple containing the annotated frame and the detection results.
    """

    results = self.model.predict(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    frame = self.label_annotator.annotate(frame, detections)
    frame = self.dot_annotator.annotate(frame, detections)
    frame = self.circle_annotator.annotate(frame, detections)

    return frame, results

def main():

  """
  The main function to run the ball detection and range calculation.
  """

  # Change the buffer size to store more or fewer coordinate pairs
  buffer = RingBuffer(3)

  # The goal parameters is (length of goal, near post in normalize x coordinate, far post in normalize x coordinate, goal position in normalize y coordinate)
  range_calculator = RangeCalculator(30, 0.2, 0.8, 0.0)

  # Open the webcam
  webcam_capturer = WebcamCapturer()
  webcam_capturer.open_webcam()

  # Initialize the HTTP client to send requests to the server (change the URL to match the server's URL)
  http_client = HTTPClient("http://192.168.4.1")

  # Initialize the ball detector with the desired YOLO model
  ball_detector = BallDetector('ball_v1.pt')

  while True:
    '''
    The main loop to capture frames from the webcam, detect the ball, calculate the range, and send the prediction to the server.
    Press 'q' to exit the loop and close the program.
    '''

    # Get the frame from the webcam and detect the ball
    frame = webcam_capturer.get_frame()
    annotated_frame, results = ball_detector.detect_ball(frame)

    # Annotate the frame with the normalized coordinates
    for r in results:
      boxes = r.boxes.xyxy
      boxes = boxes.cpu().numpy()

      # Get the coordinates of the ball
      for box in boxes:
        x1, y1, x2, y2 = box

        # Normalize coordinates
        frame_width, frame_height = webcam_capturer.resolution
        x = round((x1 + x2) / 2 / frame_width, 3)
        y = round((y1 + y2) / 2 / frame_height, 3)

        # Annotate the frame with the normalized coordinates
        annotated_frame = cv2.putText(annotated_frame, f"({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # uncomment the following line to see the normalized coordinates
        # print(f"Normalized coordinates: ({x}, {y})")

        # Add the coordinates to update the buffer
        buffer.add(x, y)

        # Get the buffer values 
        buffer_values = buffer.get()
        x_values = [x for x, _ in buffer_values]
        y_values = [y for _, y in buffer_values]
        # uncomment the following line to see the buffer values
        # print(x_values, y_values)

        # Perform linear regression if there are enough values in the buffer
        if None not in x_values:
          linear_regression_model = buffer.get_linear_regression()
          predicted_x = (range_calculator.goalPosition - linear_regression_model.intercept_) / linear_regression_model.coef_
          range_value = range_calculator.moveRange(predicted_x[0])

          # Send the predicted range to the server
          http_client.send_request(round(range_value, 2))

    # Annotate the frame with a message if no ball is detected
    if len(results) == 0:
      annotated_frame = cv2.putText(annotated_frame, "No ball detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Detection Ball', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # Release the webcam resources and close the OpenCV windows
  webcam_capturer.release()
  cv2.destroyAllWindows()

# Run the main function if the script is executed
if __name__ == '__main__':
  main()