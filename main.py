import cv2
import numpy as np
import requests
from time import sleep
from ultralytics import YOLO
import serial
import time
import socket

class PersonDetectionSystem:
    def __init__(self, model_path, arduino_port, baud_rate, esp32_ip, socket_host, socket_port):
        # Initialize YOLO model
        self.model = YOLO(model_path)

        # Set up the serial connection
        self.ser = serial.Serial(arduino_port, baud_rate, timeout=1)
        time.sleep(2)  # Allow the connection to establish

        # ESP32-CAM details
        self.esp32_url = f"http://{esp32_ip}/capture"

        # Socket setup
        self.socket_host = socket_host
        self.socket_port = socket_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.socket_host, self.socket_port))
        self.socket.listen(1)  # Listen for one connection
        print(f"Socket server started at {self.socket_host}:{self.socket_port}")
        
        self.client_socket, self.client_address = self.socket.accept()
        print(f"Connection established with {self.client_address}")

        self.person_present = False

    def get_image(self):
        try:
            response = requests.get(self.esp32_url)
            if response.status_code == 200:
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                return image
            else:
                print(f"Failed to get image, status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting image: {str(e)}")
            return None

    def detect_person(self, image):
        results = self.model(image)
        boxes = results[0].boxes
        classes = boxes.cls
        scores = boxes.conf

        person_detected = False
        for i in range(len(boxes)):
            if classes[i] == 0:  # Class ID 0 corresponds to 'person' in COCO dataset
                person_detected = True
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image, person_detected

    def send_signal_to_arduino(self, signal):
        try:
            self.ser.write(signal.encode())
            print(f"Sent to Arduino: {signal}")
        except Exception as e:
            print(f"Error sending signal: {str(e)}")

    def send_signal_to_client(self, signal):
        try:
            self.client_socket.sendall(signal.encode())
            print(f"Sent to client: {signal}")
        except Exception as e:
            print(f"Error sending signal to client: {str(e)}")

    def run(self):
        try:
            while True:
                image = self.get_image()
                if image is not None:
                    image_with_person, detected = self.detect_person(image)

                    # Send signal to Arduino and client
                    if detected and not self.person_present:
                        self.send_signal_to_arduino("1")
                        self.send_signal_to_client("Person detected")
                        self.person_present = True
                    elif not detected and self.person_present:
                        self.send_signal_to_arduino("0")
                        self.send_signal_to_client("No person detected")
                        self.person_present = False

                    # Display the image
                    cv2.imshow('ESP32-CAM Stream', image_with_person)

                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                sleep(0.1)

        except KeyboardInterrupt:
            print("Program interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()
        self.ser.close()
        self.client_socket.close()
        self.socket.close()
        print("Resources released.")

if __name__ == "__main__":
    system = PersonDetectionSystem(
        model_path="yolov8n.pt",
        arduino_port="/dev/cu.usbmodem101",
        baud_rate=9600,
        esp32_ip="192.168.1.105",
        socket_host="0.0.0.0",
        socket_port=12345
    )
    system.run()
