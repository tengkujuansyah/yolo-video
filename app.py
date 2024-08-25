import cv2
from ultralytics import YOLO
from flask import Flask, render_template, jsonify, Response
from threading import Thread
import time
import requests
import json

app = Flask(__name__)

server_url = "http://localhost:5000/status"

model_path = r'X:\Skripsi\trained datasets\train12\weights\best.pt'
model = YOLO(model_path)

# Initialize video captures for four intersections
cap1 = cv2.VideoCapture(r'X:\Skripsi\Monitoring Test Video\Video\Vid1.mp4')  # Video untuk persimpangan 1
cap2 = cv2.VideoCapture(r'X:\Skripsi\Monitoring Test Video\Video\Vid2.mp4')  # Video untuk persimpangan 2
cap3 = cv2.VideoCapture(r'X:\Skripsi\Monitoring Test Video\Video\Vid3.mp4')  # Video untuk persimpangan 3
cap4 = cv2.VideoCapture(r'X:\Skripsi\Monitoring Test Video\Video\Vid4.mp4')  # Video untuk persimpangan 4

# Vehicle count for each intersection
vehicle_count_1 = 0
vehicle_count_2 = 0
vehicle_count_3 = 0
vehicle_count_4 = 0

car_count_1 = 0
car_count_2 = 0
car_count_3 = 0
car_count_4 = 0

motocycle_count_1 = 0
motocycle_count_2 = 0
motocycle_count_3 = 0
motocycle_count_4 = 0

bus_count_1 = 0
bus_count_2 = 0
bus_count_3 = 0
bus_count_4 = 0

frame_1 = None
frame_2 = None
frame_3 = None
frame_4 = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data1')
def data1():
    global vehicle_count_1, car_count_1, motocycle_count_1, bus_count_1
    return jsonify(count=vehicle_count_1, c=car_count_1, m=motocycle_count_1, b=bus_count_1)

@app.route('/data2')
def data2():
    global vehicle_count_2, car_count_2, motocycle_count_2, bus_count_2
    return jsonify(count=vehicle_count_2, c=car_count_2, m=motocycle_count_2, b=bus_count_2)

@app.route('/data3')
def data3():
    global vehicle_count_3, car_count_3, motocycle_count_3, bus_count_3
    return jsonify(count=vehicle_count_3, c=car_count_3, m=motocycle_count_3, b=bus_count_3)

@app.route('/data4')
def data4():
    global vehicle_count_4, car_count_4, motocycle_count_4, bus_count_4
    return jsonify(count=vehicle_count_4, c=car_count_4, m=motocycle_count_4, b=bus_count_4)

def generate_video_feed(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame if necessary for efficiency
        # frame = cv2.resize(frame, (640, 360))

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert the JPEG image to bytes
        frame_bytes = jpeg.tobytes()

        # Yield frame as part of a multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
def get_video_feed(intersection_id):
    global frame_1, frame_2, frame_3, frame_4
    while True:
        # Update global vehicle count
        if intersection_id == 1:
            frame = frame_1 
        elif intersection_id == 2:
            frame = frame_2
        elif intersection_id == 3:
            frame = frame_3
        elif intersection_id == 4:
            frame = frame_4

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert the JPEG image to bytes
        frame_bytes = jpeg.tobytes()

        # Yield frame as part of a multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        
        # Wait for 1 second before processing the next frame
        time.sleep(1)

@app.route('/video_feed1')
def video_feed1():
    return Response(get_video_feed(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(get_video_feed(2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(get_video_feed(3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(get_video_feed(4),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def detect_vehicles(cap, intersection_id):
    global vehicle_count_1, vehicle_count_2, vehicle_count_3, vehicle_count_4
    global car_count_1, car_count_2, car_count_3, car_count_4
    global motocycle_count_1, motocycle_count_2, motocycle_count_3, motocycle_count_4
    global bus_count_1, bus_count_2, bus_count_3, bus_count_4
    global frame_1, frame_2, frame_3, frame_4

    while True:
        start_time = time.time()  # Record the start time of the frame processing
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Perform object detection
            results = model(frame)

            # Extract bounding boxes and labels
            vehicle_count = 0
            car_count = 0
            motocycle_count = 0
            bus_count = 0

            for result in results:
                for box in result.boxes:
                    # Assuming class 2 is for 'car', modify if different
                    if int(box.cls) == 2:
                        vehicle_count += 1
                        car_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Convert confidence score to a float
                        confidence = float(box.conf.item())
                        # Add label for the class with confidence score
                        label = f"Mobil: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    elif int(box.cls) == 3:
                        vehicle_count += 1
                        motocycle_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        # Convert confidence score to a float
                        confidence = float(box.conf.item())
                        # Add label for the class with confidence score
                        label = f"Motor: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    elif int(box.cls) == 1:
                        vehicle_count += 1
                        bus_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        # Convert confidence score to a float
                        confidence = float(box.conf.item())
                        # Add label for the class with confidence score
                        label = f"Bis: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Update global vehicle count
            if intersection_id == 1:
                vehicle_count_1 = vehicle_count
                car_count_1 = car_count
                motocycle_count_1 = motocycle_count
                bus_count_1 = bus_count
                frame_1 = frame
            elif intersection_id == 2:
                vehicle_count_2 = vehicle_count
                car_count_2 = car_count
                motocycle_count_2 = motocycle_count
                bus_count_2 = bus_count
                frame_2 = frame
            elif intersection_id == 3:
                vehicle_count_3 = vehicle_count
                car_count_3 = car_count
                motocycle_count_3 = motocycle_count
                bus_count_3 = bus_count
                frame_3 = frame
            elif intersection_id == 4:
                vehicle_count_4 = vehicle_count
                car_count_4 = car_count
                motocycle_count_4 = motocycle_count
                bus_count_4 = bus_count
                frame_4 = frame

        except Exception as e:
            print(f"Error during model prediction or processing: {e}")
            continue

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        # Sleep to ensure that the frame processing time is exactly 2 seconds
        if elapsed_time < 2.0:
            time.sleep(2.0 - elapsed_time)

    cap.release()

def sendToServer():
    global vehicle_count_1, vehicle_count_2, vehicle_count_3, vehicle_count_4
    while True:
        start_time = time.time()

        # Prepare the data to be sent
        data = {
            1 : vehicle_count_1,
            2 : vehicle_count_2,
            3 : vehicle_count_3,
            4 : vehicle_count_4,
        }

        # Send the data as a POST request to the server
        headers = {'Content-Type': 'application/json'}
        requests.post(server_url, data=json.dumps(data), headers=headers)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        # Sleep to ensure that the request time is exactly 2 seconds
        if elapsed_time < 2.0:
            time.sleep(2.0 - elapsed_time)

if __name__ == '__main__':
    # Start detection threads for all four intersections
    detection_thread_1 = Thread(target=detect_vehicles, args=(cap1, 1))
    detection_thread_2 = Thread(target=detect_vehicles, args=(cap2, 2))
    detection_thread_3 = Thread(target=detect_vehicles, args=(cap3, 3))
    detection_thread_4 = Thread(target=detect_vehicles, args=(cap4, 4))

    send = Thread(target=sendToServer, args=())

    detection_thread_1.start()
    detection_thread_2.start()
    detection_thread_3.start()
    detection_thread_4.start()
    send.start()

    app.run(host='0.0.0.0', port=7000)
