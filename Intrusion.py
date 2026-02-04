import cv2
import numpy as np
import time
import csv
import winsound
from ultralytics import YOLO
import threading
from datetime import datetime

# Initialize variables for region drawing
regions = []
drawing = False
ix, iy = -1, -1

class TrackedPerson:
    def __init__(self, id, startX, startY, endX, endY):
        self.id = id
        self.bbox = (startX, startY, endX, endY)
        self.last_seen = time.time()
        self.enter_time = None
        self.exit_time = None
        self.in_region = False
        self.current_region = None
        self.center = ((startX + endX) // 2, (startY + endY) // 2)

    def update(self, startX, startY, endX, endY):
        self.bbox = (startX, startY, endX, endY)
        self.last_seen = time.time()
        self.center = ((startX + endX) // 2, (startY + endY) // 2)

def draw_region(event, x, y, flags, param):
    global ix, iy, drawing, regions, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Frame', frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        regions.append({"x1": min(ix, x), "y1": min(iy, y), "x2": max(ix, x), "y2": max(iy, y)})
        cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('Frame', frame)

alert_playing = False
alert_lock = threading.Lock()
last_alert_time = 0
alert_cooldown = 7  # seconds
alert_message = ""
alert_start_time = 0

def alert(message):
    global alert_message, alert_start_time
    print(message)
    winsound.Beep(1200, 600)
    alert_message = message
    alert_start_time = time.time()

# Open video capture
video_source = "video.mp4" # Change to "0" for webcam
cap = cv2.VideoCapture(video_source if video_source != "0" else 0)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_region)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

output_csv = 'out.csv'

def get_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)
    
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def is_new_detection(startX, startY, endX, endY, tracked_persons):
    current_time = time.time()
    center = ((startX + endX) // 2, (startY + endY) // 2)
    
    for person in tracked_persons:
        iou = get_iou((startX, startY, endX, endY), person.bbox)
        if iou > 0.3:  # Adjust this threshold as needed
            if current_time - person.last_seen < 5:  # Consider the person the same if seen within 5 seconds
                person.update(startX, startY, endX, endY)
                return False, person.id
        
        # Check if the centers are close
        distance = np.sqrt((center[0] - person.center[0])**2 + (center[1] - person.center[1])**2)
        if distance < 50:  # Adjust this threshold as needed
            if current_time - person.last_seen < 5:
                person.update(startX, startY, endX, endY)
                return False, person.id
    
    return True, None

# Open CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Person', 'Region', 'Start Time', 'End Time', 'Duration', 'System Timestamp'])

    # Draw regions
    print("Draw regions on the frame. You have 30 seconds to draw up to 5 regions.")
    start_time = time.time()
    max_regions = 5
    drawing_time_limit = 30  # seconds

    person_counter = 0
    frame_num = 0
    tracked_persons = []

    while len(regions) < max_regions and time.time() - start_time < drawing_time_limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = time.time() - start_time

        # Perform detection with YOLOv8
        results = model(frame, conf=0.5)  # Increased confidence threshold

        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # class 0 is person in COCO dataset
                    startX, startY, endX, endY = box.xyxy[0].cpu().numpy().astype(int)

                    if (endX - startX) * (endY - startY) < 1000:  # Increased size threshold
                        continue

                    is_new, existing_id = is_new_detection(startX, startY, endX, endY, tracked_persons)
                    if is_new:
                        person_id = f"person_{person_counter}"
                        person_counter += 1
                        tracked_persons.append(TrackedPerson(person_id, startX, startY, endX, endY))
                        tracked_person = next(p for p in tracked_persons if p.id == person_id)
                        tracked_person.enter_time = timestamp
                    else:
                        person_id = existing_id
                        tracked_person = next(p for p in tracked_persons if p.id == person_id)

                    for region_idx, region in enumerate(regions):
                        x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
                        area_threshold = (x2 - x1) * (y2 - y1) * 0.5  # 50% area threshold

                        person_area = (endX - startX) * (endY - startY)
                        if person_area < area_threshold:
                            continue

                        if (startX < x2 and endX > x1 and startY < y2 and endY > y1):
                            if is_new:
                                tracked_person.enter_time = timestamp
                                tracked_person.in_region = True

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            if time.time() - last_alert_time > alert_cooldown:
                                alert("ALERT: Intrusion detected!")
                                last_alert_time = time.time()

        # Remove old tracked persons
        current_time = time.time()
        tracked_persons = [p for p in tracked_persons if current_time - p.last_seen < 5]

        for region in regions:
            cv2.rectangle(frame, (region['x1'], region['y1']), (region['x2'], region['y2']), (0, 255, 0), 2)

        if alert_message and time.time() - alert_start_time < 3:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Drawing phase complete. {len(regions)} regions drawn.")

    time.sleep(2)

    # Reset person counter and tracked persons list
    person_counter = 0
    tracked_persons = []

    # Start the detection phase
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = time.time() - start_time
        system_timestamp = datetime.now().strftime("%H:%M:%S")

        results = model(frame, conf=0.5)  # Increased confidence threshold

        for result in results:
            for box in result.boxes:
                if box.cls == 0:
                    startX, startY, endX, endY = box.xyxy[0].cpu().numpy().astype(int)

                    if (endX - startX) * (endY - startY) < 1000:  # Increased size threshold
                        continue

                    is_new, existing_id = is_new_detection(startX, startY, endX, endY, tracked_persons)
                    if is_new:
                        person_id = f"person_{person_counter}"
                        person_counter += 1
                        tracked_persons.append(TrackedPerson(person_id, startX, startY, endX, endY))
                        tracked_person = next(p for p in tracked_persons if p.id == person_id)
                        tracked_person.enter_time = timestamp
                    else:
                        person_id = existing_id
                        tracked_person = next(p for p in tracked_persons if p.id == person_id)

                    for region_idx, region in enumerate(regions):
                        x1, y1, x2, y2 = region['x1'], region['y1'], region['x2'], region['y2']
                        area_threshold = (x2 - x1) * (y2 - y1) * 0.5

                        person_area = (endX - startX) * (endY - startY)
                        if person_area < area_threshold:
                            continue

                        if (startX < x2 and endX > x1 and startY < y2 and endY > y1):
                            if tracked_person.current_region is None or tracked_person.current_region != region_idx:
                                if tracked_person.in_region:
                                    tracked_person.exit_time = timestamp
                                    duration = int(tracked_person.exit_time - tracked_person.enter_time)
                                    if duration > 0:
                                        writer.writerow([person_id, tracked_person.current_region, 
                                                         int(tracked_person.enter_time), 
                                                         int(tracked_person.exit_time), 
                                                         duration, system_timestamp])
                                
                                tracked_person.enter_time = timestamp
                                tracked_person.current_region = region_idx
                                tracked_person.in_region = True

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            if time.time() - last_alert_time > alert_cooldown:
                                alert("ALERT: Intrusion detected!")
                                last_alert_time = time.time()
                        else:
                            if tracked_person.in_region and tracked_person.current_region == region_idx:
                                tracked_person.exit_time = timestamp
                                duration = int(tracked_person.exit_time - tracked_person.enter_time)
                                if duration > 0:
                                    writer.writerow([person_id, region_idx, 
                                                     int(tracked_person.enter_time), 
                                                     int(tracked_person.exit_time), 
                                                     duration, system_timestamp])
                                tracked_person.in_region = False
                                tracked_person.current_region = None
                                tracked_person.enter_time = None

        current_time = time.time()
        tracked_persons = [p for p in tracked_persons if current_time - p.last_seen < 5]

        for region in regions:
            cv2.rectangle(frame, (region['x1'], region['y1']), (region['x2'], region['y2']), (0, 255, 0), 2)

        if alert_message and time.time() - alert_start_time < 3:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()