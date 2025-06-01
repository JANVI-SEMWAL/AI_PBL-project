import os
import warnings
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
from predict import predict_face, face_cascade
import json

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        
        # Load relative details
        with open('relative_details.json', 'r') as f:
            self.relative_details = json.load(f)
        
        style = ttk.Style()
        style.configure("TButton", padding=10, font=('Helvetica', 12))
        style.configure("TLabel", font=('Helvetica', 12))
        
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=BOTH, expand=YES)
        
        # Left side - Video feed
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Video Feed", padding="10")
        self.video_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=BOTH, expand=YES)
        
        # Right side - Controls and Details
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=RIGHT, fill=Y, padx=(10, 0))
        
        # Controls section
        self.control_frame = ttk.LabelFrame(self.right_frame, text="Controls", padding="10")
        self.control_frame.pack(fill=X, pady=(0, 10))
        
        self.status_label = ttk.Label(self.control_frame, text="Status: Ready", font=('Helvetica', 12))
        self.status_label.pack(pady=10)
        
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Camera",
            command=self.toggle_camera,
            style="primary.TButton"
        )
        self.start_button.pack(pady=10, fill=X)
        
        # Recognition Results section
        self.results_frame = ttk.LabelFrame(self.right_frame, text="Recognition Results", padding="10")
        self.results_frame.pack(fill=X, pady=(0, 10))
        
        self.person_label = ttk.Label(self.results_frame, text="Person: -", font=('Helvetica', 12))
        self.person_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(self.results_frame, text="Confidence: -", font=('Helvetica', 12))
        self.confidence_label.pack(pady=5)
        
        # Relative Details section
        self.details_frame = ttk.LabelFrame(self.right_frame, text="Relative Details", padding="10")
        self.details_frame.pack(fill=X, pady=(0, 10))
        
        # Create a frame for details with a scrollbar
        self.details_container = ttk.Frame(self.details_frame)
        self.details_container.pack(fill=BOTH, expand=YES)
        
        # Create a canvas and scrollbar
        self.details_canvas = tk.Canvas(self.details_container)
        self.scrollbar = ttk.Scrollbar(self.details_container, orient="vertical", command=self.details_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.details_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.details_canvas.configure(scrollregion=self.details_canvas.bbox("all"))
        )
        
        self.details_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.details_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.details_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Add labels for relative details in the scrollable frame
        self.patient_name_label = ttk.Label(self.scrollable_frame, text="Patient: -", font=('Helvetica', 12))
        self.patient_name_label.pack(pady=5, anchor="w")
        
        self.relative_name_label = ttk.Label(self.scrollable_frame, text="Relative: -", font=('Helvetica', 12))
        self.relative_name_label.pack(pady=5, anchor="w")
        
        self.relation_label = ttk.Label(self.scrollable_frame, text="Relation: -", font=('Helvetica', 12))
        self.relation_label.pack(pady=5, anchor="w")
        
        self.extra_info_label = ttk.Label(self.scrollable_frame, text="Extra Info: -", font=('Helvetica', 12))
        self.extra_info_label.pack(pady=5, anchor="w")
        
        self.is_running = False
        self.cap = None
        self.thread = None
        
    def toggle_camera(self):
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Status: Error opening camera")
            return
        
        self.is_running = True
        self.start_button.config(text="Stop Camera")
        self.status_label.config(text="Status: Running")
        self.thread = threading.Thread(target=self.update_frame)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.start_button.config(text="Start Camera")
        self.status_label.config(text="Status: Stopped")
        self.video_label.config(image='')
        self.clear_labels()
    
    def clear_labels(self):
        self.person_label.config(text="Person: -")
        self.confidence_label.config(text="Confidence: -")
        self.patient_name_label.config(text="Patient: -")
        self.relative_name_label.config(text="Relative: -")
        self.relation_label.config(text="Relation: -")
        self.extra_info_label.config(text="Extra Info: -")
    
    def update_frame(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert to grayscale for face detection required for haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                label, conf = predict_face(face_img)
                
                # Check if the label exists in our dataset
                if label in self.relative_details:
                    details = self.relative_details[label]
                    # Update labels with details
                    self.person_label.config(text=f"Person: {label}")
                    self.confidence_label.config(text=f"Confidence: {conf*100:.1f}%")
                    self.patient_name_label.config(text=f"Patient: {details['patient_name']}")
                    self.relative_name_label.config(text=f"Relative: {details['relative_name']}")
                    self.relation_label.config(text=f"Relation: {details['relation']}")
                    self.extra_info_label.config(text=f"Extra Info: {details['extra_info']}")
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"{label} ({conf*100:.1f}%)"
                else:
                    # Unknown person
                    self.clear_labels()
                    self.person_label.config(text="Person: Unknown")
                    self.confidence_label.config(text=f"Confidence: {conf*100:.1f}%")
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    text = f"Unknown ({conf*100:.1f}%)"
                
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 0), 2)
            
            # Convert frame to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            self.root.after(10)
    
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() 