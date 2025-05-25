import tkinter as tk
from tkinter import ttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
from predict import predict_face, face_cascade

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        
        style = ttk.Style()
        style.configure("TButton", padding=10, font=('Helvetica', 12))
        style.configure("TLabel", font=('Helvetica', 12))
        
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=BOTH, expand=YES)
        
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Video Feed", padding="10")
        self.video_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 10))
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=BOTH, expand=YES)
        
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding="10")
        self.control_frame.pack(side=RIGHT, fill=Y, padx=(10, 0))
        
        self.status_label = ttk.Label(self.control_frame, text="Status: Ready", font=('Helvetica', 12))
        self.status_label.pack(pady=10)
        
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Camera",
            command=self.toggle_camera,
            style="primary.TButton"
        )
        self.start_button.pack(pady=10, fill=X)
        
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Recognition Results", padding="10")
        self.results_frame.pack(fill=X, pady=10)
        
        self.person_label = ttk.Label(self.results_frame, text="Person: -", font=('Helvetica', 12))
        self.person_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(self.results_frame, text="Confidence: -", font=('Helvetica', 12))
        self.confidence_label.pack(pady=5)
        
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
        self.person_label.config(text="Person: -")
        self.confidence_label.config(text="Confidence: -")
    
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
                
                # Update labels
                self.person_label.config(text=f"Person: {label}")
                self.confidence_label.config(text=f"Confidence: {conf*100:.1f}%")
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{label} ({conf*100:.1f}%)"
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