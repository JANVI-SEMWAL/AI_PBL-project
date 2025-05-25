import cv2
import os

# === CONFIGURATION ===
video_path = r"C:\Users\Janvi\Downloads\VID-20250522-WA0015.mp4"
output_folder = 'extracted_frames'
frame_interval = 10  # Extract every 30th frame (~1 frame per second at 30 FPS)

# === CREATE OUTPUT FOLDER ===
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# === LOAD VIDEO ===
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

print("📽️ Starting frame extraction...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save every N-th frame
    if frame_count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
        print(f"✅ Saved {frame_filename}")

    frame_count += 1

cap.release()
print(f"\n🎉 Done! {saved_count} frames saved in '{output_folder}' folder.")
