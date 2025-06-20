# Memory Recaller Project

The Memory Recaller is an AI-based system designed to assist individuals, especially Alzheimer's patients, by recognizing faces of their relatives and recalling important details associated with them.

## Project Objective

To help patients recognize their relatives using facial recognition. Once a known face is detected, the system retrieves stored information such as name, relation, birthday, and memories, and reads it aloud using a text-to-speech engine.

## Features

- Face detection using OpenCV's Haar cascades.
- Facial recognition using a custom-trained Convolutional Neural Network (CNN).
- Automatic retrieval of relative-specific details from JSON files.
- Text-to-speech output using the pyttsx3 library.
- GUI interface built using Tkinter and ttkbootstrap.
