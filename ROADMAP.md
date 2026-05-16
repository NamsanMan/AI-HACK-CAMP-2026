\# Roadmap



\## Goal

Real-time video call deepfake detection plugin prototype.



\## Core Pipeline

Video frame

→ Face detection

→ Landmark-based ROI extraction

→ rPPG temporal branch

→ Visual artifact branch

→ Fusion classifier

→ Risk score



\## Milestones



\### Phase 1. Real-time Demo Baseline

\- Webcam/video input

\- Face detection

\- Landmark-based cheek/forehead ROI visualization

\- Temporal RGB buffer

\- FPS display

\- Random-weight risk score output



\### Phase 2. Model Components

\- rPPG TCN branch

\- Artifact CNN branch

\- Fusion classifier



\### Phase 3. Training

\- UBFC-rPPG for rPPG branch

\- FaceForensics++ for artifact branch

\- Celeb-DF-v2 for evaluation



\### Phase 4. Chrome Extension Prototype

\- Screen/tab capture

\- Frame extraction

\- Real-time inference

\- Warning UI



\## Priority

The first priority is a working real-time demo, not final model accuracy.

