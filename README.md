#  Office Activity Tracking using YOLOv8 + DeepSORT

This project provides an **automated office activity tracking system** using **YOLOv8** for detection and **DeepSORT** for multi-object tracking.  
Each employee is detected, tracked across frames, and classified into one of six activity states.  
The system calculates **working vs. non-working time**, logs it in a CSV file, and saves annotated videos.

---

##  Features

⬤ Real-time activity detection using **YOLOv8**  
⬤ Consistent employee tracking with **DeepSORT**  
⬤ Calculates **total working and idle time** for each employee  
⬤ Saves activity summary as **CSV logs**  
⬤ Exports **annotated output video** with live tracking  
⬤ **Color-coded bounding boxes:**
- 🟢 **Green** → *Working*
- 🔴 **Red** → *All other classes (Not Available, Not Working, Roaming, Sleeping, Talking.)*

---
✅
##  Activity Classes

| Class ID | Activity Label | Description                |
|----------|----------------|----------------------------|
|     0    | Not Available  | No valid detection         |
|     1    | Not Working    | Idle or not engaged        |
|     2    | Roaming        | Moving without active work |
|     3    | Sleeping       | Inactive or resting        |
|     4    | Talking        | Engaged in conversation    |
|     5    | Working        | Actively performing tasks  |*(shown in green)*


