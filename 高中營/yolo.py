import cv2
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame)
    print(results)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO26n Mask", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()