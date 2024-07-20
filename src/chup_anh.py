import cv2
import time

cap = cv2.VideoCapture(0)
i=1
name = 'Linh'

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit (0)
    if i==15:
        break
    cv2.imwrite(f'./Dataset/FaceData/raw/Sang/{name}.{i}.jpg', frame)
    i+=1
    time.sleep(1)


cap.release()
cv2.destroyAllWindows()


# video_path = "/home/linh/Downloads/LanNgoc.mp4"
# vidcap = cv2.VideoCapture(video_path)

# fps = vidcap.get(cv2.CAP_PROP_FPS)
# delay = 1 / fps  # Lấy mẫu mỗi giây

# count = 0
# while True:
#     success, frame = vidcap.read()
#     count += 1
#     if count % 1:
#         continue
#     if not success:
#         print("Đã đọc hết video")
#         break

#     # Lưu frame
#     image_path = f"/home/linh/Downloads/Face_Recognize/Dataset/FaceData/raw/LanNgoc/{count}.LanNgoc.jpg"
#     cv2.imwrite(image_path, frame)


#     # Chờ `delay` giây trước khi đọc frame tiếp theo
#     time.sleep(delay)

# vidcap.release()