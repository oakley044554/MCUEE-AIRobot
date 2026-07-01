import cv2
import numpy as np
from ultralytics import YOLO
from pydobot import Dobot
import time

model = YOLO("yolo26s-seg.pt")
MTX = np.array([[1445.92178, 0.0, 646.279407],
                [0.0, 1448.39078, 328.581742],
                [0.0, 0.0, 1.0]])
DIST = np.array([[-0.032202, -0.02876593, -0.00931688, 0.0029928, 1.30470928]])

##### Step 1 #####
CAMERA_HEIGHT_CM = 19       # 相機高度（cm），請根據實際情況調整
CAMERA_PITCH_DEG = 27       # 相機俯仰角度（度），請根據實際情況調整


##### Step 2 #####
# ─── 關鍵：相機與機械手臂的相對位置校準 ───
# 請根據你相機擺放的位置調整這兩個數值（單位：mm）
# 假設相機架在 Dobot 正前方/後方，面對同一個方向：
X_OFFSET_MM = 150  # 相機原點距離 Dobot 底座中心的 X 軸前後距離
Y_OFFSET_MM = 230   # 相機原點距離 Dobot 底座中心的 Y 軸左右偏差

# 機器人連線設定
DOBOT_PORT = 'COM5'  # 請確認你的 COM Port

def get_dobot_coordinates(u, v, mtx, height_cm, pitch_deg):
    ray = np.linalg.inv(mtx).dot(np.array([u, v, 1.0]))

    theta = np.deg2rad(pitch_deg)
    normal = np.array([0, -np.cos(theta), -np.sin(theta)])
    denominator = np.dot(normal, ray)

    if denominator >= 0:
        return None

    s = -height_cm / denominator
    P_camera = s * ray 
   
    x_ground_cm = P_camera[0]
    y_ground_cm = P_camera[2] * np.cos(theta) - P_camera[1] * np.sin(theta)

    print(f"偵測到手機！相機座標: ({x_ground_cm}, {y_ground_cm})")

    ##### Step 3 #####
    ### 根據相機座標轉換為 Dobot 座標系統 ###
    dobot_x = (x_ground_cm * 10.0) + X_OFFSET_MM
    dobot_y = (y_ground_cm * 10.0) - Y_OFFSET_MM

    return dobot_x, dobot_y

try:
    print(f"正在連線至 Dobot ({DOBOT_PORT})...")
    device = Dobot(port=DOBOT_PORT, verbose=False)
    print("Dobot 連線成功！")
    # 先移動到一個安全的高位等待
    device.move_to(200, 0, 50, 0, wait=True)
except Exception as e:
    print(f"Dobot 連線失敗: {e}")
    device = None

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n系統就緒！")
print("指令說明：")
print("  - 畫面上會即時顯示預估的 Dobot X, Y 座標")
print("  - 偵測到手機時，按下鍵盤 'g' 鍵，手臂會移動去抓取該位置")
print("  - 按下 'q' 鍵退出程式")

target_robot_pos = None 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    display_frame = cv2.undistort(frame, MTX, DIST, None, MTX)
    results = model(display_frame)
    
    target_robot_pos = None  

    if results[0].masks is not None:
        for box, mask_data in zip(results[0].boxes, results[0].masks):
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            if class_name == "cell phone":
                contour = np.array(mask_data.xy[0], dtype=np.int32)
                if len(contour) == 0: continue

                bottom_point = contour[contour[:, 1].argmax()]
                bottom_x, bottom_y = bottom_point[0], bottom_point[1]

                coords = get_dobot_coordinates(bottom_x, bottom_y, MTX, CAMERA_HEIGHT_CM, CAMERA_PITCH_DEG)
                if coords is not None:
                    target_robot_pos = coords  
                    rx, ry = coords

                    cv2.polylines(display_frame, [contour], True, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(display_frame, (bottom_x, bottom_y), 6, (255, 0, 0), -1) # 藍色接地點
                    
                    info_text = f"Target Dobot -> X: {rx:.1f}, Y: {ry:.1f}"
                    cv2.putText(display_frame, info_text, (bottom_x + 10, bottom_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Vision_Controlled_Dobot", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        if device is not None and target_robot_pos is not None:
            tx, ty = target_robot_pos
            print(f"\n[Action] 收到抓取指令！目標 Dobot 座標: X={tx:.1f}, Y={ty:.1f}")
            
            ##### Step 4 #####
            try:
                print("1. 移動到物體上方高位 (Z=50)...")
                device.move_to(tx, ty, 50, 0, wait=True) 
                time.sleep(0.5)
                
                print("2. 降下到物體表面 (Z=-45，請根據你的桌面高度修正)...")
                device.move_to(tx+10, ty+10, -70, 0, wait=True) 
                
                print("3. 開啟吸盤...")
                device.suck(True)
                time.sleep(1.0)
                
                print("4. 抬起物體...")
                device.move_to(tx, ty, 50, 0, wait=True)
                
                print("5. 搬運至安全放置點...")
                device.move_to(200, -100, 50, 0, wait=True) 
                device.move_to(200, -100, -45, 0, wait=True)
                
                print("6. 釋放物品...")
                device.suck(False)
                time.sleep(0.5)
                
                # 回到初始高位等待下一次命令
                device.move_to(200, 0, 50, 0, wait=True)
                print("[Action] 抓取任務完成！")
                
            except Exception as arm_error:
                print(f"手臂執行動作時發生錯誤: arm_error")
        elif device is None:
            print("[Warning] 手臂未連線，無法執行抓取。")
        elif target_robot_pos is None:
            print("[Warning] 畫面上沒有偵測到 cell phone，找不到目標座標。")

    # 按下 'q' 鍵退出
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if device is not None:
    device.close()
    print("Dobot 連線已安全關閉。")