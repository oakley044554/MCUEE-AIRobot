import cv2
import numpy as np
from ultralytics import YOLO
from pydobot import Dobot
import time

# ==========================================================
# 1. 載入 AI 模型與跨實驗參數整合 (單位全數統一為 mm)
# ==========================================================
model = YOLO("yolo26s-seg.pt")

# Step 1 (Camera_Matrix.py) 實驗求得的相機內參矩陣與畸變係數
MTX = np.array([[1445.92178, 0.0, 646.279407],
                [0.0, 1448.39078, 328.581742],
                [0.0, 0.0, 1.0]])

DIST = np.array([[-0.032202, -0.02876593, -0.00931688, 0.0029928, 1.30470928]])

# 📐 單位統一：相機實體架設參數 (全數改為 mm)
CAMERA_HEIGHT_MM = 190.0     # 相機原點距離桌面的垂直高度：19 cm -> 190 mm
CAMERA_PITCH_DEG = 40.0      # 相機光軸下俯傾斜角度（度）

# 📐 單位統一：手眼外參校準（單位：mm）
X_OFFSET_MM = 270.0          # 相機原點到 Dobot 底座中心的 X 軸前後距離
Y_OFFSET_MM = 240.0          # 相機原點到 Dobot 底座中心的 Y 軸左右偏差

# 機械手臂連線通訊埠
DOBOT_PORT = 'COM5'         

# 全域變數，用來記錄最新一幀解算出來的相機地平面座標 (mm)
last_camera_coords_mm = None

# ==========================================================
# 2. 核心幾何演算法：單目反投影射線與地面(Z=0)平面求交點
# ==========================================================
def get_dobot_coordinates(u, v, mtx, height_mm, pitch_deg):
    """
    透過相機幾何逆矩陣將 2D 像素中心點 (u,v) 反投影成 3D 空間射線，
    直接計算該射線與桌面平面的交點，全程以毫米 (mm) 為單位計算與輸出。
    """
    global last_camera_coords_mm
    
    # 透過內參逆矩陣，將 2D 像素點轉為相機座標系下的 3D 虛擬幾何射線 (Ray)
    ray = np.linalg.inv(mtx).dot(np.array([u, v, 1.0]))

    # 將俯仰角轉為弧度，並建立桌面的法向量 (Normal Vector)
    theta = np.deg2rad(pitch_deg)
    normal = np.array([0, -np.cos(theta), -np.sin(theta)])
    
    # 計算射線與平面法向量的點積 (分母項)
    denominator = np.dot(normal, ray)

    if denominator >= 0:
        return None  # 異常處理：射線未朝向桌面

    # 求解比例因子 s (計算射線延伸多遠會直接撞擊到桌面，此處完全基於 mm)
    s = -height_mm / denominator
    P_camera = s * ray 
   
    # 解算物體在相機地面投影下的實際空間位置 (單位直接輸出：毫米 mm)
    x_ground_mm = P_camera[0]
    y_ground_mm = P_camera[2] * np.cos(theta) - P_camera[1] * np.sin(theta)

    # 紀錄給畫面 print 顯示用 (mm)
    last_camera_coords_mm = (x_ground_mm, y_ground_mm)

    # ─── 座標系轉換：全系統 mm 對齊，直接進行線性平移 ───
    dobot_x = x_ground_mm + X_OFFSET_MM
    dobot_y = y_ground_mm - Y_OFFSET_MM

    return dobot_x, dobot_y

# ==========================================================
# 3. 初始化機械手臂通訊連線
# ==========================================================
try:
    print(f"正在建立與 Dobot 機械手臂的連線 ({DOBOT_PORT})...")
    device = Dobot(port=DOBOT_PORT, verbose=False)
    print("Dobot 機械手臂連線成功！")
    device.move_to(200, 0, 50, 0, wait=True) # 安全初始位置 (mm)
except Exception as e:
    print(f"Dobot 連線失敗: {e}")
    device = None

# ==========================================================
# 4. 開啟感測器與即時推論主迴圈
# ==========================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   

print("\n全系統單位統一 mm 就緒！【雙座標系實時 mm 對照模式】已啟動。")

target_robot_pos = None 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 影像去畸變
    display_frame = cv2.undistort(frame, MTX, DIST, None, MTX)
    results = model(display_frame)
    target_robot_pos = None  
    last_camera_coords_mm = None 

    if results[0].masks is not None:
        for box, mask_data in zip(results[0].boxes, results[0].masks):
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            if class_name == "cell phone":
                contour = np.array(mask_data.xy[0], dtype=np.int32)
                if len(contour) == 0: continue

                # 利用 OpenCV Moments 計算幾何重心點 (Centroid)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # 計算座標 (全程 mm)
                coords = get_dobot_coordinates(center_x, center_y, MTX, CAMERA_HEIGHT_MM, CAMERA_PITCH_DEG)
                if coords is not None:
                    target_robot_pos = coords  
                    rx, ry = coords

                    # 繪製綠色實例分割邊緣輪廓
                    cv2.polylines(display_frame, [contour], True, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # 繪製中心十字標誌
                    cv2.circle(display_frame, (center_x, center_y), 6, (0, 0, 255), -1)
                    cv2.drawMarker(display_frame, (center_x, center_y), (255, 255, 255), 
                                   cv2.MARKER_CROSS, 12, 2)
                    
                    # ─── 🚀 畫面上即時印出 (print) 統一為 mm 的雙座標數值 ───
                    if last_camera_coords_mm is not None:
                        cx, cy = last_camera_coords_mm
                        # 1. 相機地平面座標 (單位: mm)
                        cam_text = f"Cam XY: ({cx:.1f}, {cy:.1f}) mm"
                        cv2.putText(display_frame, cam_text, (center_x + 15, center_y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2, cv2.LINE_AA)
                        
                        # 2. 手臂基座座標 (單位: mm)
                        robot_text = f"Arm XY: ({rx:.1f}, {ry:.1f}) mm"
                        cv2.putText(display_frame, robot_text, (center_x + 15, center_y + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Vision_Controlled_Dobot_Final", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # 自動吸取控制 (按下 'g' 鍵)
    if key == ord('g'):
        if device is not None and target_robot_pos is not None:
            tx, ty = target_robot_pos
            print(f"\n[Action] 執行抓取任務，目標手臂座標: X={tx:.1f} mm, Y={ty:.1f} mm")
            
            try:
                print("1. 手臂快速移動至物體中心上方安全點 (Z=50)...")
                device.move_to(tx, ty, 50, 0, wait=True) 
                time.sleep(0.5)
                
                print("2. 垂直下降接觸物體表面 (Z=-55)...")
                device.move_to(tx, ty, -55, 0, wait=True) 
                
                print("3. 啟動末端氣壓吸盤...")
                device.suck(True)
                time.sleep(1.0)
                
                print("4. 垂直抬起物件...")
                device.move_to(tx, ty, 50, 0, wait=True)
                
                print("5. 搬運至目標放置點...")
                device.move_to(200, -100, 50, 0, wait=True) 
                device.move_to(200, -100, -45, 0, wait=True)
                
                print("6. 釋放物品，重置手臂位置...")
                device.suck(False)
                time.sleep(0.5)
                
                device.move_to(200, 0, 50, 0, wait=True)
                print("[Action] 任務順利完成！\n")
                
            except Exception as arm_error:
                print(f"機械手臂執行動作錯誤: {arm_error}")
        elif device is None:
            print("[Warning] 手臂未成功連線。")
        elif target_robot_pos is None:
            print("[Warning] 視覺畫面中未鎖定目標，拒絕發送動作。")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if device is not None:
    device.close()
    print("Dobot 機械手臂通訊連線已安全斷開。")