import cv2
import numpy as np
from ultralytics import YOLO
from pydobot import Dobot
import time

# ==========================================================
# 1. 載入 AI 模型與跨實驗參數整合
# ==========================================================
# 載入最新版 YOLO26s 實例分割模型
model = YOLO("yolo26s-seg.pt")

# 帶入 Step 1 (Camera_Matrix.py) 實驗求得的相機內參矩陣與畸變係數
MTX = np.array([[1445.92178, 0.0, 646.279407],
                [0.0, 1448.39078, 328.581742],
                [0.0, 0.0, 1.0]])

DIST = np.array([[-0.032202, -0.02876593, -0.00931688, 0.0029928, 1.30470928]])

# 相機實體架設參數（請依據實體環境量測並微調）
CAMERA_HEIGHT_CM = 19       # 相機距離桌面的垂直高度（cm）
CAMERA_PITCH_DEG = 40       # 相機下俯傾斜角度（度）

# ─── 手眼外參校準：相機與機械手臂基座的相對位移 ───
X_OFFSET_MM = 270           # 相機原點到 Dobot 底座中心的 X 軸前後距離（mm）
Y_OFFSET_MM = 240           # 相機原點到 Dobot 底座中心的 Y 軸左右偏差（mm）

# 機械手臂連線通訊埠
DOBOT_PORT = 'COM5'         # 請確認你的電腦裝置管理員顯示的 COM Port

# ==========================================================
# 2. 核心幾何演算法：單目反投影射線與平面求交點
# ==========================================================
def get_dobot_coordinates(u, v, mtx, height_cm, pitch_deg):
    """
    透過相機幾何逆矩陣將 2D 像素點 (u,v) 反投影成 3D 空間射線，
    並計算該射線與桌面的交點，最終轉換為機械手臂基座標。
    """
    # 透過內參逆矩陣，將像素點轉為相機座標系下的 3D 虛擬幾何射線 (Ray)
    ray = np.linalg.inv(mtx).dot(np.array([u, v, 1.0]))

    # 將俯仰角轉為弧度，並建立桌面的法向量 (Normal Vector)
    theta = np.deg2rad(pitch_deg)
    normal = np.array([0, -np.cos(theta), -np.sin(theta)])
    
    # 計算射線與平面法向量的點積 (分母項)
    denominator = np.dot(normal, ray)

    if denominator >= 0:
        return None  # 射線未朝向桌面（異常狀況處理）

    # 求解比例因子 s (計算射線延伸多遠會撞擊到桌面)
    s = -height_cm / denominator
    P_camera = s * ray 
   
    # 解算物體在相機地面投影下的實際位置 (單位：公分)
    x_ground_cm = P_camera[0]
    y_ground_cm = P_camera[2] * np.cos(theta) - P_camera[1] * np.sin(theta)

    print(f"視覺解算成功 -> 相機地平面座標: (X: {x_ground_cm:.2f} cm, Y: {y_ground_cm:.2f} cm)")

    # ─── 座標系轉換：將相機公分座標平移至 Dobot 的毫米基座標系 ───
    dobot_x = (x_ground_cm * 10.0) + X_OFFSET_MM
    dobot_y = (y_ground_cm * 10.0) - Y_OFFSET_MM

    return dobot_x, dobot_y

# ==========================================================
# 3. 初始化機械手臂連線
# ==========================================================
try:
    print(f"正在建立與 Dobot 機械手臂的連線 ({DOBOT_PORT})...")
    device = Dobot(port=DOBOT_PORT, verbose=False)
    print("Dobot 機械手臂連線成功！")
    # 安全機制：初始時先移動到上方高位等待
    device.move_to(200, 0, 50, 0, wait=True)
except Exception as e:
    print(f"Dobot 連線失敗: {e}")
    device = None

# ==========================================================
# 4. 開啟影像感測器與即時推論主迴圈
# ==========================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 設定視訊寬度解析度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 設定視訊高度解析度

print("\n系統全功能就緒！")
print("操作說明：")
print("  - 視窗將即時渲染目標預估之 Dobot X, Y 實體座標")
print("  - 當辨識到手機時，按下鍵盤 'g' 鍵，手臂會自動執行全套自動抓取與搬運任務")
print("  - 按下鍵盤 'q' 鍵安全退出系統")

target_robot_pos = None 

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        break

    # 利用 Step 1 的校正矩陣進行即時影像去畸變 (Undistortion)，拉直邊緣
    display_frame = cv2.undistort(frame, MTX, DIST, None, MTX)
    
    # 送入 YOLO26s-seg 進行物件推論
    results = model(display_frame)
    target_robot_pos = None  

    # 確保畫面中有成功提取出實例分割的 Mask 遮罩
    if results[0].masks is not None:
        for box, mask_data in zip(results[0].boxes, results[0].masks):
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            # 鎖定目標類別：手機 (cell phone)
            if class_name == "cell phone":
                # 提取實例分割的二維輪廓點
                contour = np.array(mask_data.xy[0], dtype=np.int32)
                if len(contour) == 0: 
                    continue

                # 🚀 關鍵技術：尋找物件的最下端接地點 (v 座標最大者)，精準定位實體觸地面
                bottom_point = contour[contour[:, 1].argmax()]
                bottom_x, bottom_y = bottom_point[0], bottom_point[1]

                # 調用反投影幾何演算法，計算機械手臂目標毫米座標
                coords = get_dobot_coordinates(bottom_x, bottom_y, MTX, CAMERA_HEIGHT_CM, CAMERA_PITCH_DEG)
                if coords is not None:
                    target_robot_pos = coords  
                    rx, ry = coords

                    # 影像渲染：繪製綠色實例分割輪廓與藍色接地點
                    cv2.polylines(display_frame, [contour], True, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.circle(display_frame, (bottom_x, bottom_y), 6, (255, 0, 0), -1) 
                    
                    # 畫面即時秀出預估的手臂座標位置
                    info_text = f"Target Dobot -> X: {rx:.1f}, Y: {ry:.1f}"
                    cv2.putText(display_frame, info_text, (bottom_x + 10, bottom_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # 顯示主監控視窗
    cv2.imshow("Vision_Controlled_Dobot", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # ─── 自動抓取觸發控制事件 (按下 'g' 鍵) ───
    if key == ord('g'):
        if device is not None and target_robot_pos is not None:
            tx, ty = target_robot_pos
            print(f"\n[Action] 收到視覺抓取指令！傳送絕對目標座標: X={tx:.1f}, Y={ty:.1f}")
            
            try:
                # 階段 1：移動到目標物正上方 50mm 高位（安全緩衝，並加入補償偏移量）
                print("1. 手臂快速移動至物體上方安全點 (Z=50)...")
                device.move_to(tx + 50, ty + 20, 50, 0, wait=True) 
                time.sleep(0.5)
                
                # 階段 2：垂直降落至物體表面進行抓取位置接觸
                print("2. 垂直下降至物體表面 (Z=-60)...")
                device.move_to(tx + 50, ty + 20, -60, 0, wait=True) 
                
                # 階段 3：驅動末端吸盤執行氣壓抓取
                print("3. 啟動末端氣壓吸盤...")
                device.suck(True)
                time.sleep(1.0)
                
                # 階段 4：垂直抬起物件（防止與周圍碰撞）
                print("4. 成功抓取，垂直抬起物件...")
                device.move_to(tx, ty, 50, 0, wait=True)
                
                # 階段 5：搬運至預設的安全放置點並降落
                print("5. 執行路徑規劃，搬運至安全放置點...")
                device.move_to(200, -100, 50, 0, wait=True) 
                device.move_to(200, -100, -45, 0, wait=True)
                
                # 階段 6：釋放物品並重置手臂位置
                print("6. 關閉吸盤，安全釋放物品...")
                device.suck(False)
                time.sleep(0.5)
                
                # 回到初始高位等待下一次視覺命令
                device.move_to(200, 0, 50, 0, wait=True)
                print("[Action] 自動抓取與搬運任務順利完成！\n")
                
            except Exception as arm_error:
                print(f"機械手臂執行動作時發生非預期錯誤: {arm_error}")
        elif device is None:
            print("[Warning] 手臂未成功連線，無法發送控制指令。")
        elif target_robot_pos is None:
            print("[Warning] 視覺畫面上未鎖定 cell phone，無法定位目標。")

    # 按下 'q' 鍵退出程式
    elif key == ord('q'):
        break

# 清放硬體資源
cap.release()
cv2.destroyAllWindows()

if device is not None:
    device.close()
    print("Dobot 機械手臂通訊連線已安全關閉。")