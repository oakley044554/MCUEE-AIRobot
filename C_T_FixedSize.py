import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. 參數設定區
# ==========================================
# 載入 YOLO 模型 (建議確認您的模型檔名，此處使用官方 Nano 版示意)
model = YOLO("yolo26s-seg.pt") 

# Logitech C270 真實內參矩陣與畸變係數 (您校正出來的精準數據)
MTX = np.array([[1445.92178, 0.0, 646.279407],
                [0.0, 1448.39078, 328.581742],
                [0.0, 0.0, 1.0]])
DIST = np.array([[-0.032202, -0.02876593, -0.00931688, 0.0029928, 1.30470928]])

# 手機的真實尺寸 (單位: 公分)
PHONE_REAL_WIDTH_CM = 7.75
PHONE_REAL_HEIGHT_CM = 15

# 繪圖設定 (讓畫面看起來專業點)
COLOR_CONTOUR = (0, 255, 0)   # 綠色輪廓
COLOR_CENTROID = (0, 0, 255)  # 紅色重心點
COLOR_TEXT = (0, 255, 255)    # 黃色文字
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==========================================
# 2. 輔助函式區
# ==========================================
def calculate_phone_distance(pixel_w, pixel_h):
    """使用焦距與實際尺寸計算距離"""
    if pixel_w <= 0 or pixel_h <= 0: return 0.0
    fx, fy = MTX[0][0], MTX[1][1]
    distance_w = (PHONE_REAL_WIDTH_CM * fx) / pixel_w
    distance_h = (PHONE_REAL_HEIGHT_CM * fy) / pixel_h
    # 取平均值降低誤差
    return (distance_w + distance_h) / 2.0

def get_contour_centroid(contour):
    """
    計算輪廓的多邊形重心 (利用 OpenCV Moments)
    """
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    return None

# ==========================================
# 3. 主程式執行區
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 務必設定解析度為校正時的 1280x720
W, H = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

print("開始實景 AR 測距... 畫面上將標註手機重心與距離。 按 'q' 結束。")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # [步驟 A] 影像去畸變：把畫面拉平，確保幾何計算正確
    # 我們將在這個去畸變後的「真實畫面」上繪圖
    display_frame = cv2.undistort(frame, MTX, DIST, None, MTX)

    # [步驟 B] YOLO 預測
    results = model(display_frame)
    
    # 確認有偵測到物件
    if results[0].masks is not None:
        # 配對 Bounding Box 與 Mask 資料
        for box, mask_data in zip(results[0].boxes, results[0].masks):
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            
            # 只針對「手機 (cell phone)」進行處理
            if class_name == "cell phone":
                # --- 1. 取得 Mask 輪廓 (Polygon) ---
                # mask_data.xy[0] 回傳的是已經對應回原圖解析度的輪廓點座標
                contour = np.array(mask_data.xy[0], dtype=np.int32)
                
                # 需要至少 3 個點才能構成輪廓
                if len(contour) < 3: continue

                # --- 2. 計算實體距離 (使用最小外接矩形處理傾斜) ---
                rect = cv2.minAreaRect(contour)
                (_, _), (w_px, h_px), _ = rect
                # 強制區分長短邊 (短邊為寬，長邊為高)
                if w_px > h_px: w_px, h_px = h_px, w_px
                
                distance = calculate_phone_distance(w_px, h_px)

                # --- 3. 計算並繪製重心與深度 ---
                centroid = get_contour_centroid(contour)
                if centroid:
                    cX, cY = centroid
                    
                    # 繪製物件輪廓線 (讓用戶知道程式抓到哪裡)
                    cv2.polylines(display_frame, [contour], True, COLOR_CONTOUR, 2, cv2.LINE_AA)
                    
                    # 繪製重心 (紅色小圓點)
                    cv2.circle(display_frame, (cX, cY), 6, COLOR_CENTROID, -1, cv2.LINE_AA)
                    
                    # 準備距離文字
                    text = f"{distance:.1f} cm"
                    font_scale = 0.8
                    thickness = 2
                    
                    # 獲取文字大小，以便居中繪製
                    (t_w, t_h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
                    
                    # 文字顯示在紅點下方約 20 像素的位置
                    text_x = cX - t_w // 2
                    text_y = cY + t_h + 20
                    
                    # 畫文字底色背景 (防止文字被背景顏色遮擋看不清楚)
                    cv2.rectangle(display_frame, (text_x - 5, cY + 15), 
                                  (text_x + t_w + 5, text_y + 5), (0, 0, 0), -1)

                    # 繪製黃色距離文字
                    cv2.putText(display_frame, text, (text_x, text_y), FONT, 
                                font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)

    # 顯示實景畫面
    cv2.imshow("C_T_FixedSize", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()