import numpy as np
import cv2
import glob

# ==========================================
# 1. 參數設定 (請根據你的實際情況修改)
# ==========================================
# 假設你的方塊是 7x10，內角點就是 6x9。這裡寬高順序可以互換，但要跟你的擺放一致
CHECKERBOARD = (9, 6) 
# 每個格子的真實尺寸 (單位：公分)。設定為 2.5，算出來的距離單位就會直接是公分！
SQUARE_SIZE = 2.5 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ==========================================
# 2. 準備真實世界的 3D 座標點 (Object Points)
# ==========================================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# 用來儲存所有圖片對應的 3D 點與 2D 像素點
objpoints = [] # 真實世界的 3D 點
imgpoints = [] # 圖片上的 2D 像素點

# ==========================================
# 3. 讀取照片並尋找角點
# ==========================================
# 讀取 images 資料夾下所有的 jpg 照片
images = glob.glob('images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 尋找棋盤格內角點
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        objpoints.append(objp)
        
        # 提高角點的精確度
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100) # 顯示 0.1 秒

cv2.destroyAllWindows()

# ==========================================
# 4. 進行相機校正
# ==========================================
print("正在計算相機參數...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n--- 校正結果 ---")
print("相機內參矩陣 (Camera Matrix):\n", mtx)
print("\n畸變係數 (Distortion Coefficients):\n", dist)