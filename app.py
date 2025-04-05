import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# Başlık
st.title("🧠 Yüz Değiştirme Uygulaması (Face Swap) - MediaPipe")

# MediaPipe yüz mesh modelini yükle
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Dosya yükleme
uploaded_image1 = st.file_uploader("🔼 Kaynak Yüz (Source Face) Görselini Yükleyin", type=["jpg", "jpeg", "png"])
uploaded_image2 = st.file_uploader("🔼 Hedef Yüz (Target Face) Görselini Yükleyin", type=["jpg", "jpeg", "png"])

def get_landmarks(img, face_mesh):
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                landmarks.append((x, y))
    return landmarks

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

if uploaded_image1 and uploaded_image2:
    image1 = Image.open(uploaded_image1).convert("RGB").resize((300, 300))
    image2 = Image.open(uploaded_image2).convert("RGB").resize((300, 300))

    img = np.array(image1)
    img2 = np.array(image2)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    mask = np.zeros_like(img_gray)
    img2_new_face = np.zeros_like(img2)

    # MediaPipe ile landmark'ları al
    landmarks_points = get_landmarks(img, face_mesh)
    landmarks_points2 = get_landmarks(img2, face_mesh)
    
    if len(landmarks_points) > 0 and len(landmarks_points2) > 0:
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = extract_index_nparray(np.where((points == pt1).all(axis=1)))
            index_pt2 = extract_index_nparray(np.where((points == pt2).all(axis=1)))
            index_pt3 = extract_index_nparray(np.where((points == pt3).all(axis=1)))

            if None not in (index_pt1, index_pt2, index_pt3):
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

        for triangle_index in indexes_triangles:
            tr1_pt1 = landmarks_points[triangle_index[0]]
            tr1_pt2 = landmarks_points[triangle_index[1]]
            tr1_pt3 = landmarks_points[triangle_index[2]]

            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y: y + h, x: x + w]

            cropped_tr1_mask = np.zeros((h, w), np.uint8)
            points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

            tr2_pt1 = landmarks_points2[triangle_index[0]]
            tr2_pt2 = landmarks_points2[triangle_index[1]]
            tr2_pt3 = landmarks_points2[triangle_index[2]]

            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            rect2 = cv2.boundingRect(triangle2)
            (x2, y2, w2, h2) = rect2

            cropped_tr2_mask = np.zeros((h2, w2), np.uint8)
            points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2],
                                [tr2_pt2[0] - x2, tr2_pt2[1] - y2],
                                [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)
            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2: x2 + w2]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_RGB2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area

        # Sonuç görüntüsünü oluştur
        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
        seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        st.image(seamlessclone, caption="🧬 Yüz Değiştirme Sonucu", channels="RGB")
    else:
        st.error("Bir veya her iki görselde yüz algılanamadı!")


