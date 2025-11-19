import cv2
import numpy as np
from pathlib import Path
import subprocess
import urllib.request

LABELS = {
    "face": "Oblicej",
    "eye": "Oko",
    "smile": "Usmev"
}
IMAGE_INSTRUCTIONS_TEXT = "s = ulozit, q/ESC = konec"
WEBCAM_INSTRUCTIONS_TEXT = "s = ulozit, g = bryle, q/ESC = konec"

def download_cascades():
    cascade_dir = Path("haarcascades")
    cascade_dir.mkdir(exist_ok=True)
    cascades = {
        "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
        "haarcascade_smile.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml"
    }
    for name, url in cascades.items():
        path = cascade_dir / name
        if not path.exists():
            print(f"Stahuji {name}...")
            urllib.request.urlretrieve(url, path)
            print(f"Stazeno do {path}")
    return cascade_dir

def get_color_name(rgb):
    colors = {
        "Cervena": (255, 0, 0),
        "Modra": (0, 0, 255),
        "Zelena": (0, 255, 0),
        "Hneda": (150, 100, 50),
        "Modra svetla": (100, 150, 200),
        "Zelena tmava": (90, 170, 100),
        "Seda": (130, 130, 140),
        "Oriskova": (120, 100, 40),
        "Jantarova": (255, 191, 0),
        "Zluta": (255, 255, 0),
        "Fialova": (128, 0, 128),
        "Ruzova": (255, 192, 203),
        "Oranzova": (255, 165, 0),
        "Tyrkysova": (64, 224, 208),
        "Bezova": (245, 245, 220),
        "Vinova": (128, 0, 32),
        "Indigo": (75, 0, 130),
        "Okrova": (204, 119, 34),
        "Limetka": (50, 205, 50),
        "Azurova": (0, 127, 255)
    }
    return min(colors.keys(), key=lambda color: np.linalg.norm(np.array(colors[color]) - np.array(rgb)))

def draw_instructions(image, mode="image"):
    text = WEBCAM_INSTRUCTIONS_TEXT if mode == "webcam" else IMAGE_INSTRUCTIONS_TEXT
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def is_overlapping(new_rect, existing_rects):
    x, y, w, h = new_rect
    return any(not (x > ex + ew or x + w < ex or y > ey + eh or y + h < ey) for (ex, ey, ew, eh) in existing_rects)

def detect_face_eyes_smile(image, cascade_dir):
    face_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))
    smile_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_smile.xml"))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, LABELS["face"], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray[:h//2, :], 1.1, 3, minSize=(20, 20))
        eye_regions = []
        
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(roi_color, LABELS["eye"], (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            eye_regions.append((ex, ey, ew, eh))
            iris_roi = roi_color[ey:ey+eh, ex:ex+ew]
            if iris_roi.size > 0:
                avg_color = np.mean(iris_roi, axis=(0, 1))
                iris_color = tuple(map(int, avg_color))
                cv2.circle(roi_color, (ex + ew // 2, ey + eh // 2), min(ew, eh) // 4, iris_color, 2)
                color_text = get_color_name(iris_color)
                cv2.putText(roi_color, color_text, (ex + ew // 2 - 30, ey + eh // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        smiles = smile_cascade.detectMultiScale(roi_gray[h//2:, :], 1.7, 25, minSize=(25, 25))
        detected_smiles = []
        
        for (sx, sy, sw, sh) in smiles:
            sy += h // 2
            new_smile = (sx, sy, sw, sh)
            if not is_overlapping(new_smile, detected_smiles):
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                cv2.putText(roi_color, LABELS["smile"], (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                detected_smiles.append(new_smile)
    
    return image

def select_image_file():
    try:
        script = 'POSIX path of (choose file with prompt "Vyberte obrazek" of type {"public.image"})'
        result = subprocess.check_output(['osascript', '-e', script])
        file_path = result.decode('utf-8').strip()
        return file_path if file_path else None
    except subprocess.CalledProcessError:
        return None

def process_image(cascade_dir):
    image_path = select_image_file()
    if not image_path:
        print("Zadny obrazek nebyl vybran.")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Nelze nahrat obrazek: {image_path}")
        return
    
    processed_img = detect_face_eyes_smile(img, cascade_dir)
    draw_instructions(processed_img, mode="image")
    cv2.imshow('OpenCV - Obrazek', processed_img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("output_image.jpg", processed_img)
            print("Vysledek ulozen jako output_image.jpg")
        elif key in [27, ord('q')]:
            break
    
    cv2.destroyAllWindows()

def process_webcam(cascade_dir):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nelze otevrit webkameru!")
        return
    
    show_glasses = False
    detect_faces = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nelze ziskat snimek z webkamery!")
            break
        
        if detect_faces:
            processed_frame = detect_face_eyes_smile(frame, cascade_dir)
        else:
            processed_frame = frame
        
        if show_glasses:
            detect_faces = False
            processed_frame = apply_party_accessories(processed_frame, cascade_dir, show_glasses)
        else:
            detect_faces = True
        
        draw_instructions(processed_frame, mode="webcam")
        cv2.imshow('OpenCV - Webkamera', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            show_glasses = not show_glasses
            print(f"Bryle {'zapnuty' if show_glasses else 'vypnuty'}")
        elif key == ord('s'):
            cv2.imwrite("output_image.jpg", processed_frame)
            print("Snimek ulozen jako output_image.jpg")
        elif key in [27, ord('q')]:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def apply_party_accessories(image, cascade_dir, show_glasses=True):
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    glasses_img = cv2.imread(str(assets_dir / "glasses.png"), cv2.IMREAD_UNCHANGED)
    
    if glasses_img is None and show_glasses:
        print("Nelze nacist obrazky bryli!")
        return image
    
    glasses_has_alpha = glasses_img is not None and glasses_img.shape[2] == 4

    face_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    result_image = image.copy()
    
    for (x, y, w, h) in faces:
        if show_glasses and glasses_img is not None:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            
            if len(eyes) >= 2:
                eyes = sorted(eyes, key=lambda e: e[1])[:2]
                eyes = sorted(eyes, key=lambda e: e[0])
                
                left_eye, right_eye = eyes
                eye_center_y = y + (left_eye[1] + right_eye[1]) // 2
                
                glasses_width = int(w * 0.9)
                glasses_height = int(glasses_img.shape[0] * glasses_width / glasses_img.shape[1])
                glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
                
                glasses_x = x + w // 2 - glasses_width // 2
                glasses_y = eye_center_y - glasses_height // 2
                
                if glasses_y < 0:
                    glasses_y = 0
                if glasses_x < 0:
                    glasses_x = 0
                
                glasses_h = min(glasses_height, result_image.shape[0] - glasses_y)
                glasses_w = min(glasses_width, result_image.shape[1] - glasses_x)
                
                if glasses_h > 0 and glasses_w > 0:
                    glasses_cropped = glasses_resized[:glasses_h, :glasses_w]
                    for i in range(glasses_h):
                        for j in range(glasses_w):
                            if glasses_has_alpha and glasses_cropped[i, j, 3] > 0:
                                alpha = glasses_cropped[i, j, 3] / 255.0
                                for c in range(3):
                                    result_image[glasses_y + i, glasses_x + j, c] = int(
                                        glasses_cropped[i, j, c] * alpha + result_image[glasses_y + i, glasses_x + j, c] * (1 - alpha))
                            elif not glasses_has_alpha:
                                result_image[glasses_y + i, glasses_x + j] = glasses_cropped[i, j]
    return result_image

def print_help():
    print("\nOpenCV")
    print("================================")
    print("1. Obrazek")
    print("2. Webkamera")
    print("q. Ukoncit program")
    print("================================")

def main():
    cascade_dir = download_cascades()
    print_help()
    while True:
        choice = input("\nZadejte volbu: ").strip().lower()
        if choice == '1':
            process_image(cascade_dir)
        elif choice == '2':
            process_webcam(cascade_dir)
        elif choice == 'q':
            print("Ukoncuji program...")
            break
        else:
            print("Neznama volba. Zadejte 1, 2 nebo q.")

if __name__ == "__main__":
    main()
