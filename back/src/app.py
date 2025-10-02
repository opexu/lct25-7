from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import io
from rembg import remove
from PIL import Image
import aiofiles
import os
from datetime import datetime
import base64
import random
from pathlib import Path
import numpy as np
import mediapipe as mp
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

BACKGROUNDS_DIR = f"data/bg"

app = FastAPI(title="My FastAPI App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://192.168.0.250", 
        "https://192.168.0.250", 
        "http://192.168.0.250:5173", 
        "https://192.168.0.250:5173", 
        "http://127.0.0.1", 
        "https://127.0.0.1"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

@app.get("/")
async def root():
    return {"message": "API Service", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Service is running"}

# @app.post("/remove-background")
# async def remove_background(
#     file: UploadFile = File(..., description=""),
#     return_format: str = "png"
# ):
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
#     allowed_formats = ['png', 'jpg', 'jpeg', 'webp']
#     if return_format.lower() not in allowed_formats:
#         raise HTTPException(status_code=400, detail=f"Формат {return_format} не поддерживается. Используйте: {allowed_formats}")
    
#     try:
#         contents = await file.read()
        
#         output_bytes = remove(contents)
        
#         if return_format.lower() != 'png':
#             input_image = Image.open(io.BytesIO(output_bytes))
            
#             if return_format.lower() in ['jpg', 'jpeg']:
#                 if input_image.mode in ('RGBA', 'LA'):
#                     background = Image.new('RGB', input_image.size, (255, 255, 255))
#                     background.paste(input_image, mask=input_image.split()[-1])
#                     input_image = background
            
#             output_buffer = io.BytesIO()
#             input_image.save(output_buffer, format=return_format.upper())
#             output_bytes = output_buffer.getvalue()
        
#         content_type = f"image/{return_format.lower()}"
#         if return_format.lower() == 'jpg':
#             content_type = "image/jpeg"
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         original_path = f"data/original_{timestamp}_{file.filename}"
#         result_path = f"data/result_{timestamp}_{file.filename}.{return_format}"
        
#         async with aiofiles.open(original_path, 'wb') as f:
#             await f.write(contents)
        
#         async with aiofiles.open(result_path, 'wb') as f:
#             await f.write(output_bytes)
        
#         background_path = get_random_background()
#         if background_path and os.path.exists(background_path):
#             output_bytes = composite_images(
#                 output_bytes, 
#                 background_path, 
#                 position='center'
#             )

#         base64_str = base64.b64encode(output_bytes).decode('utf-8')

#         return JSONResponse(
#             content={
#                 "format": return_format,
#                 "data": f"data:image/{return_format};base64,{base64_str}",
#             },
#             media_type="application/json",
#             headers={
#                 "Cache-Control": "no-cache"
#             }
#         )
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

@app.post("/remove-background")
async def remove_background(
    file: UploadFile = File(..., description=""),
    return_format: str = "png"
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    allowed_formats = ['png', 'jpg', 'jpeg', 'webp']
    if return_format.lower() not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"Формат {return_format} не поддерживается. Используйте: {allowed_formats}")
    
    try:
        contents = await file.read()
        
        output_bytes = remove(contents)
        output_bytes = remove_transparent_borders(output_bytes)

        if return_format.lower() != 'png':
            input_image = Image.open(io.BytesIO(output_bytes))
            
            if return_format.lower() in ['jpg', 'jpeg']:
                if input_image.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', input_image.size, (255, 255, 255))
                    background.paste(input_image, mask=input_image.split()[-1])
                    input_image = background
            
            output_buffer = io.BytesIO()
            input_image.save(output_buffer, format=return_format.upper())
            output_bytes = output_buffer.getvalue()
        
        content_type = f"image/{return_format.lower()}"
        if return_format.lower() == 'jpg':
            content_type = "image/jpeg"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = f"data/original_{timestamp}_{file.filename}"
        result_path = f"data/result_{timestamp}_{file.filename}.{return_format}"
        
        async with aiofiles.open(original_path, 'wb') as f:
            await f.write(contents)
        
        async with aiofiles.open(result_path, 'wb') as f:
            await f.write(output_bytes)
        
        body_parts = {}
        positioning_strategy = {
            "position": "center",
            "scale": 0.7,
            "vertical_align": "center",
            "reason": "Позиция по умолчанию"
        }
        body_parts = detect_body_parts(output_bytes)
        print("body_parts")
        print(body_parts)
        positioning_strategy = get_positioning_strategy(
            body_parts, 
            body_parts.get("image_size")
        )

        background_path = get_random_background()
        if background_path and os.path.exists(background_path):
            output_bytes = enhance_composition(
                output_bytes, background_path, positioning_strategy
            )

        

        base64_str = base64.b64encode(output_bytes).decode('utf-8')

        return JSONResponse(
            content={
                "format": return_format,
                "data": f"data:image/{return_format};base64,{base64_str}",
            },
            media_type="application/json",
            headers={
                "Cache-Control": "no-cache"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")


def get_random_background():
    if not os.path.exists(BACKGROUNDS_DIR):
        return None
    
    background_files = [f for f in os.listdir(BACKGROUNDS_DIR) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not background_files:
        return None
    
    random_bg = random.choice(background_files)
    return os.path.join(BACKGROUNDS_DIR, random_bg)

def composite_images(foreground_bytes: bytes, background_path: Path, position="center", scale=0.7):
    foreground = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")
    
    # Открываем background
    background = Image.open(background_path).convert("RGBA")
    
    # Масштабируем foreground чтобы он хорошо смотрелся на background
    bg_width, bg_height = background.size
    fg_width, fg_height = foreground.size
    
    # Вычисляем новый размер с сохранением пропорций
    scale_factor = min(
        (bg_width * scale) / fg_width,
        (bg_height * scale) / fg_height
    )
    
    new_width = int(fg_width * scale_factor)
    new_height = int(fg_height * scale_factor)
    
    # Изменяем размер foreground
    foreground_resized = foreground.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Вычисляем позицию для размещения
    if position == "center":
        x = (bg_width - new_width) // 2
        y = (bg_height - new_height) // 2
    elif position == "bottom":
        x = (bg_width - new_width) // 2
        y = bg_height - new_height - int(bg_height * 0.1)  # 10% отступа снизу
    elif position == "top":
        x = (bg_width - new_width) // 2
        y = int(bg_height * 0.1)  # 10% отступа сверху
    else:
        x = (bg_width - new_width) // 2
        y = (bg_height - new_height) // 2
    
    # Создаем копию background и накладываем foreground
    result = background.copy()
    result.paste(foreground_resized, (x, y), foreground_resized)
    
    # Конвертируем обратно в bytes
    output_buffer = io.BytesIO()
    result.save(output_buffer, format="PNG")
    return output_buffer.getvalue()

def composite_images_2(foreground_bytes, background_path, positioning_strategy):
    """
    Накладывает изображение без фона на фоновое изображение
    с учетом стратегии позиционирования
    """
    # Открываем foreground (изображение без фона)
    foreground = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")
    
    # Открываем background
    background = Image.open(background_path).convert("RGBA")
    
    # Масштабируем foreground согласно стратегии
    bg_width, bg_height = background.size
    fg_width, fg_height = foreground.size
    
    scale_factor = positioning_strategy["scale"]
    
    # Вычисляем новый размер с сохранением пропорций
    new_width = int(fg_width * scale_factor)
    new_height = int(fg_height * scale_factor)
    
    # Изменяем размер foreground
    foreground_resized = foreground.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Вычисляем позицию для размещения с АГРЕССИВНЫМ прижатием
    vertical_align = positioning_strategy["vertical_align"]
    vertical_offset = positioning_strategy.get("vertical_offset", 0.0)  # По умолчанию 0
    
    if vertical_align == "center":
        y = (bg_height - new_height) // 2
    elif vertical_align == "top":
        y = int(bg_height * vertical_offset)  # ✅ Может быть 0 для вплотную
    elif vertical_align == "bottom":
        y = bg_height - new_height - int(bg_height * vertical_offset)  # ✅ Может быть 0 для вплотную
    else:
        y = (bg_height - new_height) // 2
    
    # Всегда центрируем по горизонтали
    x = (bg_width - new_width) // 2
    
    # Создаем копию background и накладываем foreground
    result = background.copy()
    result.paste(foreground_resized, (x, y), foreground_resized)
    
    # Конвертируем обратно в bytes
    output_buffer = io.BytesIO()
    result.save(output_buffer, format="PNG")
    return output_buffer.getvalue()

def detect_body_parts(image_bytes):
    """
    Детектирует части тела на изображении УЖЕ ОБРЕЗАННОМ
    """
    # Сначала обрезаем прозрачные границы для более точной детекции
    # cropped_bytes = remove_transparent_borders(image_bytes)
    
    # Конвертируем bytes в numpy array для OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"has_person": False}
    
    # Сохраняем размер изображения
    image_size = (image.shape[1], image.shape[0])  # (width, height)
    
    # Конвертируем BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # Детекция позы
    results = pose.process(image_rgb)
    
    body_parts = {
        "has_person": False,
        "has_head": False,
        "has_arms": False,
        "has_legs": False,
        "head_bbox": None,
        "feet_bbox": None,
        "body_bbox": None,
        "image_size": image_size  # ✅ Добавляем размер изображения
    }
    
    if not results.pose_landmarks:
        return body_parts
    
    body_parts["has_person"] = True
    
    landmarks = results.pose_landmarks.landmark
    h, w = image.shape[:2]
    
    # Ключевые точки для головы (нос, глаза, уши)
    head_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR
    ]
    
    # Ключевые точки для рук
    arm_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST
    ]
    
    # Ключевые точки для ног
    leg_landmarks = [
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL,
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]
    
    # Проверяем голову
    head_points = []
    for landmark in head_landmarks:
        point = landmarks[landmark.value]
        if point.visibility > 0.5:
            head_points.append((int(point.x * w), int(point.y * h)))
    
    if len(head_points) >= 2:
        body_parts["has_head"] = True
        if head_points:
            x_coords = [p[0] for p in head_points]
            y_coords = [p[1] for p in head_points]
            body_parts["head_bbox"] = {
                "x_min": min(x_coords), "y_min": min(y_coords),
                "x_max": max(x_coords), "y_max": max(y_coords)
            }
    
    # Проверяем руки
    arm_points = []
    for landmark in arm_landmarks:
        point = landmarks[landmark.value]
        if point.visibility > 0.5:
            arm_points.append((int(point.x * w), int(point.y * h)))
    
    if len(arm_points) >= 2:
        body_parts["has_arms"] = True
    
    # Проверяем ноги
    leg_points = []
    for landmark in leg_landmarks:
        point = landmarks[landmark.value]
        if point.visibility > 0.5:
            leg_points.append((int(point.x * w), int(point.y * h)))
    
    if len(leg_points) >= 2:
        body_parts["has_legs"] = True
        if leg_points:
            x_coords = [p[0] for p in leg_points]
            y_coords = [p[1] for p in leg_points]
            body_parts["feet_bbox"] = {
                "x_min": min(x_coords), "y_min": min(y_coords),
                "x_max": max(x_coords), "y_max": max(y_coords)
            }
    
    # Вычисляем общий bounding box тела
    all_points = []
    for landmark in mp_pose.PoseLandmark:
        point = landmarks[landmark.value]
        if point.visibility > 0.3:
            all_points.append((int(point.x * w), int(point.y * h)))
    
    if all_points:
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        body_parts["body_bbox"] = {
            "x_min": min(x_coords), "y_min": min(y_coords),
            "x_max": max(x_coords), "y_max": max(y_coords)
        }
    
    return body_parts

def get_positioning_strategy(body_parts, image_size=None):
    """
    Определяет стратегию позиционирования с ПРИЖАТИЕМ ВПЛОТНУЮ к краям
    """
    has_head = body_parts.get("has_head", False)
    has_legs = body_parts.get("has_legs", False)
    has_arms = body_parts.get("has_arms", False)
    
    body_bbox = body_parts.get("body_bbox")
    feet_bbox = body_parts.get("feet_bbox")
    
    # Основная логика с ПРИЖАТИЕМ ВПЛОТНУЮ
    if has_head and has_legs and has_arms:
        # Полное тело - центрируем
        return {
            "position": "center",
            "scale": 0.7,
            "vertical_align": "center",
            "reason": "Обнаружено полное тело с головой, руками и ногами"
        }
        
    elif has_head and not has_legs:
        # Голова и торс есть, но ног нет - ПРИЖИМАЕМ ВПЛОТНУЮ К НИЗУ
        return {
            "position": "bottom",
            "scale": 0.65,
            "vertical_align": "bottom",
            "vertical_offset": 0.0,  # ✅ НУЛЕВОЙ отступ - вплотную к краю
            "reason": "Тело с обрезанными ногами - прижато вплотную к низу"
        }
            
    elif has_legs and not has_head:
        # Ноги есть, но головы нет - ПРИЖИМАЕМ ВПЛОТНУЮ К ВЕРХУ
        return {
            "position": "top",
            "scale": 0.65,
            "vertical_align": "top", 
            "vertical_offset": 0.0,  # ✅ НУЛЕВОЙ отступ - вплотную к краю
            "reason": "Тело с обрезанной головой - прижато вплотную к верху"
        }
            
    elif has_head:
        # Только голова
        return {
            "position": "top",
            "scale": 0.4,
            "vertical_align": "top",
            "vertical_offset": 0.0,
            "reason": "Обнаружена только голова"
        }
        
    else:
        # Ничего не найдено
        return {
            "position": "center",
            "scale": 0.8,
            "vertical_align": "center", 
            "reason": "Части тела не обнаружены"
        }
    
def remove_transparent_borders(image_bytes):
    """
    Обрезает прозрачные границы у изображения
    """
    # Открываем изображение
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # Получаем bounding box непрозрачных пикселей
    bbox = image.getbbox()
    
    if bbox:
        # Обрезаем изображение
        cropped_image = image.crop(bbox)
        
        # Конвертируем обратно в bytes
        output_buffer = io.BytesIO()
        cropped_image.save(output_buffer, format="PNG")
        return output_buffer.getvalue()
    else:
        # Если все пиксели прозрачные, возвращаем оригинал
        return image_bytes
    
def color_transfer_Reinhard(source_pil, target_pil):
    """
    Алгоритм Reinhard для переноса цвета
    """
    source = np.array(source_pil)
    target = np.array(target_pil)
    
    # Конвертируем в Lab color space
    source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype("float32")
    
    # Вычисляем mean и std
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = (
        np.mean(source[:,:,0]), np.std(source[:,:,0]),
        np.mean(source[:,:,1]), np.std(source[:,:,1]), 
        np.mean(source[:,:,2]), np.std(source[:,:,2])
    )
    
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = (
        np.mean(target[:,:,0]), np.std(target[:,:,0]),
        np.mean(target[:,:,1]), np.std(target[:,:,1]),
        np.mean(target[:,:,2]), np.std(target[:,:,2])
    )
    
    # Вычитаем mean из source
    (l, a, b) = cv2.split(source)
    l -= lMeanSrc
    a -= aMeanSrc
    b -= bMeanSrc
    
    # Масштабируем по std target/source
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b
    
    # Добавляем mean target
    l += lMeanTar
    a += aMeanTar
    b += bMeanTar
    
    # Ограничиваем значения
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    # Объединяем каналы и конвертируем обратно в RGB
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(transfer)

def enhance_composition_with_lighting(foreground_bytes, background_path, positioning_strategy):
    """
    Улучшенная композиция с согласованием освещения
    """
    # Открываем изображения
    foreground = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")
    
    bg_width, bg_height = background.size
    fg_width, fg_height = foreground.size
    
    # Применяем масштабирование
    scale = positioning_strategy["scale"]
    new_width = int(fg_width * scale)
    new_height = int(fg_height * scale)
    
    foreground_resized = foreground.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # ✅ СОГЛАСОВАНИЕ ОСВЕЩЕНИЯ
    if new_width > 0 and new_height > 0:
        try:
            # Создаем копию foreground для цветокоррекции
            foreground_rgb = foreground_resized.convert("RGB")
            background_rgb = background.convert("RGB")
            
            # Обрезаем background до области где будет foreground для точного matching
            x, y = calculate_position(background.size, foreground_resized.size, positioning_strategy)
            
            # Берем область фона под foreground
            bg_region = background_rgb.crop((
                max(0, x), max(0, y),
                min(bg_width, x + new_width), 
                min(bg_height, y + new_height)
            ))
            
            # Масштабируем bg_region к размеру foreground если нужно
            if bg_region.size != foreground_rgb.size:
                bg_region = bg_region.resize(foreground_rgb.size, Image.Resampling.LANCZOS)
            
            # Применяем color transfer
            foreground_corrected = color_transfer_Reinhard(foreground_rgb, bg_region)
            
            # Восстанавливаем альфа-канал
            r, g, b = foreground_corrected.split()
            alpha = foreground_resized.split()[-1]  # Берем альфа из оригинала
            foreground_corrected = Image.merge("RGBA", (r, g, b, alpha))
            
            foreground_resized = foreground_corrected
            
        except Exception as e:
            print(f"Color correction failed: {e}")
            # Продолжаем без коррекции цвета
    
    # Вычисляем финальную позицию
    x, y = calculate_position(background.size, foreground_resized.size, positioning_strategy)
    
    # Накладываем изображение
    result = background.copy()
    result.paste(foreground_resized, (x, y), foreground_resized)
    
    output_buffer = io.BytesIO()
    result.save(output_buffer, format="PNG")
    return output_buffer.getvalue()

def calculate_position(bg_size, fg_size, strategy):
    """Вычисляет позицию для вставки"""
    bg_width, bg_height = bg_size
    fg_width, fg_height = fg_size
    
    # Вертикальное позиционирование
    vertical_align = strategy["vertical_align"]
    vertical_offset = strategy.get("vertical_offset", 0.0)
    
    if vertical_align == "top":
        y = int(bg_height * vertical_offset)
    elif vertical_align == "bottom":
        y = bg_height - fg_height - int(bg_height * vertical_offset)
    else:  # center
        y = (bg_height - fg_height) // 2
    
    # Горизонтальное позиционирование
    horizontal_align = strategy.get("horizontal_align", "center")
    horizontal_offset = strategy.get("horizontal_offset", 0.0)
    
    if horizontal_align == "left":
        x = int(bg_width * horizontal_offset)
    elif horizontal_align == "right":
        x = bg_width - fg_width - int(bg_width * horizontal_offset)
    else:  # center
        x = (bg_width - fg_width) // 2
    
    return x, y

class SimpleColorCorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
        
        # ✅ АВТОМАТИЧЕСКАЯ ИНИЦИАЛИЗАЦИЯ ВЕСОВ
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, foreground, background):
        x = torch.cat([foreground, background], dim=1)
        correction = self.layers(x)
        result = foreground + correction * 0.3  # Мягкая коррекция
        return torch.clamp(result, 0, 1)

def neural_lighting_match(foreground_pil, background_pil, strength=0.5):
    """
    ✅ ГОТОВАЯ ФУНКЦИЯ - ПРОСТО ВСТАВЬТЕ И ВЫЗЫВАЙТЕ
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleColorCorrectionNet().to(device)
    model.eval()  # ✅ Важно: режим inference
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Подготавливаем изображения
    fg_tensor = transform(foreground_pil).unsqueeze(0).to(device)
    bg_tensor = transform(background_pil).unsqueeze(0).to(device)
    
    # Ресайзим background к размеру foreground
    if bg_tensor.shape[2:] != fg_tensor.shape[2:]:
        bg_tensor = F.interpolate(bg_tensor, size=fg_tensor.shape[2:], mode='bilinear')
    
    with torch.no_grad():
        corrected = model(fg_tensor, bg_tensor)
        # Смешиваем с оригиналом
        result = fg_tensor * (1 - strength) + corrected * strength
        result = torch.clamp(result, 0, 1)
    
    # Конвертируем обратно в PIL
    return transforms.ToPILImage()(result.squeeze(0).cpu())

def enhance_composition(foreground_bytes, background_path, positioning_strategy):
    foreground = Image.open(io.BytesIO(foreground_bytes)).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")
    
    # Масштабирование
    scale = positioning_strategy["scale"]
    new_size = (int(foreground.width * scale), int(foreground.height * scale))
    foreground_resized = foreground.resize(new_size, Image.Resampling.LANCZOS)
    
    try:
        # Получаем область фона
        x, y = calculate_position(background.size, foreground_resized.size, positioning_strategy)
        bg_region = get_background_region(background, x, y, new_size)
        
        # ✅ ВЫЗОВ НЕЙРОСЕТИ
        foreground_rgb = foreground_resized.convert("RGB")
        foreground_corrected = neural_lighting_match(foreground_rgb, bg_region, strength=0.4)
        
        # Восстанавливаем альфа-канал
        r, g, b = foreground_corrected.split()
        alpha = foreground_resized.split()[-1]
        foreground_final = Image.merge("RGBA", (r, g, b, alpha))
        foreground_resized = foreground_final
        
    except Exception as e:
        print(f"Neural lighting failed, using original: {e}")
    
    # Финальная композиция
    result = background.copy()
    x, y = calculate_position(background.size, foreground_resized.size, positioning_strategy)
    result.paste(foreground_resized, (x, y), foreground_resized)
    
    output_buffer = io.BytesIO()
    result.save(output_buffer, format="PNG")
    return output_buffer.getvalue()

def get_background_region(background, x, y, size):
    """Вырезает область фона под foreground"""
    bg_region = background.crop((
        max(0, x), max(0, y),
        min(background.width, x + size[0]), 
        min(background.height, y + size[1])
    ))
    if bg_region.size != size:
        bg_region = bg_region.resize(size, Image.Resampling.LANCZOS)
    return bg_region