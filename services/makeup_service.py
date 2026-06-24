import cv2
import numpy as np

def apply_full_foundation_stream(src_img, skin_mask, foundation_rgb, alpha=0.22):
    """
    [설명] 사용자의 피부 영역 마스크를 기반으로 지정된 파운데이션 색상을 자연스럽게 입히는 가상 메이크업 함수.
    """
    if src_img is None:
        print("원본 이미지 데이터가 올바르지 않습니다.")
        return None
        
    # OpenCV 채널 스왑 (RGB -> BGR)
    foundation_bgr = [foundation_rgb[2], foundation_rgb[1], foundation_rgb[0]]
    
    # 1. LAB 색상 공간 변환
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB)
    
    found_pixel = np.uint8([[foundation_bgr]])
    found_lab = cv2.cvtColor(found_pixel, cv2.COLOR_BGR2LAB)[0][0]
    
    blended_lab = src_lab.copy()
    
    # 2. 색상 채널(A, B) 알파 블렌딩 합성
    blended_lab[:, :, 1] = cv2.addWeighted(src_lab[:, :, 1], 1 - alpha, np.full_like(src_lab[:, :, 1], found_lab[1]), alpha, 0)
    blended_lab[:, :, 2] = cv2.addWeighted(src_lab[:, :, 2], 1 - alpha, np.full_like(src_lab[:, :, 2], found_lab[2]), alpha, 0)
    
    # 3. 밝기 채널(L) 미세 조정 (음영 보존을 위해 alpha * 0.5 가중치 적용)
    blended_lab[:, :, 0] = cv2.addWeighted(src_lab[:, :, 0], 1 - (alpha * 0.5), np.full_like(src_lab[:, :, 0], found_lab[0]), alpha * 0.5, 0)
    
    # LAB 이미지를 다시 BGR 이미지로 역변환
    blended_full = cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
    
    # 4. 부드러운 소프트 마스크 블렌딩(Feathering) 경계선 예외 처리
    if skin_mask is not None:
        if skin_mask.shape[:2] != src_img.shape[:2]:
            skin_mask = cv2.resize(skin_mask, (src_img.shape[1], src_img.shape[0]))
            
        # 0~255 범위의 마스크를 0.0~1.0 사이의 확률 값으로 정규화
        mask_normalized = skin_mask.astype(float) / 255.0
        mask_3d = np.expand_dims(mask_normalized, axis=2)
        
        # 알파 블렌딩 공식 적용: Final = (Makeup * Mask) + (Original * (1 - Mask))
        final_output = (blended_full * mask_3d + src_img * (1.0 - mask_3d)).astype(np.uint8)
        return final_output
        
    return blended_full