import cv2
import numpy as np

class PersonalColorAnalyzer:
    """
    [설명] 피부(볼/이마), 눈동자, 입술의 색상 데이터를 기반으로 한 사계절 퍼스널 컬러 진단 시스템.
    """
    def __init__(self):
        self.global_skin_criteria = {
            'light': {
                'Spring (봄 웜)':   {'L': 80.0, 'b': 10.0, 's': 85.0},
                'Summer (여름 쿨)': {'L': 78.0, 'b': 3.0,  's': 60.0},
                'Autumn (가을 웜)': {'L': 68.0, 'b': 14.0, 's': 100.0},
                'Winter (겨울 쿨)': {'L': 65.0, 'b': 2.0,  's': 75.0}
            },
            'asian': {
                'Spring (봄 웜)':   {'L': 70.0, 'b': 12.0, 's': 90.0},
                'Summer (여름 쿨)': {'L': 68.0, 'b': 4.0,  's': 65.0},
                'Autumn (가을 웜)': {'L': 58.0, 'b': 16.0, 's': 110.0},
                'Winter (겨울 쿨)': {'L': 55.0, 'b': 3.0,  's': 80.0}
            },
            'dark': {
                'Spring (봄 웜)':   {'L': 40.0, 'b': 14.0, 's': 95.0},
                'Summer (여름 쿨)': {'L': 38.0, 'b': 5.0,  's': 70.0},
                'Autumn (가을 웜)': {'L': 30.0, 'b': 18.0, 's': 115.0},
                'Winter (겨울 쿨)': {'L': 28.0, 'b': 4.0,  's': 85.0}
            }
        }

    def _get_mean_rgb_from_region(self, region_bgr):
        """
        [추가] face_mesh_extractor가 반환한 region 이미지(배경이 검은색인 mask 영역)에서
        검은색(0, 0, 0)을 제외한 실제 유효 픽셀들의 평균 RGB 값을 추출합니다.
        """
        if region_bgr is None:
            return [128, 128, 128] # 폴백 기본값
            
        # 검은색 배경이 아닌 유효한 픽셀만 필터링
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        idx = np.where(gray > 0)
        
        if len(idx[0]) == 0:
            return [128, 128, 128]
            
        # BGR 평균 계산 후 RGB 순서로 전환
        mean_b = np.mean(region_bgr[:, :, 0][idx])
        mean_g = np.mean(region_bgr[:, :, 1][idx])
        mean_r = np.mean(region_bgr[:, :, 2][idx])
        
        return [int(mean_r), int(mean_g), int(mean_b)]

    def _extract_features(self, rgb):
        img_bgr = np.uint8([[ [rgb[2], rgb[1], rgb[0]] ]]) # RGB -> BGR 내부 변환 보정
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        return {
            'L': img_lab[0][0][0],
            'b': img_lab[0][0][2],
            's': img_hsv[0][0][1]
        }

    def diagnose_from_regions(self, regions_dict):
        """
        [추가] face_mesh_extractor의 아웃풋 딕셔너리를 통째로 받아 진단하는 편리한 엔트리 포인트
        """
        forehead_rgb = self._get_mean_rgb_from_region(regions_dict.get("forehead_region"))
        left_cheek_rgb = self._get_mean_rgb_from_region(regions_dict.get("left_cheek_region"))
        right_cheek_rgb = self._get_mean_rgb_from_region(regions_dict.get("right_cheek_region"))
        eye_rgb = self._get_mean_rgb_from_region(regions_dict.get("iris_region"))
        eyebrow_rgb = self._get_mean_rgb_from_region(regions_dict.get("eyebrow_region"))
        lip_rgb = self._get_mean_rgb_from_region(regions_dict.get("lip_region"))
        
        # 볼은 좌우 평균값 사용
        cheek_rgb = [
            int((left_cheek_rgb[0] + right_cheek_rgb[0]) / 2),
            int((left_cheek_rgb[1] + right_cheek_rgb[1]) / 2),
            int((left_cheek_rgb[2] + right_cheek_rgb[2]) / 2)
        ]
        
        return self.diagnose(cheek_rgb, forehead_rgb, eye_rgb, eyebrow_rgb, lip_rgb)

    def diagnose(self, cheek_rgb, forehead_rgb, eye_rgb, eyebrow_rgb, lip_rgb):
        # 1. 입력 부위별 색상 공간 특성 추출
        cheek_feat = self._extract_features(cheek_rgb)
        forehead_feat = self._extract_features(forehead_rgb)
        eye_feat = self._extract_features(eye_rgb)
        lip_feat = self._extract_features(lip_rgb) 
        
        # 2. 피부 최종 특성 결정
        skin_L = (cheek_feat['L'] + forehead_feat['L']) / 2.0
        skin_b = (cheek_feat['b'] + forehead_feat['b']) / 2.0
        skin_s = (cheek_feat['s'] + forehead_feat['s']) / 2.0
        
        user_skin_rgb = [
            int((cheek_rgb[0] + forehead_rgb[0]) / 2),
            int((cheek_rgb[1] + forehead_rgb[1]) / 2),
            int((cheek_rgb[2] + forehead_rgb[2]) / 2)
        ]
        
        if skin_L >= 75.0:
            target_group = 'light'
        elif skin_L <= 45.0:
            target_group = 'dark'
        else:
            target_group = 'asian'
            
        selected_criteria = self.global_skin_criteria[target_group]
        
        contrast = abs(skin_L - eye_feat['L'])
        scores = {}
        
        for season, criterion in selected_criteria.items():
            l_diff = abs(skin_L - criterion['L'])
            b_diff = abs(skin_b - criterion['b'])
            s_diff = abs(skin_s - criterion['s'])
            
            total_diff = (l_diff * 0.2) + (b_diff * 0.6) + (s_diff * 0.2)
            
            if target_group == 'dark':
                if season == 'Winter (겨울 쿨)' and contrast < 15: total_diff += 10
                elif season == 'Summer (여름 쿨)' and contrast > 25: total_diff += 10
            else:
                if season == 'Winter (겨울 쿨)' and contrast < 25: total_diff += 10  
                elif season == 'Summer (여름 쿨)' and contrast > 35: total_diff += 10  

            if '쿨' in season and lip_feat['b'] > 12: 
                total_diff += 2

            scores[season] = total_diff

        best_season = min(scores, key=scores.get)
        
        return {
            "personal_color": best_season,
            "skin_rgb": user_skin_rgb
        }