<div align="center">   

# 3D X-RAY Calibration
</div>

# Examples

## Beads detection 

    python find_beads.py

## Calibration 

    python calibration.py 

![Example GIF](figs/calibration.gif)

## Open3d viewer

    python viewer_o3d.py

![IMAGE](figs/viewer_3d.png)

## Box fitting 

    python fit_bbox.py

![IMAGE](figs/visualhull.gif)
![IMAGE](figs/bbox2d.png)

## Triangulation 

    python triangulation.py

![IMAGE](figs/triangulation.png)

## VisualHull RANSAC

    python ransac_visualhull.py

![IMAGE](figs/ransac.gif)

## TODO 

- [ ] Calibration 고도화 (Distortion 모델링, PNP initialization, 타원 피팅... )   
- [ ] phantom 여러개 들어올 경우 추가 
- [ ] RANSAC 
- [ ] phantom 없는 데이터에서 calibration (NIA 9view 데이터...)
- [ ] 2d label -> 3d label 생성 
- [ ] 3d detection 모델 학습 
- [ ] reconstruction 모델 테스트 
- [ ] ...


