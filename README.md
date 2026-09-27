# neurocle_preemployment_test
뉴로클 채용과제 레포

## 설치 방법
1. library 설치
- python version 3.10 권장
- 아래 명령어를 이용해 설치 후 사용
```bash
cd autolabel
pip install .
```

## 사용 방법
1. Inference 
```bash
autolabel inference --input_dir "./assignment_data/images"
```
- 기능
    - 이미지들을 모델 추론 하여 bounding box 및 mask 정보를 json 및 db 형태로 저장합니다.
- 인자 설명
    -  `--input_dir` 
        - 추론할 이미지들이 있는 디렉터리 경로 입니다.
    - `--output_dir`
        - 출력결과를 저장할 디렉터리 경로입니다. 기본값은 './outputs/inference' 입니다.
    - `--model_dir`
        - 모델에 대한 디렉터리 경로입니다. 기본값은 './model' 이며 존재하지 않을 시 다운로드하여 사용합니다.
    - `--vis`
        - 시각화한 결과를 출력할 것인지에 대한 설정입니다. 기본값은 False이며 결과는 './outputs/inference/vis'에 저장됩니다.
