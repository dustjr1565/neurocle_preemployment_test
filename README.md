# neurocle_preemployment_test
뉴로클 채용과제 레포

## 설치 방법
1. library 설치
- python version 3.10 권장
- 코드 다운로드 후 아래 명령어를 이용해 설치 후 사용
```bash
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


## 결과 확인
- 주어진 데이터에 대한 추론결과(JSON, DB 파일, 시각화 이미지)는 아래 경로에서 다운로드 받아 확인할 수 있습니다.
    - download link: https://drive.google.com/drive/folders/1u0jlCFLUWjNKpmbf8m5aOQydz9-U06hG?usp=sharing


## 개발 환경
| 구분 | 사양 |
|---|---|
| CPU | 13th Gen Intel® Core™ i7-1355U |
| Memory | 16GB DDR4 |
| OS | Windows 11 |

- GPU는 사용하지않음.