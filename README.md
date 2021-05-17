# CIFAR10_Classification
>  로컬 파일 또는 URL을 통해 받은 이미지를 10개의 클래스로 분류 및 출력하는 프로그램입니다. 
>
> Keras를 통한 CNN 구현으로 이미지 분류 모델을 개발하였으며, 이미지 분류 label은 아래와 같습니다.
>
> ![라벨](https://user-images.githubusercontent.com/55572533/118454066-e7caaf80-b732-11eb-9a03-2faa648ea6aa.JPG)



### CIFAR-10 이란?

인공지능 알고리즘을 훈련시키는 데에 사용되는 이미지 모음입니다. 

>  각 이미지는 0부터 9까지의 label이 대응됩니다.

* airplane : 0
* automobile : 1
* bird : 2
* cat : 3
* deer : 4
* dog : 5

* frog : 6
* horse : 7
* ship : 8
* truck : 9

##### 

### Install

1. 우선, 임시 이미지로 temp.jpg를 준비합니다. (어떤 사이즈, 이미지여도 괜찮습니다.)

> URL을 통해 이미지를 불러올 경우, temp.jpg 파일에 해당 이미지를 저장한 후, temp.jpg를 이미지 분류에 사용합니다.

2. .py 파일을 실행합니다.

3. ![ui2](https://user-images.githubusercontent.com/55572533/118457433-b4892000-b734-11eb-8a70-b7f279861f3c.JPG)

   ​	① : 분류할 이미지 파일의 경로 또는 URL을 입력합니다.

   ​	② : Upload image 또는 Upload URL 버튼을 클릭하여 분류를 실행합니다.

   ​	③ : 좌측에 해당 이미지가, 우측의 ???에 이미지 분류 결과가 나타납니다.



### 사용 예시

* airplane

![airplane](https://user-images.githubusercontent.com/55572533/118453502-42afd700-b732-11eb-9675-ed0ef0f05220.JPG)



* cat

  ![cat](https://user-images.githubusercontent.com/55572533/118453769-915d7100-b732-11eb-9f65-162bbfe76347.JPG)

  

