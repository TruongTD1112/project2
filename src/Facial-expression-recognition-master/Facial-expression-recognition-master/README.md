# Facial-expression-recognition

## 1. Dataset
  bộ data sử dụng là FER2013, đường link để tải là : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
  Dataset is FER2013. You can find it on Kaggle: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
  chúng ta sẽ chạy file python.py để tải về, bộ dữ liệu bao gồm 35 nghìn ảnh chia thành 7 class ,0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
 
## 2. Preprocess
  mỗi một ảnh sẽ được chuyển về dạng ảnh 48*48 mảng numpy và chuyển thành ảnh xám, chúng ta sẽ tiền xử lý dữ liệu và lưu 
  dữ liệu thành 3 tập để phục vụ cho quá trình huấn luyện, đó là 3 tập : train set, dev set, và test set
  chúng ta sẽ làm giàu data bằng cách sinh data ngẫu nhiên như quay tập ảnh có sẵn, lật ảnh sang trái hay sảng phải
  để có thể có nhiều data trong quá trình huấn luyện.

## 3. Train model
  với bài toán phân loại này thì em sẽ sử dụng vgg19, các phiên bản resnet(ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
 
  ``` python train_model.py --model ResNet18 --epochs 100 --lr 0.01 --batch_size 128 ```
  sau khi huấn luyện thì sẽ lưu lại giá trị best val_acc trong model folder, bây giời thì sẽ load lại model với bộ trọng số vừa
  có được trong quá trình huấn luyện33
  
  The best accuracy~70% on test set.
  
  Training time ~30s / 1 epochs on Tesla K80 GPU(Google Colab).
  
  
  Trọng số cho mô hình được sàng lọc trên tập dữ liệu của hình ảnh có thể làm giảm thời gian huấn luyện.

## 4. Real time facial expression recognition with OpenCV
  
Sau khi lưu lại giá trị checkpoint của mô hình, em sử dụng OpenCV để triển khai.
  
  
