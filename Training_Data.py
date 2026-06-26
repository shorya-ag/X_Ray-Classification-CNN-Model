from Libraries import *
train_data_dir = r"C:\Users\shory\OneDrive\Desktop\X_Ray classification Project\chest_xray\train"
classes = ["NORMAL","PNEUMONIA"]
img_size = 100
training_data = []
def create_training_data():
  for i in classes:
    path = os.path.join(train_data_dir,i)
    class_num = classes.index(i)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(img_size,img_size))
        training_data.append([new_array,class_num])
      except Exception as e:
        pass
create_training_data()
#Shuffling the data
random.shuffle(training_data)
for sample in training_data[:10]:
  print(sample)
print(training_data)
#splitting the features and labels
x = []
y=[]
for features,labels in training_data:
  x.append(features)
  y.append(labels)
print(x)
#print(x[0].reshape(-1,img_size,img_size,1))
y_train = np.array(y)
#reshaping the features for making it compatible with tensorflow
x_train = np.array(x).reshape(-1,img_size,img_size,1)
x_train = x_train.astype('float32')
x_train /= 255.
print(x_train.shape)
