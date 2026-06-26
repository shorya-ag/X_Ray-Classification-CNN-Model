from Libraries import *
validation_data = []
val_data_dir = r'C:\Users\shory\OneDrive\Desktop\X_Ray classification Project\chest_xray\val'
img_size = 100
classes = ["NORMAL","PNEUMONIA"]
def create_val_data():
  for i in classes:
    path = os.path.join(val_data_dir,i)
    class_num = classes.index(i)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array,(img_size,img_size))
        validation_data.append([new_array,class_num])
      except Exception as e:
        pass
random.shuffle(validation_data)
for sample in validation_data[:10]:
  print(sample)
create_val_data()
#splitting the features and labels
X = []
Y = []
for features,labels in validation_data:
  X.append(features)
  Y.append(labels)
y_val = np.array(Y)
#reshaping the features for making it compatible with tensorflow
x_val = np.array(X).reshape(-1,img_size,img_size,1)
x_val = x_val.astype('float32')
x_val /= 255.