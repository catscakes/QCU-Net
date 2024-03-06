from pyvqnet.nn.loss import BinaryCrossEntropy
from pyvqnet.optim.adam import Adam
import matplotlib as plt
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.utils.storage import load_parameters, save_parameters
from qcnn import *
from unet import *

PREPROCESS = False

class MyDataset():
    def __init__(self, x_data, x_label):
        self.x_set = x_data
        self.label = x_label

    def __getitem__(self, item):
        img, target = self.x_set[item], self.label[item]
        img_np = np.uint8(img).transpose(2, 0, 1)
        target_np = np.uint8(target).transpose(2, 0, 1)

        img = img_np
        target = target_np
        return img, target

    def __len__(self):
        return len(self.x_set)

if not os.path.exists("./result"):
    os.makedirs("./result")
else:
    pass
if not os.path.exists("./Intermediate_results"):
    os.makedirs("./Intermediate_results")
else:
    pass

# prepare train/test data and label
path0 = 'training_data'
path1 = 'testing_data'

train_images, train_labels = PreprocessingData(path0).read()
test_images, test_labels = PreprocessingData(path1).read()

print('train: ', train_images.shape, '\ntest: ', test_images.shape)
print('train: ', train_labels.shape, '\ntest: ', test_labels.shape)
train_images = train_images / 255
test_images = test_images / 255


# use quantum encoder to preprocess data
# or you can provide your own quantum data
if PREPROCESS == True:
    print("Quantum pre-processing of train images:")
    q_train_images = quantum_data_preprocessing(train_images)
    q_test_images = quantum_data_preprocessing(test_images)
    q_train_label = quantum_data_preprocessing(train_labels)
    q_test_label = quantum_data_preprocessing(test_labels)

    # Save pre-processed images
    print('Quantum Data Saving...')
    np.save("./result/q_train.npy", q_train_images)
    np.save("./result/q_test.npy", q_test_images)
    np.save("./result/q_train_label.npy", q_train_label)
    np.save("./result/q_test_label.npy", q_test_label)
    print('Quantum Data Saving Over!')

# loading quantum data
SAVE_PATH = "./result/"
train_x = np.load(SAVE_PATH + "q_train.npy")
train_labels = np.load(SAVE_PATH + "q_train_label.npy")
test_x = np.load(SAVE_PATH + "q_test.npy")
test_labels = np.load(SAVE_PATH + "q_test_label.npy")

train_x = train_x.astype(np.uint8)
test_x = test_x.astype(np.uint8)
train_labels = train_labels.astype(np.uint8)
test_labels = test_labels.astype(np.uint8)
train_y = train_labels
test_y = test_labels

trainset = MyDataset(train_x, train_y)
testset = MyDataset(test_x, test_y)
x_train = []
y_label = []
model = UNet()
optimizer = Adam(model.parameters(), lr=0.01)
loss_func = BinaryCrossEntropy()
epochs = 200

loss_list = []
SAVE_FLAG = True
temp_loss = 0
file = open("./result/result.txt", 'w').close()
for epoch in range(1, epochs):
    total_loss = []
    model.train()
    for i, (x, y) in enumerate(trainset):
        x_img = QTensor(x, dtype=kfloat32)
        x_img_Qtensor = tensor.unsqueeze(x_img, 0)
        y_img = QTensor(y, dtype=kfloat32)
        y_img_Qtensor = tensor.unsqueeze(y_img, 0)
        optimizer.zero_grad()
        img_out = model(x_img_Qtensor)

        print(f"=========={epoch}==================")
        loss = loss_func(y_img_Qtensor, img_out)  # target output
        if i == 1:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.title("predict")
            img_out_tensor = tensor.squeeze(img_out, 0)

            if matplotlib.__version__ >= '3.4.2':
                plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
            else:
                plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
            plt.subplot(1, 2, 2)
            plt.title("label")
            y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
            if matplotlib.__version__ >= '3.4.2':
                plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
            else:
                plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))

            plt.savefig("./Intermediate_results/" + str(epoch) + "_" + str(i) + ".jpg")

        loss_data = np.array(loss.data)
        print("{} - {} loss_data: {}".format(epoch, i, loss_data))
        loss.backward()
        optimizer._step()
        total_loss.append(loss_data)

    loss_list.append(np.sum(total_loss) / len(total_loss))
    out_read = open("./result/result.txt", 'a')
    out_read.write(str(loss_list[-1]))
    out_read.write(str("\n"))
    out_read.close()
    print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
    if SAVE_FLAG:
        temp_loss = loss_list[-1]
        save_parameters(model.state_dict(), "./result/QCU-Net_End.model")
        SAVE_FLAG = False
    else:
        if temp_loss > loss_list[-1]:
            temp_loss = loss_list[-1]
            save_parameters(model.state_dict(), "./result/QCU-Net_End.model")
out_read = open("./result/result.txt", 'r')
plt.figure()
lines_read = out_read.readlines()
data_read = []
for line in lines_read:
    float_line = float(line)
    data_read.append(float_line)
out_read.close()
plt.plot(data_read)
plt.title('Unet Training')
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.savefig("./result/traing_loss.pdf")

modela = load_parameters("./result/QCU-Net_End.model")
print("----------------PREDICT-------------")
model.load_state_dict(modela)
model.eval()

for i, (x1, y1) in enumerate(testset):
    x_img = QTensor(x1, dtype=kfloat32)
    x_img_Qtensor = tensor.unsqueeze(x_img, 0)
    y_img = QTensor(y1, dtype=kfloat32)
    y_img_Qtensor = tensor.unsqueeze(y_img, 0)
    img_out = model(x_img_Qtensor)
    loss = loss_func(y_img_Qtensor, img_out)
    loss_data = np.array(loss.data)
    print("{} loss_eval: {}".format(i, loss_data))
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("predict")
    img_out_tensor = tensor.squeeze(img_out, 0)
    if matplotlib.__version__ >= '3.4.2':
        plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
    else:
        plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
    plt.subplot(1, 2, 2)
    plt.title("label")
    y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
    if matplotlib.__version__ >= '3.4.2':
        plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
    else:
        plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))
    plt.savefig("./result/eval_" + str(i) + "_1" + ".pdf")

