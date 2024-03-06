from pyvqnet.dtype import *
import pyqpanda as pq
from pyqpanda import *
import matplotlib



import cv2
# quantum data prepare
class PreprocessingData:
    def __init__(self, path):
        self.path = path
        self.x_data = []
        self.y_label = []


    def processing(self):
        list_path = os.listdir((self.path+"/images"))
        for i in range(len(list_path)):

            temp_data = cv2.imread(self.path+"/images" + '/' + list_path[i], cv2.IMREAD_COLOR)
            temp_data = cv2.resize(temp_data, (128, 128))
            grayimg = cv2.cvtColor(temp_data, cv2.COLOR_BGR2GRAY)
            temp_data = grayimg.reshape(temp_data.shape[0], temp_data.shape[0], 1).astype(np.float32)
            self.x_data.append(temp_data)

            label_data = cv2.imread(self.path+"/labels" + '/' +list_path[i].split(".")[0] + "_mask.png", cv2.IMREAD_COLOR)
            # print(self.path+"/labels" + '/' +list_path[i].split(".")[0] + "_mask.png")
            label_data = cv2.resize(label_data, (128, 128))

            label_data = cv2.cvtColor(label_data, cv2.COLOR_BGR2GRAY)
            label_data = label_data.reshape(label_data.shape[0], label_data.shape[0], 1).astype(np.int64)
            self.y_label.append(label_data)

        return self.x_data, self.y_label

    def read(self):
        self.x_data, self.y_label = self.processing()
        x_data = np.array(self.x_data)
        y_label = np.array(self.y_label)

        return x_data, y_label

# quantum cnn layer
class QCNN_:
    def __init__(self, image):
        self.image = image

    def encode_cir(self, qlist, pixels):
        cir = pq.QCircuit()
        for i, pix in enumerate(pixels):
            theta = np.arctan(pix)
            phi = np.arctan(pix**2)
            cir.insert(pq.RY(qlist[i], theta))
            cir.insert(pq.RZ(qlist[i], phi))
        return cir

    def entangle_cir(self, qlist):
        k_size = len(qlist)
        cir = pq.QCircuit()
        for i in range(k_size):
            ctr = i
            ctred = i+1
            if ctred == k_size:
                ctred = 0
            cir.insert(pq.CNOT(qlist[ctr], qlist[ctred]))
        return cir

    def qcnn_circuit(self, pixels):
        k_size = len(pixels)
        # to gpu
        machine = pq.MPSQVM()

        machine.init_qvm()
        qlist = machine.qAlloc_many(k_size)
        cir = pq.QProg()

        cir.insert(self.encode_cir(qlist, np.array(pixels) * np.pi / 2))
        cir.insert(self.entangle_cir(qlist))

        result0 = machine.prob_run_list(cir, [qlist[0]], -1)
        result1 = machine.prob_run_list(cir, [qlist[1]], -1)
        result2 = machine.prob_run_list(cir, [qlist[2]], -1)
        result3 = machine.prob_run_list(cir, [qlist[3]], -1)

        result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
        machine.finalize()
        return result

def quanconv_(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((64, 64, 1))

    for j in range(0, 128, 2):
        for k in range(0, 128, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results = QCNN_(image).qcnn_circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )

            for c in range(1):
                out[j // 2, k // 2, c] = q_results[c]
    return out

def quantum_data_preprocessing(images):
    quantum_images = []
    for _, img in enumerate(images):
        quantum_images.append(quanconv_(img))
    quantum_images = np.asarray(quantum_images)
    return quantum_images

