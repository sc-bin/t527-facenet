import awnn_t527
import numpy as np
import cv2


class facenet_result:
    def __init__(self, tensor: np.ndarray):
        self.tensor = tensor

    def save(self, path):
        np.save(path, self.tensor)


def compare_faces(face1: np.ndarray, face2: np.ndarray):
    """
    比较两个人脸的相似度。
    """
    # 计算两个人脸的 L2 范数
    diff = np.linalg.norm(face1 - face2)

    # 返回相似度
    return 1 / (1 + diff)


def load_face_data(path: str):
    face_data = np.load(path)
    return face_data


class facenet(facenet_result):
    def __init__(self, model_path):
        self.npu = awnn_t527.awnn(model_path)

    def preprocess_image(self, image_path):
        """
        读取图像，缩放到 160x160，并转换为 FaceNet 模型所需的格式。
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法找到图像文件：{image_path}")

        # 将 BGR 格式转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 缩放到 160x160
        image = cv2.resize(image, (160, 160))

        # 归一化到 [0, 1]
        # image = image.astype(np.float32) / 255.0

        # 转换为 CHW 格式 (通道在前)
        # image = np.transpose(image, (2, 0, 1))

        # 添加批次维度 (Batch Size)
        image = np.expand_dims(image, axis=0)

        return image

    def detect(self, image):
        """
        使用 FaceNet 模型进行人脸识别。
        """
        # 预处理图像
        input_tensor = self.preprocess_image(image)
        self.npu.run(bytearray(input_tensor.tobytes()))
        return facenet_result(self.npu.output_buffer.get(0, 1 * 512))
