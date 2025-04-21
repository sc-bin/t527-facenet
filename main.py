import facenet
import os

image_path = "image/"

face = facenet.facenet("model/facenet.nb")

# 遍历image-npy路径下的所有图片

for file in os.listdir(image_path):
    if file.endswith(".jpg"):
        face.detect(image_path + file).save(image_path + file.split(".")[0] + ".npy")
face1 = facenet.load_face_data(image_path + "jack1.npy")
face2 = facenet.load_face_data(image_path + "jack2.npy")
face3 = facenet.load_face_data(image_path + "mask1.npy")
face4 = facenet.load_face_data(image_path + "mask2.npy")
print(f"f1:f2  {facenet.compare_faces(face1,face2)}")
print(f"f1:f3  {facenet.compare_faces(face1,face3)}")
print(f"f1:f4  {facenet.compare_faces(face1,face4)}")
