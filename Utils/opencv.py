import cv2
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os

import asyncio
import websockets

from Utils.lstm_model import LSTMModel
from Utils.support_functions import SupportFunctions


class VideoPoseEstimation :
    EDGES = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }

    def __init__(self) :
        # Download the model from TF Hub
        self.model_loc = "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1"
        self.model = hub.load(self.model_loc)
        self.movenet = self.model.signatures['serving_default']
        
        self.viz_colors = [(245,117,16), (117,245,16), (16,117,245),(200,103,27)]
        
    # Draw Edges
    def draw_connections(self, frame, keypoints, edges, confidence_thr) :
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        
        for edge, color in edges.items() :
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if (c1 > confidence_thr and c2 > confidence_thr) :
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
    # Draw Keypoints
    def draw_keypoints(self, frame, keypoints, confidence_thr) :
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        
        for kp in shaped :
            ky, kx, kp_conf = kp
            if kp_conf > confidence_thr :
                    cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
                    
    # Function to loop through each person detected and render
    def loop_through_people(self, frame, keypoints_with_scores, edges, confidence_thr) :
        for person in keypoints_with_scores :
            self.draw_connections(frame, person, edges, confidence_thr)
            self.draw_keypoints(frame, person, confidence_thr)
            
    
    def simple_pose_detection(self) :
        cap = cv2.VideoCapture("./vids/normal/3.mp4")

        while cap.isOpened() :
            ret, frame = cap.read()
            
            if ret :
                frame = cv2.resize(frame, (800, 800))
            else:
                break

            # resize img
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 288, 288)
            inp_img = tf.cast(img, dtype=tf.int32)
            
            #Detection section
            results = self.movenet(inp_img)
            keypoints_with_scores = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))
            
            #Render Keypoints
            self.loop_through_people(frame, keypoints_with_scores, self.EDGES, 0.1)
            
            cv2.imshow("Movenet Multiperson", frame)
            
            if cv2.waitKey(10) & 0xFF == ord("q") :
                break
        cap.release()
        cv2.destroyAllWindows()
        
        
    def collecting_training_data(self) :
        br = False
        for action in support.actions :
            for vid_no in range(support.no_sequences) :
                vid_file = os.path.join(os.getcwd(),"./vids", action, f"{vid_no}.mp4")
                print(f"vid_file: {vid_file}")
                cap = cv2.VideoCapture(vid_file)
                if not cap.isOpened :
                    print("Cap is not opened")

                for frame_num in range(support.sequence_length) :
                    # Read feed
                    ret, frame = cap.read()

                    if ret :
                        frame = cv2.resize(frame, (800, 800))
                    else:
                        print("Breaking")
                        break

                    # resize img
                    img = frame.copy()
                    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 288, 288)
                    inp_img = tf.cast(img, dtype=tf.int32)

                    #Detection section
                    results = self.movenet(inp_img)
                    keypoints_with_scores = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))

                    #Render Keypoints
                    self.loop_through_people(frame, keypoints_with_scores, self.EDGES, 0.2)

                    if frame_num == 0 :
                        cv2.putText(frame,action + "  " + "Collecting", (150,300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, f'Collecting frames for {action} Video file {vid_no}.mp4', (100,100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(500)
                    else :
                        cv2.putText(frame, f'Collecting frames for {action} Video Number {vid_no}.mp4', (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)

                    # NEW Export keypoints
                    npy_path = os.path.join(support.DATA_PATH, action, str(vid_no), str(frame_num))
                    np.save(npy_path, keypoints_with_scores)
                    # Break gracefully
                    if (cv2.waitKey(10) & 0xFF == ord('q')) or br:
                        br = True
                        break
                if br :
                    break
            if br :
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
    def prob_viz(self, res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        return output_frame
        
        
    async def client(self, mssg) :
            uri = "ws://0.tcp.in.ngrok.io:12672"
            async with websockets.connect(uri) as websocket :
                
                await websocket.send(mssg)  #str(res["probability"])
                print(f"Client sent: {mssg}")
                # estimator.predict_from_video("./vids/drowning/5.mp4", func)
                
                server_res = await websocket.recv()
                print(f"Client received: {server_res}")
        
        
    def predict_from_video(self, vid=None, emit_func=False) :
        # 1. New detection variables
        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5
        model = lstm.load_lstm_model("action.h5")

        if vid :
            cap = cv2.VideoCapture(vid)
        else :
            cap = cv2.VideoCapture(vid)
        # Set mediapipe model 
        while cap.isOpened():
            print("READING FRAMES")
            # Read feed
            ret, frame = cap.read()
            if ret :
                frame = cv2.resize(frame, (800, 800))
            else:
                print("Breaking")
                break

            # resize img
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 288, 288)
            inp_img = tf.cast(img, dtype=tf.int32)

            #Detection section
            results = self.movenet(inp_img)
            keypoints_with_scores = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))
            #Render Keypoints
            self.loop_through_people(frame, keypoints_with_scores, self.EDGES, 0.2)

            keypoints = support.convert_to_1d(keypoints_with_scores)
            sequence.append(keypoints)
            sequence = sequence[-80:]

            if len(sequence) == 80:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(support.actions[np.argmax(res)])
                predictions.append(np.argmax(res))


            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 

                        if len(sentence) > 0: 
                            if support.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(support.actions[np.argmax(res)])
                        else:
                            sentence.append(support.actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                frame = self.prob_viz(res, support.actions, frame, self.viz_colors)
                
                if emit_func==True :
                    asyncio.run(self.client(str(100 * res[0])))

            cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', frame)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        

lstm = LSTMModel()
support = SupportFunctions()

if __name__ == "__main__" :
    pass