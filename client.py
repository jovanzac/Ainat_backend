import asyncio
import websockets

from Utils.opencv import VideoPoseEstimation

async def hello() :
    uri = "ws://0.tcp.in.ngrok.io:17280"
    async with websockets.connect(uri) as websocket :
        
        func = lambda mssg : await websocket.send(mssg)  #str(res["probability"])
        estimator.predict_from_video("./vids/drowning/5.mp4", func)
        
        server_res = await websocket.recv()
        print(f"Client received: {server_res}")
        
        
estimator = VideoPoseEstimation()

if __name__ == "__main__" :
    asyncio.run(hello())