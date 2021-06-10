import cv2 
import numpy as np

class ImgProcessor:
    def __init__(self, gray_img, img_width, img_height):
        self.gray_img = gray_img
        self.img_width = img_width
        self.img_height = img_height
    
    def convert_img(self, img):
        img = cv2.resize(img, dsize=(self.img_width, self.img_height))
        if self.gray_img:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2,0,1)
        return img
    
class GifMaker:
    def __init__(self, env):
        self.env = env
        self.now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
    def make_gif(self, agent, step, algo_name, env_name):
        gif_path = './gif/' + env_name + '/' + algo_name + '/' + self.now + '/'
        
        os.makedirs("./images4gif", exist_ok=True)
        os.makedirs(gif_path, exist_ok=True)
        
        count = 0
        
        done = False
#         state = self.env.reset_raw()
        state = self.env.reset()
        
        state_frame = state[:,-1,:,:]
        state_frame = np.reshape(state_frame, (state_frame.shape[1], state_frame.shape[2], 1))
        
        count_str = str(count).zfill(4)
        cv2.imwrite('./images4gif/'+count_str+'.jpg', state_frame)
        
        while not done:
            action = agent.act(state, training=False) #하 state.... 
            state, reward, done = self.env.step(np.array([1]))
#             state, reward, done = self.env.step_raw(action)
            
            count += 1
            
            state_frame = state[:,-1,:,:]
            state_frame = np.reshape(state_frame, (state_frame.shape[1], state_frame.shape[2], 1))

            count_str = str(count).zfill(4)
            cv2.imwrite('./images4gif/'+count_str+'.jpg', state_frame)

        # Make gif 
        path = [f"./images4gif/{i}" for i in os.listdir("./images4gif")]
        paths = [Image.open(i) for i in path]
        
        imageio.mimsave(gif_path + str(step)+'.gif', paths, fps=20)
        
        print("=================== gif file is saved at {} ===================".format(gif_path))
        shutil.rmtree("./images4gif")
