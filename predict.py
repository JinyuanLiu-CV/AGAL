import time
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
save_dir = os.path.join("./result/")
# test
print("-------dataset size:   ",len(dataset))
flag = 0.0
k = 0
for i, data in enumerate(dataset):
    model.set_input(data)
    start_time = time.time()
    output = model.predict()
    temp = time.time() - start_time

    img_path = model.get_image_paths()
    filename = 'img_{:03d}.jpg'.format(i+1)
    print('process image... %s' % img_path)
    image_pil = Image.fromarray(output)
    s_path = os.path.join(save_dir,filename)
    image_pil.save(s_path)
    if(k>=1):
        flag = temp + flag
        print('cost time: %.3f' % (temp))
    k = k + 1
k=k-1

print('average time: %.3f' % (flag/k))