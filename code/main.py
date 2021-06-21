import tensorflow as tf
import os

from absl import flags
from absl import app

from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers import Adam


from utils import generate_color
from dataset import load_pascal_voc_dataset
from train import save_validation_result
from learning_env_setting import dir_setting, save_tensorboard_log, save_checkpoint
from train import train_step


# set cat label dictionary (object detection에서 자주 사용되는 dict 패턴)
# computer가 인지한 숫자를 사람이 알아볼 수 있게 key는 integer, value는 string 으로 set
cat_label_dict = {
  0: "cat"
}
cat_class_to_label_dict = {v: k for k, v in cat_label_dict.items()}
# 위의 cat_label_dict의 key와 value의 위치(역할)을 바꾼 dict
# 해당 code에서는 cat에 대한 class만 classification할 것이기 때문에 cat만 set


#flags instance로 hyper parameters setting
flags.DEFINE_string('checkpoint_path', default='saved_model', help='path to a directory to save model checkpoints during training')
flags.DEFINE_integer('save_checkpoint_steps', default=50, help='period at which checkpoints are saved (defaults to every 50 steps)')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log', help='path to a directory to save tensorboard log')
flags.DEFINE_integer('validation_steps', default=50, help='period at which test prediction result and save image')
# 몇 번의 step마다 validation data로 test를 할지 결정
flags.DEFINE_integer('num_epochs', default=135, help='training epochs') # original paper : 135 epoch
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate') # original paper : 0.001 (1epoch) -> 0.01 (75epoch) -> 0.001 (30epoch) -> 0.0001 (30epoch)
flags.DEFINE_float('lr_decay_rate', default=0.5, help='decay rate for the learning rate')
flags.DEFINE_integer('lr_decay_steps', default=2000, help='number of steps after which the learning rate is decayed by decay rate') # 2000번 마다 init_learning_rate * lr_decay_rate 을 실행
# 2000 step : init_learning_rate = 0.00005, 4000 step : init_learning_rate = 0.000025
flags.DEFINE_integer('num_visualize_image', default=8, help='number of visualize image for validation')
# 중간중간 validation을 할 때마다 몇 개의 batch size로 visualization을 할지 결정하는 변수

FLAGS = flags.FLAGS

# set configuration value
batch_size = 24 # original paper : 64
input_width = 224 # original paper : 448
input_height = 224 # original paper : 448
cell_size = 7
num_classes = 1 # original paper : 20
boxes_per_cell = 2

# set color_list for drawing
color_list = generate_color(num_classes)

# set loss function coefficients
coord_scale = 10 # original paper : 5  
class_scale = 0.1  # original paper : 1
object_scale = 1	# original paper : None
noobject_scale = 0.5	# original paper : None

train_data, validation_data = load_pascal_voc_dataset(batch_size)

# set learning rate decay
lr_schedule = schedules.ExponentialDecay(
    FLAGS.init_learning_rate,
    decay_steps=FLAGS.lr_decay_steps,
    decay_rate=FLAGS.lr_decay_rate,
    staircase=True)
    # learning rate detail을 결정. 0.0001에서 2000번 마다 0.5씩 곱
    # default steps = 2000, decay_rate = 0.5
    # initail learning rate = 0.0001

# set optimizer
optimizer = Adam(lr_schedule) 
# original paper에서는 
# optimizer = tf.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 0.0005)

# check if checkpoint path exists

ckpt, ckpt_manager = dir_setting(checkpoint_path, input_height, input_width, cell_size
				    			 boxes_per_cell, num_classes)

 # set tensorboard log
# tensorboard_log를 write하기 위한 writer instance 만들기
train_summary_writer = tf.summary.create_file_writer(FLAGS.tensorboard_log_path +  '/train')   
validation_summary_writer = tf.summary.create_file_writer(FLAGS.tensorboard_log_path +  '/validation')

for epoch in range(FLAGS.num_epochs):
    num_batch = len(list(train_data))
    for iter, features in enumerate(train_data):
        batch_image = features['image']
        batch_bbox = features['objects']['bbox']
        batch_labels = features['objects']['label']

        batch_image = tf.squeeze(batch_image, axis=1)
        # dummy dimension을 삭제
        batch_bbox = tf.squeeze(batch_bbox, axis=1)
        batch_labels = tf.squeeze(batch_labels, axis=1)

        # run optimization and compute loss
        
        total_loss, coord_loss, object_loss, noobject_loss, class_loss = train_step(optimizer,
                                                                                  YOLOv1_model,
                                                                                  batch_image,
                                                                                  batch_bbox,
                                                                                  batch_labels)

        # print log
        print("Epoch: %d, Iter: %d/%d, Loss: %f" % ((epoch+1), (iter+1), num_batch, total_loss.numpy()))

        # save tensorboard log
		save_tensorboard_log(train_summary_writer, optimizer, total_loss, coord_loss, object_loss, noobject_loss, class_loss, ckpt)

		# save checkpoint
		save_checkpoint(ckpt, FLAGS.save_checkpoint_steps):


        ckpt.step.assign_add(1) # epoch나 train data의 개수와는 별개로, step 증가

        # occasionally check validation data and save tensorboard log
        # 반복이 validation_steps에 도달하면, 현재 step의 기준으로 model의 parameter에 기반한 validation을 진행
    
        if iter % FLAGS.validation_steps == 0:
            save_validation_result(YOLOv1_model, ckpt, validation_summary_writer, FLAGS.num_visualize_image, validation_data)

if __name__ == '__main__':  # entry point 지정
    app.run(main)
