# learing_env_setting



code

```python
import tensorflow as tf
import os
from model import YOLOv1

def dir_setting(checkpoint_path, input_height, input_width, cell_size
				boxes_per_cell, num_classes): 
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # create YOLO model
    YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)

    # set checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
    # step 설정은 초기 train entry point
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                            directory=checkpoint_path,
                                            max_to_keep=None)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    # latest_checkpoint : 마지막 checkpoint에서 저장된 file의 path를 return 

    # restore latest checkpoint
    # 마지막 checkpoint의 값들을 ckpt에 저장
    if latest_ckpt:
        ckpt.restore(latest_ckpt)
        print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))
    return ckpt, ckpt_manager

def save_tensorboard_log(train_summary_writer, optimizer, total_loss,
						 coord_loss, object_loss, noobject_loss, class_loss, ckpt):
    # 현재 시점의 step의 각 loss값을 write
    with train_summary_writer.as_default():
        tf.summary.scalar('learning_rate ', optimizer.lr(ckpt.step).numpy(), step=int(ckpt.step))
        tf.summary.scalar('total_loss', total_loss, step=int(ckpt.step))
        tf.summary.scalar('coord_loss', coord_loss, step=int(ckpt.step))
        tf.summary.scalar('object_loss ', object_loss, step=int(ckpt.step))
        tf.summary.scalar('noobject_loss ', noobject_loss, step=int(ckpt.step))
        tf.summary.scalar('class_loss ', class_loss, step=int(ckpt.step))

def save_checkpoint(ckpt, FLAGS.save_checkpoint_steps):
    # save checkpoint
    # ckpt.step이 FLAGS.save_checkpoint_steps에 도달 할 때마다
    if ckpt.step % FLAGS.save_checkpoint_steps == 0:
        ckpt_manager.save(checkpoint_number=ckpt.step)  # CheckpointManager의 parameter 저장
        print('global_step : {}, checkpoint is saved!'.format(int(ckpt.step)))
```



