**CONTINUE_LEARNING**

training이 unintentionally하게 중단되었거나, data의 update로 인해 model을 새롭게 training해야 하는 경우와

이전에 진행했던 training을 이어서 진행하는 경우에 대해서 결정하기 위해 continue여부에 대한 flag를 설정



main.py에 위치한 `CONTINUE_LEARNING` flag의 True, False 여부에 따라 directory생성, 삭제, load saved model 등의 동작이 이루어진다.

- `CONTINUE_LEARNING = False` : 이전에 했던 training을 다시 시작하는 경우

- `CONTINUE_LEARNING = True `: 이전에 했던 training의 step에 이어서 진행 할 경우



해당 동작은 utilys.py에 define된 `dir_setting` function에서 정의하였다.



**dir_setting**

- directory path 정의
- set tensorboard log instance를 선언
- CONTINUE_LEARNING의 flag에 따른 training continue여부 결정

```python
def dir_setting(dir_name, 
				CONTINUE_LEARNING, 
				checkpoint_path, 
				tensorboard_log_path):

	model_path = os.path.join(os.getcwd() , dir_name)
	checkpoint_path = os.path.join(model_path, checkpoint_path)
	tensorboard_log_path = os.path.join(model_path, tensorboard_log_path)

	if CONTINUE_LEARNING == True and not os.path.isdir(model_path):
		CONTINUE_LEARNING = False
		print("CONTINUE_LEARNING flag has been converted to FALSE") 

	if CONTINUE_LEARNING == False and os.path.isdir(model_path):
		while True:
			print("\n Are you sure remove all directory and file for new training start?  [Y/N] \n")
			answer = str(input())
			if answer == 'Y' or answer == 'y':
				shutil.rmtree(model_path)
				break
			elif answer == 'N' or answer == 'n':
				print("Check 'CONTINUE_LEARNING' in main.py")
				sys.exit()
			else :
				print("wrong answer. \n Please enter any key ")
				tmp = str(input())
				os.system('clear')  # cls in window 

	# set tensorboard log
	train_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/train')
	validation_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/validation')  

	# pass if the path exist. or not, create directory on path
	if not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)
		os.mkdir(checkpoint_path)


	return checkpoint_path, train_summary_writer, validation_summary_writer
```

